from dataclasses import dataclass
from typing import NewType
import numpy as np
import pandas as pd
from datetime import datetime
from injector import inject

# エアーコンディショナー
import pyhees.section4_3 as rac
# 床下
import pyhees.section3_1 as ld
import pyhees.section3_1_d as uf
import pyhees.section3_1_e as algo
# ダクト式セントラル空調機
from pyhees.section4_1 import get_alpha_UT_H_A
import pyhees.section4_2 as dc
import pyhees.section4_2_a as dc_a

""" JJJ_EXPERIMENT OVERRIDE """

""" JJJ_EXPERIMENT ORIGINAL """
from jjjexperiment.common import *
import jjjexperiment.constants as jjj_consts
from jjjexperiment.logger import LimitedLoggerAdapter as _logger  # デバッグ用ロガー
from jjjexperiment.inputs.options import *
# データクラス
from jjjexperiment.inputs.common import HouseInfo, OuterSkin
from jjjexperiment.inputs.ac_setting import HeatingAcSetting, CoolingAcSetting
from jjjexperiment.inputs.heating import CRACSpecification as HeatCRACSpec
from jjjexperiment.inputs.cooling import CRACSpecification as CoolCRACSpec
from jjjexperiment.inputs.di_container import ClimateFile, CaseName
# ドメインサービス
from jjjexperiment.inputs.climate_service import ClimateService
from jjjexperiment.inputs.ac_quantity_service import HeatQuantityService, CoolQuantityService
# F23-1 Vサプライの上限キャップ変更
from jjjexperiment.v_supply_cap.inputs.v_supply_cap_dto import VSupplyCapDto
import jjjexperiment.v_supply_cap.cap_V_supply_d_t_i as jjj_vsupcap
# F24-4 過剰熱量繰越
from jjjexperiment.carryover_heat.inputs.carryover_heat_dto import CarryoverHeatDto
import jjjexperiment.carryover_heat as jjj_carryover_heat
# F24-5 新床下空調
import jjjexperiment.underfloor_ac.section4_2 as jjj_ufac_dc
from jjjexperiment.underfloor_ac.section3_1_e import (
    calc_Theta_uf_d_t_2023,
    calc_sum_Theta_dash_g_surf_A_m_d_t,
)
from jjjexperiment.underfloor_ac.section4_2_f52 import get_Theta_star_NR
from jjjexperiment.underfloor_ac.section4_2_f46_f48 import get_Theta_HBR_i, get_Theta_NR
from jjjexperiment.underfloor_ac.inputs.common import UnderfloorAc, UfVarsDataFrame
# F25-1 最小風量・最低電力直接入力
from jjjexperiment.v_min_input.logic import rescale_V_vent_g_i
from jjjexperiment.v_min_input.inputs.heating import InputMinVolumeInput as HeatMinVolumeInput
from jjjexperiment.v_min_input.inputs.cooling import InputMinVolumeInput as CoolMinVolumeInput

@dataclass
class Load_DTI:
    """時間ステップ毎の負荷データ"""
    L_H_d_t_i: Array5x8760
    """暖房負荷 [MJ/h]"""
    L_CS_d_t_i: Array5x8760
    """冷房顕熱負荷 [MJ/h]"""
    L_CL_d_t_i: Array5x8760
    """冷房潜熱負荷 [MJ/h]"""
    L_dash_H_R_d_t_i: Array5x8760
    """標準住戸の負荷補正前の 暖房負荷 [MJ/h]"""
    L_dash_CS_R_d_t_i: Array5x8760
    """標準住戸の負荷補正前の 冷房顕熱負荷 [MJ/h]"""

# 型解決用エイリアス
VHS_DSGN_H = NewType('VHS_DSGN_H', float)
VHS_DSGN_C = NewType('VHS_DSGN_C', float)

# NOTE: クライアントコード側で切り替える(bind)するためのギミック
@dataclass
class ActiveAcSetting:
    load: HeatingAcSetting | CoolingAcSetting


@dataclass
class AnnualGroundFeedbackContext:
    """暖房・冷房で共有する年間の実床下温度履歴。"""
    heat_ac_setting: HeatingAcSetting
    cool_ac_setting: CoolingAcSetting
    V_hs_dsgn_H: float
    V_hs_dsgn_C: float
    Theta_uf_feedback_d_t: np.ndarray | None = None


def limit_corrected_heating_output(
        Q_hat_hs_adjusted_d_t: np.ndarray) -> np.ndarray:
    """Apply the heating-output lower bound after all underfloor corrections."""
    return np.clip(Q_hat_hs_adjusted_d_t, 0, None)


def combine_corrected_cooling_output(
        Q_hat_hs_base_d_t: np.ndarray,
        Q_hat_hs_CS_base_d_t: np.ndarray,
        Q_hat_hs_adjusted_d_t: np.ndarray) -> np.ndarray:
    """床下による顕熱補正後の出力へ、補正前の潜熱出力を加える。"""
    delta_Q_hat_hs_CS_d_t = Q_hat_hs_adjusted_d_t - Q_hat_hs_base_d_t
    Q_hat_hs_CS_adjusted_d_t = np.clip(
        Q_hat_hs_CS_base_d_t + delta_Q_hat_hs_CS_d_t, 0, None
    )
    Q_hat_hs_CL_base_d_t = np.clip(
        Q_hat_hs_base_d_t - Q_hat_hs_CS_base_d_t, 0, None
    )
    return Q_hat_hs_CS_adjusted_d_t + Q_hat_hs_CL_base_d_t


def get_appendix_e_ground_parameters(
        region: int,
        Q: float,
        Theta_ex_d_t: np.ndarray) -> tuple[float, float, float]:
    """3章付録E(4)(5)(10)から床・基礎・地盤の値を算定する。"""
    return (
        algo.get_U_s_vert(region, Q),
        algo.get_phi(region, Q),
        algo.get_Theta_g_avg(Theta_ex_d_t),
    )


# 地盤応答へ実際の床下温度を反映する反復計算の設定。
# 現行の計算は8760時間をベクトル計算するため、1回目で実際の床下温度を求め、
# 2回目以降の式(40)ではその系列の前時刻値を地盤応答へ渡す。
GROUND_FEEDBACK_MAX_ITERATIONS = 24
GROUND_FEEDBACK_TOLERANCE = 1.0e-4


def merge_annual_floor_temperature(
        Theta_uf_H_d_t: np.ndarray,
        Theta_uf_C_d_t: np.ndarray,
        Theta_uf_M_d_t: np.ndarray,
        H: np.ndarray,
        C: np.ndarray) -> np.ndarray:
    """暖房期・冷房期・中間期を一本の実床下温度系列へ統合する。"""
    Theta_uf_feedback_d_t = np.where(
        H,
        Theta_uf_H_d_t,
        np.where(C, Theta_uf_C_d_t, Theta_uf_M_d_t),
    )
    assert Theta_uf_feedback_d_t.shape == (24 * 365,)
    return Theta_uf_feedback_d_t


def calc_intermediate_floor_temperature(
        V_dash_supply_1F_d_t: np.ndarray,
        Theta_star_HBR_d_t: np.ndarray,
        Theta_ex_d_t: np.ndarray,
        sum_Theta_dash_g_surf_A_m_d_t: np.ndarray,
        U_s_supply: float,
        A_s_ufac_A: float,
        phi: float,
        L_uf: float,
        R_g: float,
        Phi_A_0: float,
        Theta_g_avg: float) -> np.ndarray:
    """中間期の式(21)床下温度を、Excelと同じ熱収支で求める。"""
    C_sa_d_t = (
        algo.get_ro_air() * algo.get_c_p_air() * V_dash_supply_1F_d_t
    )
    K1 = U_s_supply * A_s_ufac_A
    K2 = phi * L_uf
    K3 = (A_s_ufac_A / R_g) / (1.0 + Phi_A_0 / R_g)
    return (
        C_sa_d_t * Theta_star_HBR_d_t
        + (
            K1 * Theta_star_HBR_d_t
            + K2 * Theta_ex_d_t
            + K3 * (sum_Theta_dash_g_surf_A_m_d_t + Theta_g_avg)
        ) * 3.6
    ) / (C_sa_d_t + (K1 + K2 + K3) * 3.6)


def _calc_Q_UT_A_once(
        case_name: CaseName,
        climateFile: ClimateFile,
        house: HouseInfo,
        ac_setting: ActiveAcSetting,
        skin: OuterSkin,
        heat_CRAC: HeatCRACSpec,
        cool_CRAC: CoolCRACSpec,
        new_ufac: UnderfloorAc,
        new_ufac_df: UfVarsDataFrame,
        v_min_heat_input: HeatMinVolumeInput,
        v_min_cool_input: CoolMinVolumeInput,
        V_hs_dsgn_H: VHS_DSGN_H,
        V_hs_dsgn_C: VHS_DSGN_C,
        v_supply_cap_dto: VSupplyCapDto,
        carryover_heat_dto: CarryoverHeatDto,
        load: Load_DTI,
        Theta_uf_ground_feedback_d_t: np.ndarray | None = None,
        export_outputs: bool = True):
    """未処理負荷と機器の計算に必要な変数を取得"""

    # NOTE: 暖房・冷房で二回実行される。q_hs_rtd_H, q_hs_rtd_C のどちらが None かで判別している
    def flg_char() -> str:
        match ac_setting:
            case HeatingAcSetting(): return '_H'
            case CoolingAcSetting(): return '_C'
            case _: raise ValueError
    def q_hs_rtd_H() -> float | None:
        match ac_setting:
            case HeatingAcSetting(): return HeatQuantityService(ac_setting, house.region, house.A_A).q_hs_rtd
            case CoolingAcSetting(): return None
            case _: raise ValueError
    def q_hs_rtd_C() -> float | None:
        match ac_setting:
            case HeatingAcSetting(): return None
            case CoolingAcSetting(): return CoolQuantityService(ac_setting, house.region, house.A_A).q_hs_rtd
            case _: raise ValueError

    match V_hs_dsgn_H, V_hs_dsgn_C:
        case 0, _:
            V_hs_dsgn_H = None
        case _, 0:
            V_hs_dsgn_C = None
        case _:
            raise ValueError("暖房・冷房の判別がつかない")

    df_output  = pd.DataFrame(index = pd.date_range(datetime(2023,1,1,1,0,0), datetime(2024,1,1,0,0,0), freq='h'))
    df_output2 = pd.DataFrame()
    df_output3 = pd.DataFrame()

    # 熱繰越調査用出力ファイル
    df_carryover_output  = pd.DataFrame(index = pd.date_range(datetime(2023,1,1,1,0,0), datetime(2024,1,1,0,0,0), freq='h'))

    # 気象条件
    climate = ClimateService(house.region, new_ufac, climateFile)
    Theta_ex_d_t = climate.get_Theta_ex_d_t()
    X_ex_d_t = climate.get_X_ex_d_t()
    J_d_t = climate.get_J_d_t()
    h_ex_d_t = climate.get_h_ex_d_t()

    df_output['Theta_ex_d_t']  = Theta_ex_d_t
    df_output['X_ex_d_t']      = X_ex_d_t
    df_output['J_d_t']    = J_d_t
    df_output['h_ex_d_t'] = h_ex_d_t

    #主たる居室・その他居室・非居室の面積
    A_HCZ_i = np.array([ld.get_A_HCZ_i(i, house.A_A, house.A_MR, house.A_OR) for i in range(1, 6)])
    A_HCZ_R_i = np.array([ld.get_A_HCZ_R_i(i) for i in range(1, 6)])
    A_NR = ld.get_A_NR(house.A_A, house.A_MR, house.A_OR)

    df_output2['A_HCZ_i'] = A_HCZ_i
    df_output2['A_HCZ_R_i'] = A_HCZ_R_i
    df_output3['A_NR'] = [A_NR]

    # (67)  水の蒸発潜熱
    L_wtr = dc.get_L_wtr()
    df_output3['L_wtr'] = [L_wtr]

    # (66d)　非居室の在室人数
    n_p_NR_d_t = dc.calc_n_p_NR_d_t(A_NR)
    df_output['n_p_NR_d_t'] = n_p_NR_d_t
    # (66c)　その他居室の在室人数
    n_p_OR_d_t = dc.calc_n_p_OR_d_t(house.A_OR)
    df_output['n_p_OR_d_t'] = n_p_OR_d_t
    # (66b)　主たる居室の在室人数
    n_p_MR_d_t = dc.calc_n_p_MR_d_t(house.A_MR)
    df_output['n_p_MR_d_t'] = n_p_MR_d_t
    # (66a)　在室人数
    n_p_d_t = dc.get_n_p_d_t(n_p_MR_d_t, n_p_OR_d_t, n_p_NR_d_t)
    df_output['n_p_d_t'] = n_p_d_t

    # 人体発熱
    q_p_H = dc.get_q_p_H()
    q_p_CS = dc.get_q_p_CS()
    q_p_CL = dc.get_q_p_CL()
    df_output3['q_p_H'] = [q_p_H]
    df_output3['q_p_CS'] = [q_p_CS]
    df_output3['q_p_CL'] = [q_p_CL]

    # (65d)　非居室の内部発湿
    w_gen_NR_d_t = dc.calc_w_gen_NR_d_t(A_NR)
    df_output['w_gen_NR_d_t'] = w_gen_NR_d_t
    # (65c)　その他居室の内部発湿
    w_gen_OR_d_t = dc.calc_w_gen_OR_d_t(house.A_OR)
    df_output['w_gen_OR_d_t'] = w_gen_OR_d_t
    # (65b)　主たる居室の内部発湿
    w_gen_MR_d_t = dc.calc_w_gen_MR_d_t(house.A_MR)
    df_output['w_gen_MR_d_t'] = w_gen_MR_d_t
    # (65a)　内部発湿
    w_gen_d_t = dc.get_w_gen_d_t(w_gen_MR_d_t, w_gen_OR_d_t, w_gen_NR_d_t)
    df_output['w_gen_d_t'] = w_gen_d_t

    # (64d)　非居室の内部発熱
    q_gen_NR_d_t = dc.calc_q_gen_NR_d_t(A_NR)
    df_output['q_gen_NR_d_t'] = q_gen_NR_d_t
    # (64c)　その他居室の内部発熱
    q_gen_OR_d_t = dc.calc_q_gen_OR_d_t(house.A_OR)
    df_output['q_gen_OR_d_t'] = q_gen_OR_d_t
    # (64b)　主たる居室の内部発熱
    q_gen_MR_d_t = dc.calc_q_gen_MR_d_t(house.A_MR)
    df_output['q_gen_MR_d_t'] = q_gen_MR_d_t
    # (64a)　内部発熱
    q_gen_d_t = dc.get_q_gen_d_t(q_gen_MR_d_t, q_gen_OR_d_t, q_gen_NR_d_t)
    df_output['q_gen_d_t'] = q_gen_d_t

    # (63)　局所排気量
    V_vent_l_NR_d_t = dc.get_V_vent_l_NR_d_t()
    V_vent_l_OR_d_t = dc.get_V_vent_l_OR_d_t()
    V_vent_l_MR_d_t = dc.get_V_vent_l_MR_d_t()
    V_vent_l_d_t = dc.get_V_vent_l_d_t(V_vent_l_MR_d_t, V_vent_l_OR_d_t, V_vent_l_NR_d_t)
    df_output = df_output.assign(
        V_vent_l_NR_d_t = V_vent_l_NR_d_t,
        V_vent_l_OR_d_t = V_vent_l_OR_d_t,
        V_vent_l_MR_d_t = V_vent_l_MR_d_t,
        V_vent_l_d_t = V_vent_l_d_t
    )

    match ac_setting:
        case HeatingAcSetting(): v_min_input = v_min_heat_input
        case CoolingAcSetting(): v_min_input = v_min_cool_input
        case _: raise ValueError

    # (62)　全般換気量
    if v_min_input.input_V_hs_min == 最低風量直接入力.入力する:  # ダックタイピング
        # 最低風量指定を満たすように調整
        V_vent_g_i = rescale_V_vent_g_i(
            dc.get_V_vent_g_i(A_HCZ_i, A_HCZ_R_i),  # 従来式
            v_min_input.V_hs_min)
    else:
        V_vent_g_i = dc.get_V_vent_g_i(A_HCZ_i, A_HCZ_R_i)  # 従来式
    df_output2['V_vent_g_i'] = V_vent_g_i

    # (61)　間仕切の熱貫流率
    U_prt = dc.get_U_prt()
    df_output3['U_prt'] = [U_prt]

    # (60)　非居室の間仕切の面積
    A_prt_i = dc.get_A_prt_i(A_HCZ_i, skin.r_env, house.A_MR, A_NR, house.A_OR)
    df_output3['r_env'] = [skin.r_env]
    df_output2['A_prt_i'] = A_prt_i

    # (59)　等価外気温度
    Theta_SAT_d_t = dc.get_Theta_SAT_d_t(Theta_ex_d_t, J_d_t)
    df_output['Theta_SAT_d_t'] = Theta_SAT_d_t

    # (58)　断熱区画外を通るダクトの長さ
    l_duct_ex_i = dc.get_l_duct_ex_i(house.A_A)
    df_output2['l_duct_ex_i'] = l_duct_ex_i

    # (57)　断熱区画内を通るダクト長さ
    l_duct_in_i = dc.get_l_duct_in_i(house.A_A)
    df_output2['l_duct_in_i'] = l_duct_in_i

    # (56)　ダクト長さ
    l_duct_i = dc.get_l_duct__i(l_duct_in_i, l_duct_ex_i)
    df_output2['l_duct_i'] = l_duct_i

    # (51)　負荷バランス時の居室の絶対湿度
    X_star_HBR_d_t = dc.get_X_star_HBR_d_t(X_ex_d_t, house.region)  # X_ex_d_t [g/kg(DA)] 想定
    df_output['X_star_HBR_d_t'] = X_star_HBR_d_t

    # (50)　負荷バランス時の居室の室温
    Theta_star_HBR_d_t = dc.get_Theta_star_HBR_d_t(Theta_ex_d_t, house.region)
    df_output['Theta_star_HBR_d_t'] = Theta_star_HBR_d_t

    # (55)　小屋裏の空気温度
    Theta_attic_d_t = dc.get_Theta_attic_d_t(Theta_SAT_d_t, Theta_star_HBR_d_t)
    df_output['Theta_attic_d_t'] = Theta_attic_d_t

    # (54)　ダクトの周囲の空気温度
    Theta_sur_d_t_i = dc.get_Theta_sur_d_t_i(Theta_star_HBR_d_t, Theta_attic_d_t, l_duct_in_i, l_duct_ex_i, ac_setting.duct_insulation)
    df_output = df_output.assign(
        Theta_sur_d_t_i_1 = Theta_sur_d_t_i[0],
        Theta_sur_d_t_i_2 = Theta_sur_d_t_i[1],
        Theta_sur_d_t_i_3 = Theta_sur_d_t_i[2],
        Theta_sur_d_t_i_4 = Theta_sur_d_t_i[3],
        Theta_sur_d_t_i_5 = Theta_sur_d_t_i[4]
    )

    # (40)-1st 熱源機の風量を計算するための熱源機の出力
    Q_hat_hs_d_t, Q_hat_hs_CS_d_t = dc.calc_Q_hat_hs_d_t(skin.Q, house.A_A, V_vent_l_d_t, V_vent_g_i, skin.mu_H, skin.mu_C, J_d_t, q_gen_d_t, n_p_d_t, q_p_H,
                                     q_p_CS, q_p_CL, X_ex_d_t, w_gen_d_t, Theta_ex_d_t, L_wtr, house.region)
    df_output['Q_hat_hs_d_t'] = Q_hat_hs_d_t
    # 式(40)の床下補正前の出力。1階負荷の面積按分にはこの値を用いる。
    Q_hat_hs_base_d_t = Q_hat_hs_d_t.copy()
    Q_hat_hs_CS_base_d_t = Q_hat_hs_CS_d_t.copy()

    # (39)　熱源機の最低風量
    V_hs_min = dc.get_V_hs_min(V_vent_g_i)
    df_output3['V_hs_min'] = [V_hs_min]

    ####################################################################################################################
    if ac_setting.type in [
            計算モデル.ダクト式セントラル空調機,
            計算モデル.RAC活用型全館空調_潜熱評価モデル
        ]:
        # (38)
        Q_hs_rtd_C = dc.get_Q_hs_rtd_C(q_hs_rtd_C())
        # (37)
        Q_hs_rtd_H = dc.get_Q_hs_rtd_H(q_hs_rtd_H())
    elif ac_setting.type in [
            計算モデル.RAC活用型全館空調_現行省エネ法RACモデル,
            計算モデル.電中研モデル
        ]:
        # (38)　冷房時の熱源機の定格出力
        Q_hs_rtd_C = dc.get_Q_hs_rtd_C(cool_CRAC.q_rtd)  #ルームエアコンディショナの定格能力 q_rtd_C を入力するよう書き換え
        # (37)　暖房時の熱源機の定格出力
        Q_hs_rtd_H = dc.get_Q_hs_rtd_H(heat_CRAC.q_rtd)  #ルームエアコンディショナの定格能力 q_rtd_H を入力するよう書き換え
    else:
        raise Exception('設備機器の種類の入力が不正です。')

    df_output3['Q_hs_rtd_C'] = [Q_hs_rtd_C]
    df_output3['Q_hs_rtd_H'] = [Q_hs_rtd_H]
    ####################################################################################################################

    match ac_setting:
        case HeatingAcSetting(): Theta_in_d_t = uf.get_Theta_in_d_t('H')
        case CoolingAcSetting(): Theta_in_d_t = uf.get_Theta_in_d_t('CS')
        case _: raise ValueError

    # 吸熱応答係数の初項
    Phi_A_0 = 0.025504994
    # 3章付録E(4)(5)(10)から算定し、入力された丸め値は使用しない。
    U_s_load, phi, Theta_g_avg = get_appendix_e_ground_parameters(
        house.region, skin.Q, Theta_ex_d_t
    )
    # The underfloor correction below needs the season masks even when CAV is off.
    H, C, M = dc.get_season_array_d_t(house.region)
    # 脱出条件:
    should_be_adjusted_Q_hat_hs_d_t = new_ufac.new_ufac_flg == 床下空調ロジック.変更する
    while True:
        # (36)　VAV 調整前の熱源機の風量
        if skin.hs_CAV:
            H, C, M = dc_a.get_season_array_d_t(house.region)
            V_dash_hs_supply_d_t = np.zeros(24 * 365)
            V_dash_hs_supply_d_t[H] = V_hs_dsgn_H or 0
            V_dash_hs_supply_d_t[C] = V_hs_dsgn_C or 0
            V_dash_hs_supply_d_t[M] = 0
        else:
            if ac_setting.type == 計算モデル.RAC活用型全館空調_潜熱評価モデル:
                # FIXME: 方式3が他方式と比較して大きくなる問題
                match (Q_hs_rtd_H, Q_hs_rtd_C):
                    case (None, None):
                        raise Exception("どちらかのみを想定")
                    case (_, None):  # 暖房期(=q_hs_rtd_H) => 全熱負荷
                        V_dash_hs_supply_d_t = dc.get_V_dash_hs_supply_d_t_2023(Q_hat_hs_d_t, house.region, False)
                    case (None, _):  # 冷房期: 式(40-2a)の顕熱＋潜熱出力を使用
                        V_dash_hs_supply_d_t = dc.get_V_dash_hs_supply_d_t_2023(Q_hat_hs_d_t, house.region, True)
                    case (_, _):
                        raise Exception("どちらかのみを想定")

                df_output['V_dash_hs_supply_d_t'] = V_dash_hs_supply_d_t
            else:
                updated_V_hs_dsgn_H = V_hs_dsgn_H or 0 if Q_hs_rtd_H is not None  \
                        else None
                updated_V_hs_dsgn_C = V_hs_dsgn_C or 0 if Q_hs_rtd_C is not None  \
                    else None

                V_dash_hs_supply_d_t = \
                    dc.get_V_dash_hs_supply_d_t(V_hs_min, updated_V_hs_dsgn_H, updated_V_hs_dsgn_C, Q_hs_rtd_H, Q_hs_rtd_C, Q_hat_hs_d_t, house.region)
                df_output['V_dash_hs_supply_d_t'] = V_dash_hs_supply_d_t

        if ac_setting.VAV and jjj_consts.change_supply_volume_before_vav_adjust == VAVありなしの吹出風量.数式を統一する.value:
            # (45)　風量バランス
            r_supply_des_d_t_i = dc.get_r_supply_des_d_t_i_2023(house.region, load.L_CS_d_t_i, load.L_H_d_t_i)
            assert r_supply_des_d_t_i.shape == (5, 24*365)
            # 出力用
            r_supply_des_i = r_supply_des_d_t_i[:, 0:1]
            # (44)　VAV 調整前の吹き出し風量
            V_dash_supply_d_t_i = dc.get_V_dash_supply_d_t_i_2023(r_supply_des_d_t_i, V_dash_hs_supply_d_t, V_vent_g_i)
        else:
            # (45)　風量バランス
            r_supply_des_i = dc.get_r_supply_des_i(A_HCZ_i)
            assert r_supply_des_i.shape == (5,)
            # 出力用
            r_supply_des_d_t_i = np.tile(r_supply_des_i, 24 * 365).reshape(5, 24 * 365)
            # (44)　VAV 調整前の吹き出し風量
            V_dash_supply_d_t_i = dc.get_V_dash_supply_d_t_i(r_supply_des_i, V_dash_hs_supply_d_t, V_vent_g_i)

        if not should_be_adjusted_Q_hat_hs_d_t:
            break

        # (40)-2nd 床下空調時 熱源機の風量を計算するための熱源機の出力 補正
        # 1. 床下 -> 居室全体 (目標方向の熱移動)
        # 床下から室への供給は無断熱床、元の負荷に含まれる損失は一般床断熱を使う。
        U_s_supply = new_ufac.U_s_vert  # 無断熱床: 2.223 W/(m2・K)
        A_s_ufac_i, _ = jjj_ufac_dc.get_A_s_ufac_i(house.A_A, house.A_MR, house.A_OR)
        mask_uf_HCZ_i = jjj_ufac_dc.get_r_A_uf_i().flatten()[:5] > 0
        #260112 IGUCHI デバッグ用
        #print("Q_hat_hs_d_t[0]: ", Q_hat_hs_d_t[0])
        assert A_s_ufac_i.ndim == 2
        delta_L_room2uf_d_t_i  \
            = np.hstack([
                jjj_ufac_dc.calc_delta_L_room2uf_i(
                    U_s_load,
                    A_s_ufac_i,
                    np.abs(Theta_ex_d_t[t] - Theta_in_d_t[t])
                ) for t in range(24*365)  # 各要素が shape(12,1)
            ])
        assert delta_L_room2uf_d_t_i.ndim == 2
        # 元の熱源機出力から除く床損失も、空調対象室（ゾーン1・2）のみ。
        Q_hat_hs_d_t -= np.sum(delta_L_room2uf_d_t_i[:5, :], axis=0)
        #260112 IGUCHI デバッグ用
        #print("Q_hat_hs_d_t[0] 床下分を引く: ", Q_hat_hs_d_t[0])

        # 2. 床下 -> 外気 (逃げ方向)
        # 式(40)の補正前出力を1階空調対象室 / 全空調対象室で按分する。
        A_s_ufac_A = float(np.sum(A_s_ufac_i))
        A_s_ufac_HCZ_1F = float(np.sum(A_s_ufac_i[:5, 0][mask_uf_HCZ_i]))
        A_HCZ_A = float(np.sum(A_HCZ_i))
        match ac_setting:
            case HeatingAcSetting():
                L_d_t_flr1st = jjj_ufac_dc.calc_L_flr1st_area_apportioned_d_t(
                    Q_hat_hs_base_d_t, A_s_ufac_HCZ_1F, A_HCZ_A, cooling=False
                )
            case CoolingAcSetting():
                L_d_t_flr1st = jjj_ufac_dc.calc_L_flr1st_area_apportioned_d_t(
                    Q_hat_hs_CS_base_d_t, A_s_ufac_HCZ_1F, A_HCZ_A, cooling=True
                )
                # 床下温度の熱収支には冷房顕熱出力のみを用いる。
            case _:
                raise ValueError

        V_dash_supply_flr1st_d_t  \
            = np.sum(V_dash_supply_d_t_i[mask_uf_HCZ_i, :], axis=0)

        Theta_uf_d_t  \
            = np.array([
                jjj_ufac_dc.calc_Theta_uf(q_hs_rtd_H(), q_hs_rtd_C(),
                    L_d_t_flr1st[t],
                    A_s_ufac_HCZ_1F,
                    U_s_supply,
                    U_s_load,
                    Theta_in_d_t[t], Theta_ex_d_t[t],
                    V_dash_supply_flr1st_d_t[t]
                ) for t in range(24*365)
            ])

        #260112 IGUCHI デバッグ用
        #print("L_d_t_flr1st[0]:", L_d_t_flr1st[0])
        #print("np.sum(A_s_ufac_i):", np.sum(A_s_ufac_i))
        #print("U_s_supply:", U_s_supply)
        #print("Theta_in_d_t[0]:", Theta_in_d_t[0])
        #print("Theta_ex_d_t[0]:", Theta_ex_d_t[0])
        #print("V_dash_supply_flr1st_d_t[0]:", V_dash_supply_flr1st_d_t[0])
        #print("Theta_uf_d_t[0] 床下温度: ", Theta_uf_d_t[0])

        natural_Theta_uf_d_t = algo.get_Theta_uf_d_t_runup(
            skin.underfloor_insulation, Theta_ex_d_t
        )
        active_mask = H if isinstance(ac_setting, HeatingAcSetting) else C
        Theta_uf_d_t = np.where(active_mask, Theta_uf_d_t, natural_Theta_uf_d_t)
        Theta_uf_for_ground_response_d_t = (
            Theta_uf_d_t
            if Theta_uf_ground_feedback_d_t is None
            else np.asarray(Theta_uf_ground_feedback_d_t, dtype=float)
        )
        assert Theta_uf_for_ground_response_d_t.shape == (24 * 365,)
        sum_Theta_dash_g_surf_A_m = calc_sum_Theta_dash_g_surf_A_m_d_t(
            Theta_uf_for_ground_response_d_t,
            Theta_ex_d_t, skin.underfloor_insulation,
            Theta_g_avg=Theta_g_avg
        )
        L_uf = algo.get_L_uf(np.sum(A_s_ufac_i))

        delta_L_uf2outdoor_d_t = np.vectorize(jjj_ufac_dc.calc_delta_L_uf2outdoor)
        delta_Theta_uf2outdoor_d_t = (
            Theta_uf_d_t - Theta_ex_d_t
            if isinstance(ac_setting, HeatingAcSetting)
            else Theta_ex_d_t - Theta_uf_d_t
        )
        delta_L_uf2outdoor_d_t = delta_L_uf2outdoor_d_t(
            phi, L_uf, delta_Theta_uf2outdoor_d_t
        )
        delta_L_uf2outdoor_d_t = np.where(active_mask, delta_L_uf2outdoor_d_t, 0.0)
        assert np.shape(delta_L_uf2outdoor_d_t) == (24 * 365,)
        Q_hat_hs_d_t += delta_L_uf2outdoor_d_t

        #260112 IGUCHI デバッグ用
        #print("delta_L_uf2outdoor_d_t[0] 床下⇒外壁: ", delta_L_uf2outdoor_d_t[0])
        #print("Q_hat_hs_d_t[0] 床下⇒外壁を足す: ", Q_hat_hs_d_t[0])

        # 3. 床下 -> 地盤 (逃げ方向)
        A_s_ufac_A = np.sum(A_s_ufac_i)

        delta_L_uf2gnd_d_t = np.vectorize(jjj_ufac_dc.calc_delta_L_uf2gnd)
        delta_L_uf2gnd_d_t = \
            delta_L_uf2gnd_d_t(q_hs_rtd_H(), q_hs_rtd_C(),
                A_s_ufac_A, jjj_consts.R_g, Phi_A_0, Theta_uf_d_t, sum_Theta_dash_g_surf_A_m, Theta_g_avg)
        delta_L_uf2gnd_d_t = np.where(active_mask, delta_L_uf2gnd_d_t, 0.0)
        Q_hat_hs_d_t += delta_L_uf2gnd_d_t

        if isinstance(ac_setting, HeatingAcSetting):
            # 式(40)：床・外気・地盤の全補正後に暖房出力を0以上へ制限する。
            Q_hat_hs_d_t = limit_corrected_heating_output(Q_hat_hs_d_t)
        elif isinstance(ac_setting, CoolingAcSetting):
            # 床・外気・地盤の補正は顕熱成分にだけ適用する。
            # 式(40-2a)に従い、補正後顕熱を0以上に制限してから潜熱を加える。
            Q_hat_hs_d_t = combine_corrected_cooling_output(
                Q_hat_hs_base_d_t,
                Q_hat_hs_CS_base_d_t,
                Q_hat_hs_d_t,
            )

        #260112 IGUCHI デバッグ用
        #print("delta_L_uf2gnd_d_t[0] 床下⇒地盤: ", delta_L_uf2gnd_d_t[0])
        #print("Q_hat_hs_d_t[0] 床下⇒地盤を足す: ", Q_hat_hs_d_t[0])

        # 補正完了した Q^hs を使って V'supply を再計算する
        should_be_adjusted_Q_hat_hs_d_t = False

    df_output['Q_hat_hs_d_t_ufac'] = Q_hat_hs_d_t
    df_output2['r_supply_des_i'] = r_supply_des_i
    df_output = df_output.assign(
        r_supply_des_d_t_1 = r_supply_des_d_t_i[0],
        r_supply_des_d_t_2 = r_supply_des_d_t_i[1],
        r_supply_des_d_t_3 = r_supply_des_d_t_i[2],
        r_supply_des_d_t_4 = r_supply_des_d_t_i[3],
        r_supply_des_d_t_5 = r_supply_des_d_t_i[4]
    )
    df_output = df_output.assign(
        V_dash_supply_d_t_1 = V_dash_supply_d_t_i[0],
        V_dash_supply_d_t_2 = V_dash_supply_d_t_i[1],
        V_dash_supply_d_t_3 = V_dash_supply_d_t_i[2],
        V_dash_supply_d_t_4 = V_dash_supply_d_t_i[3],
        V_dash_supply_d_t_5 = V_dash_supply_d_t_i[4]
    )

    # (53)　負荷バランス時の非居室の絶対湿度
    X_star_NR_d_t = dc.get_X_star_NR_d_t(X_star_HBR_d_t, load.L_CL_d_t_i, L_wtr, V_vent_l_NR_d_t, V_dash_supply_d_t_i, house.region)
    df_output['X_star_NR_d_t'] = X_star_NR_d_t

    # (52)　負荷バランス時の非居室の室温
    if new_ufac.new_ufac_flg == 床下空調ロジック.変更する:
        V_dash_supply_d_t_A = np.sum(V_dash_supply_d_t_i[0:5, :], axis=0)
        L_H_NR_d_t_A = np.sum(load.L_H_d_t_i[5:, :], axis=0)
        L_CS_NR_d_t_A = np.sum(load.L_CS_d_t_i[5:, :], axis=0)

        assert A_prt_i.shape == (5,)
        A_prt_A = np.sum(A_prt_i)
        HCM = np.array(climate.get_HCM_d_t())

        r_A_NR_uf_1F = jjj_ufac_dc.get_r_A_NR_uf_1F()

        #デバッグ用 250501 IGUCHI
        #print("Theta_in_d_t[4848]", Theta_in_d_t[4848])
        #print("Q", skin.Q)
        #print("A_NR", A_NR)
        #print("V_vent_l_NR_d_t[4848]", V_vent_l_NR_d_t[4848])
        #print("V_dash_supply_A[4848]", V_dash_supply_d_t_A[4848])
        #print("A_NR", A_NR)
        #print("V_vent_l_NR_d_t[4848]", V_vent_l_NR_d_t[4848])
        #print("V_dash_supply_A[4848]", V_dash_supply_d_t_A[4848])
        #print("U_prt", U_prt)
        #print("A_prt_A", A_prt_A)
        #print("L_H_NR_A[4848]", L_H_NR_d_t_A[4848])
        #print("L_CS_NR_A[4848]", L_CS_NR_d_t_A[4848])
        #print("Theta_uf_d_t[4848]", Theta_uf_d_t[4848])
        #print("HCM[4848]", HCM[4848])

        Theta_star_NR_d_t = np.vectorize(get_Theta_star_NR)
        Theta_star_NR_d_t = \
            Theta_star_NR_d_t(
                Theta_star_HBR = Theta_star_HBR_d_t,  # (8760,)
                Q = skin.Q,
                A_NR = A_NR,
                V_vent_l_NR = V_vent_l_NR_d_t,  # (8760,)
                V_dash_supply_A = V_dash_supply_d_t_A,  # (8760,)
                U_prt = U_prt,
                A_prt_A = A_prt_A,
                L_H_NR_A = L_H_NR_d_t_A,  # (8760,)
                L_CS_NR_A = L_CS_NR_d_t_A,  # (8760,)
                Theta_NR = Theta_in_d_t,  # この時点では仮置きの値を使用⇒夏期は27℃とする必要がある　250501 井口
                Theta_uf = Theta_uf_d_t,  # (8760,)
                HCM = HCM,  # (8760,)
                r_A_NR_1F = r_A_NR_uf_1F
            )
        #print("Theta_star_HBR[0]: ", Theta_star_HBR_d_t[0])
        #print("Q: ", skin.Q)
        #print("V_vent_l_NR[0]: ", V_vent_l_NR_d_t[0])
        #print("V_dash_supply_A[0]: ", V_dash_supply_d_t_A[0])
        #print("U_prt; ", U_prt)
        #print("A_prt_A: ", A_prt_A)
        #print("L_H_NR_A[0]: ", L_H_NR_d_t_A[0])
        #print("L_CS_NR_A[0]: ", L_CS_NR_d_t_A[0])
        #print("Theta_NR[0]: ", Theta_in_d_t[0])
        #print("Theta_uf[0]: ", Theta_uf_d_t[0])
        #print("HCM[0]: ", HCM[0])
    else:
        Theta_star_NR_d_t = \
            dc.get_Theta_star_NR_d_t(
                Theta_star_HBR_d_t, skin.Q, A_NR,
                V_vent_l_NR_d_t, V_dash_supply_d_t_i,
                U_prt, A_prt_i, load.L_H_d_t_i, load.L_CS_d_t_i, house.region)

    df_output['Theta_star_NR_d_t'] = Theta_star_NR_d_t

    # (49)　実際の非居室の絶対湿度
    X_NR_d_t = dc.get_X_NR_d_t(X_star_NR_d_t)
    df_output['X_NR_d_t'] = X_NR_d_t

    # (47)　実際の居室の絶対湿度
    X_HBR_d_t_i = dc.get_X_HBR_d_t_i(X_star_HBR_d_t)
    df_output = df_output.assign(
        X_HBR_d_t_1 = X_HBR_d_t_i[0],
        X_HBR_d_t_2 = X_HBR_d_t_i[1],
        X_HBR_d_t_3 = X_HBR_d_t_i[2],
        X_HBR_d_t_4 = X_HBR_d_t_i[3],
        X_HBR_d_t_5 = X_HBR_d_t_i[4]
    )

    """ 熱損失・熱取得を含む負荷バランス時の熱負荷 - 熱損失・熱取得を含む負荷バランス時(1) """
    # (11)　熱損失を含む負荷バランス時の非居室への熱移動
    Q_star_trs_prt_d_t_i = dc.get_Q_star_trs_prt_d_t_i(U_prt, A_prt_i, Theta_star_HBR_d_t, Theta_star_NR_d_t)
    df_output = df_output.assign(
        Q_star_trs_prt_d_t_i_1 = Q_star_trs_prt_d_t_i[0],
        Q_star_trs_prt_d_t_i_2 = Q_star_trs_prt_d_t_i[1],
        Q_star_trs_prt_d_t_i_3 = Q_star_trs_prt_d_t_i[2],
        Q_star_trs_prt_d_t_i_4 = Q_star_trs_prt_d_t_i[3],
        Q_star_trs_prt_d_t_i_5 = Q_star_trs_prt_d_t_i[4]
    )

    # (10)　熱取得を含む負荷バランス時の冷房潜熱負荷
    L_star_CL_d_t_i = dc.get_L_star_CL_d_t_i(load.L_CS_d_t_i, load.L_CL_d_t_i, house.region)
    df_output = df_output.assign(
        L_star_CL_d_t_i_1 = L_star_CL_d_t_i[0],
        L_star_CL_d_t_i_2 = L_star_CL_d_t_i[1],
        L_star_CL_d_t_i_3 = L_star_CL_d_t_i[2],
        L_star_CL_d_t_i_4 = L_star_CL_d_t_i[3],
        L_star_CL_d_t_i_5 = L_star_CL_d_t_i[4]
    )

    # NOTE: 熱繰越を行うverと行わないverで 同じ処理を異なるループの粒度で二重実装が必要です
    # 実装量/計算量 の多い仕様の場合には 過剰熱繰越ナシ(一般的なパターン) のみ実装として、オプション併用を拒否する仕様も検討しましょう
    if carryover_heat_dto.carry_over_heat == 過剰熱量繰越計算.行う:

        # NOTE: 過剰熱繰越と併用しないオプションはインプットデータクラスの段階で強制オフしている

        # インデックス順に更新対象
        L_star_CS_d_t_i = np.zeros((5, 24 * 365))
        L_star_H_d_t_i = np.zeros((5, 24 * 365))

        # 実際の居室・非居室の室温
        Theta_star_hs_in_d_t = np.zeros(24 * 365)
        Theta_HBR_d_t_i = np.zeros((5, 24 * 365))
        Theta_NR_d_t = np.zeros(24 * 365)
        # TODO: 空からappendしていくロジックに変更することで
        # tインデックスの誤用がないことを保証できる

        # 過剰熱繰越の項(確認用)
        carryovers = np.zeros((5, 24 * 365))

        # 季節から計算の必要性を判断
        H, C, M = dc.get_season_array_d_t(house.region)

        for t in range(0, 24 * 365):
            # TODO: 先頭時の扱いを考慮
            isFirst = (t == 0)

            if H[t] and C[t]:
                raise ValueError("想定外の季節")
            elif isFirst:
                carryover = np.zeros((5, 1))
            # 暖房期 前時刻にて 暖かさに余裕があるとき
            elif H[t] and np.any(Theta_HBR_d_t_i[:, t-1:t] > Theta_star_HBR_d_t[t-1]):
                carryover = jjj_carryover_heat.calc_carryover(
                                    H[t], C[t], A_HCZ_i,
                                    Theta_HBR_d_t_i[:, t-1:t],
                                    Theta_star_HBR_d_t[t])
            # 冷房期 前時刻にて 涼しさに余裕があるとき
            elif C[t] and np.any(Theta_HBR_d_t_i[:, t-1:t] < Theta_star_HBR_d_t[t-1]):
                carryover = jjj_carryover_heat.calc_carryover(
                                    H[t], C[t], A_HCZ_i,
                                    Theta_HBR_d_t_i[:, t-1:t],
                                    Theta_star_HBR_d_t[t])
            else:
                carryover = np.zeros((5, 1))
                # 前時刻の Theta_HBR_d_t_i を使用するため
                # 空調がなくてもすぐ次のループに行かず (46)(48)式の計算は行う

            carryovers[:, t] = carryover[:, 0]  # 確認用

            # (8)　熱損失を含む負荷バランス時の暖房負荷
            L_star_H_d_t_i[:, t:t+1]  \
                = jjj_carryover_heat.get_L_star_H_i_2024(
                    H[t],
                    load.L_H_d_t_i[:5, t:t+1],
                    Q_star_trs_prt_d_t_i[:5, t:t+1],
                    carryover)

            # (9)　熱取得を含む負荷バランス時の冷房顕熱負荷
            L_star_CS_d_t_i[:, t:t+1]  \
                = jjj_carryover_heat.get_L_star_CS_i_2024(
                    C[t],
                    load.L_CS_d_t_i[:5, t:t+1],
                    Q_star_trs_prt_d_t_i[:5, t:t+1],
                    carryover)

            ####################################################################################################################
            if ac_setting.type in [
                    計算モデル.ダクト式セントラル空調機,
                    計算モデル.RAC活用型全館空調_潜熱評価モデル
                ]:
                # (33)
                L_star_CL_d_t = dc.get_L_star_CL_d_t(L_star_CL_d_t_i)
                # (32)
                L_star_CS_d_t = dc.get_L_star_CS_d_t(L_star_CS_d_t_i)
                # (31)
                L_star_CL_max_d_t = dc.get_L_star_CL_max_d_t(L_star_CS_d_t)
                # (30)
                L_star_dash_CL_d_t = dc.get_L_star_dash_CL_d_t(L_star_CL_max_d_t, L_star_CL_d_t)
                # (29)
                L_star_dash_C_d_t = dc.get_L_star_dash_C_d_t(L_star_CS_d_t, L_star_dash_CL_d_t)
                # (28)
                SHF_dash_d_t = dc.get_SHF_dash_d_t(L_star_CS_d_t, L_star_dash_C_d_t)
                # (27)
                Q_hs_max_C_d_t = dc.get_Q_hs_max_C_d_t_2024(ac_setting.type, q_hs_rtd_C(), cool_CRAC.input_C_af)
                # (26)
                Q_hs_max_CL_d_t = dc.get_Q_hs_max_CL_d_t(Q_hs_max_C_d_t, SHF_dash_d_t, L_star_dash_CL_d_t)
                # (25)
                Q_hs_max_CS_d_t = dc.get_Q_hs_max_CS_d_t(Q_hs_max_C_d_t, SHF_dash_d_t)
                # (24)
                C_df_H_d_t = dc.get_C_df_H_d_t(Theta_ex_d_t, h_ex_d_t)
                # (23)
                Q_hs_max_H_d_t = dc.get_Q_hs_max_H_d_t_2024(ac_setting.type, q_hs_rtd_H(), C_df_H_d_t, heat_CRAC.input_C_af)

            elif ac_setting.type in [
                    計算モデル.RAC活用型全館空調_現行省エネ法RACモデル,
                    計算モデル.電中研モデル
                ]:
                # (24)　デフロストに関する暖房出力補正係数
                C_df_H_d_t = dc.get_C_df_H_d_t(Theta_ex_d_t, h_ex_d_t)
                # 最大暖房能力比
                q_r_max_H = rac.get_q_r_max_H(heat_CRAC.q_max, heat_CRAC.q_rtd)
                # 最大暖房出力比
                Q_r_max_H_d_t = rac.calc_Q_r_max_H_d_t(cool_CRAC.q_rtd, q_r_max_H, Theta_ex_d_t)
                # 最大暖房出力
                Q_max_H_d_t = rac.calc_Q_max_H_d_t(Q_r_max_H_d_t, heat_CRAC.q_rtd, Theta_ex_d_t, h_ex_d_t, heat_CRAC.input_C_af)
                Q_hs_max_H_d_t = Q_max_H_d_t
                # 最大冷房能力比
                q_r_max_C = rac.get_q_r_max_C(cool_CRAC.q_max, cool_CRAC.q_rtd)
                # 最大冷房出力比
                Q_r_max_C_d_t = rac.calc_Q_r_max_C_d_t(q_r_max_C, cool_CRAC.q_rtd, Theta_ex_d_t)
                # 最大冷房出力
                Q_max_C_d_t = rac.calc_Q_max_C_d_t(Q_r_max_C_d_t, cool_CRAC.q_rtd, cool_CRAC.input_C_af)
                Q_hs_max_C_d_t = Q_max_C_d_t
                # 冷房負荷最小顕熱比
                SHF_L_min_c = rac.get_SHF_L_min_c()
                # 最大冷房潜熱負荷
                L_max_CL_d_t = rac.get_L_max_CL_d_t(np.sum(load.L_CS_d_t_i, axis=0), SHF_L_min_c)
                # 補正冷房潜熱負荷
                L_dash_CL_d_t = rac.get_L_dash_CL_d_t(L_max_CL_d_t, np.sum(load.L_CL_d_t_i, axis=0))
                L_dash_C_d_t = rac.get_L_dash_C_d_t(np.sum(load.L_CS_d_t_i, axis=0), L_dash_CL_d_t)
                # 冷房負荷補正顕熱比
                SHF_dash_d_t = rac.get_SHF_dash_d_t(np.sum(load.L_CS_d_t_i, axis=0), L_dash_C_d_t)
                # 最大冷房顕熱出力, 最大冷房潜熱出力
                Q_max_CS_d_t = rac.get_Q_max_CS_d_t(Q_max_C_d_t, SHF_dash_d_t)
                Q_max_CL_d_t = rac.get_Q_max_CL_d_t(Q_max_C_d_t, SHF_dash_d_t, L_dash_CL_d_t)
                Q_hs_max_C_d_t = Q_max_C_d_t
                Q_hs_max_CL_d_t = Q_max_CL_d_t
                Q_hs_max_CS_d_t = Q_max_CS_d_t

            else:
                raise Exception('設備機器の種類の入力が不正です。')
            ####################################################################################################################

            # (20)　負荷バランス時の熱源機の入口における絶対湿度
            X_star_hs_in_d_t = dc.get_X_star_hs_in_d_t(X_star_NR_d_t)

            # (19)　負荷バランス時の熱源機の入口における空気温度
            # 前時刻の非居室の温度を熱源入口温度として使用して負荷を下げる
            Theta_star_hs_in_d_t[t] = dc.get_Theta_star_hs_in_d_t(Theta_star_NR_d_t)[t] \
                if (isFirst or not (H[t] or C[t]))  \
                else Theta_NR_d_t[t-1]

            # (18)　熱源機の出口における空気温度の最低値
            X_hs_out_min_C_d_t = dc.get_X_hs_out_min_C_d_t(X_star_hs_in_d_t, Q_hs_max_CL_d_t, V_dash_supply_d_t_i)

            # (22)　熱源機の出口における要求絶対湿度
            X_req_d_t_i = dc.get_X_req_d_t_i(X_star_HBR_d_t, L_star_CL_d_t_i, V_dash_supply_d_t_i, house.region)

            # (21)　熱源機の出口における要求空気温度
            Theta_req_d_t_i = dc.get_Theta_req_d_t_i(Theta_sur_d_t_i, Theta_star_HBR_d_t, V_dash_supply_d_t_i,
                                L_star_H_d_t_i, L_star_CS_d_t_i, l_duct_i, house.region)

            if skin.underfloor_air_conditioning_air_supply:
                for i in range(2):  # i=0,1
                    Theta_uf_d_t, Theta_g_surf_d_t, *others  \
                        = algo.calc_Theta(  # 熱繰越-1st
                            house.region, house.A_A, house.A_MR, house.A_OR, skin.Q, skin.YUCACO_r_A_ufvnt, skin.underfloor_insulation,
                            Theta_req_d_t_i[i], Theta_ex_d_t, V_dash_supply_d_t_i[i],
                            '', load.L_H_d_t_i, load.L_CS_d_t_i)

                    # 暖冷房期 判別
                    match ac_setting:
                        case HeatingAcSetting():
                            mask = Theta_req_d_t_i[i] > Theta_uf_d_t
                        case CoolingAcSetting():
                            mask = Theta_req_d_t_i[i] < Theta_uf_d_t
                        case _:
                            raise ValueError

                    Theta_req_d_t_i[i]  \
                        = np.where(mask,
                                Theta_req_d_t_i[i] + (Theta_req_d_t_i[i] - Theta_uf_d_t),
                                Theta_req_d_t_i[i])

            # NOTE: 過剰熱量繰越 未利用の場合では、式(14)(46)(48)の条件に合わせてTheta_NR_d_tを初期化
            # Theta_NR_d_t = np.zeros(24 * 365)
            # 過剰熱量繰越 利用時には、初期化せず再利用する

            # (15)　熱源機の出口における絶対湿度
            X_hs_out_d_t = dc.get_X_hs_out_d_t(X_NR_d_t, X_req_d_t_i, V_dash_supply_d_t_i, X_hs_out_min_C_d_t, L_star_CL_d_t_i, house.region)

            # (17)　冷房時の熱源機の出口における空気温度の最低値
            Theta_hs_out_min_C_d_t = dc.get_Theta_hs_out_min_C_d_t(Theta_star_hs_in_d_t, Q_hs_max_CS_d_t, V_dash_supply_d_t_i)

            # (16)　暖房時の熱源機の出口における空気温度の最高値
            Theta_hs_out_max_H_d_t = dc.get_Theta_hs_out_max_H_d_t(Theta_star_hs_in_d_t, Q_hs_max_H_d_t, V_dash_supply_d_t_i)

            # L_star_H_d_t_i，L_star_CS_d_t_iの暖冷房区画1～5を合算し0以上だった場合の順序で計算
            # (14)　熱源機の出口における空気温度
            Theta_hs_out_d_t = dc.get_Theta_hs_out_d_t(ac_setting.VAV, Theta_req_d_t_i, V_dash_supply_d_t_i,
                                                    L_star_H_d_t_i, L_star_CS_d_t_i, house.region, Theta_NR_d_t,
                                                    Theta_hs_out_max_H_d_t, Theta_hs_out_min_C_d_t)

            # (43)　暖冷房区画𝑖の吹き出し風量
            V_supply_d_t_i_before = dc.get_V_supply_d_t_i(L_star_H_d_t_i, L_star_CS_d_t_i, Theta_sur_d_t_i, l_duct_i, Theta_star_HBR_d_t
                                                        , V_vent_g_i, V_dash_supply_d_t_i, ac_setting.VAV, house.region, Theta_hs_out_d_t)
            V_supply_d_t_i = jjj_vsupcap.cap_V_supply_d_t_i(v_supply_cap_dto, V_supply_d_t_i_before, V_dash_supply_d_t_i
                                                        , V_vent_g_i, house.region, V_hs_dsgn_H, V_hs_dsgn_C, print_exec=False)

            # (41)　暖冷房区画𝑖の吹き出し温度
            Theta_supply_d_t_i = dc.get_Thata_supply_d_t_i(Theta_sur_d_t_i, Theta_hs_out_d_t, Theta_star_HBR_d_t, l_duct_i,
                                                       V_supply_d_t_i, L_star_H_d_t_i, L_star_CS_d_t_i, house.region)

            if skin.underfloor_air_conditioning_air_supply:
                for i in range(2):  # i=0,1
                    Theta_uf_d_t, Theta_g_surf_d_t, *others  \
                        = algo.calc_Theta(  # 熱繰越-2nd
                            house.region, house.A_A, house.A_MR, house.A_OR, skin.Q, skin.YUCACO_r_A_ufvnt, skin.underfloor_insulation,
                            Theta_supply_d_t_i[i], Theta_ex_d_t, V_dash_supply_d_t_i[i],
                            '', load.L_H_d_t_i, load.L_CS_d_t_i)

                    match ac_setting:
                        case HeatingAcSetting():
                            # 暖房期は 床下温度以上の温度は吹き出てこない
                            Theta_supply_d_t_i[i] = np.clip(Theta_supply_d_t_i[i], None, Theta_uf_d_t)
                        case CoolingAcSetting():
                            # 冷房期は 床下温度以下の温度は吹き出てこない
                            Theta_supply_d_t_i[i] = np.clip(Theta_supply_d_t_i[i], Theta_uf_d_t, None)
                        case _:
                            raise ValueError

            # NOTE: t==0 でも最後までループを走ることに注意(途中で continue しない)
            # 0 の扱いは全てのメソッドで考慮されていること

            # (46)　暖冷房区画𝑖の実際の居室の室温
            Theta_HBR_d_t_i[:, t:t+1] \
                = jjj_carryover_heat.get_Theta_HBR_i_2023(
                    H[t], C[t], M[t],
                    Theta_star_HBR_d_t[t],
                    V_supply_d_t_i[:, t:t+1],  # (5,1)
                    Theta_supply_d_t_i[:, t:t+1],  # (5,1)
                    U_prt,
                    A_prt_i.reshape(-1,1),  # (5,1)
                    skin.Q,
                    A_HCZ_i.reshape(-1,1),  # (5,1)
                    L_star_H_d_t_i[:5, t:t+1],  # (5,1)
                    L_star_CS_d_t_i[:5, t:t+1],  # (5,1)
                    np.zeros((5,1)) if t==0 else Theta_HBR_d_t_i[:5, t-1:t])  # (5,1)

            # (48)　実際の非居室の室温
            Theta_NR_d_t[t] \
                = jjj_carryover_heat.get_Theta_NR_2023(
                    isFirst, H[t], C[t], M[t],
                    Theta_star_NR_d_t[t],
                    Theta_star_HBR_d_t[t],
                    Theta_HBR_d_t_i[:, t:t+1],  # (5,1)
                    A_NR,
                    V_vent_l_NR_d_t[t],
                    V_dash_supply_d_t_i[:, t:t+1],  # (5,1)
                    V_supply_d_t_i[:, t:t+1],  # (5,1)
                    U_prt,
                    A_prt_i.reshape(-1,1),  # (5,1)
                    skin.Q,
                    0 if t==0 else Theta_NR_d_t[t-1])

    else:  # 過剰熱繰越ナシ(一般的なパターン)

        # NOTE: 床下空調のための r_A_ufvnt の上書きはココより前に行わない
        # 外気導入の負荷削減の計算までは、削減ナシ(r_A_ufvnt=None) のままであるべきため

        # (9) 熱取得を含む負荷バランス時の冷房顕熱負荷
        L_star_CS_d_t_i = dc.get_L_star_CS_d_t_i(load.L_CS_d_t_i, Q_star_trs_prt_d_t_i, house.region)
        # (8) 熱損失を含む負荷バランス時の暖房負荷
        L_star_H_d_t_i = dc.get_L_star_H_d_t_i(load.L_H_d_t_i, Q_star_trs_prt_d_t_i, house.region)

        if new_ufac.new_ufac_flg == 床下空調ロジック.変更する:
            # 部屋→床下への熱移動分が戻ってくるため負荷控除する
            delta_L_uf2room_d_t_i = np.hstack([
                jjj_ufac_dc.calc_delta_L_room2uf_i(
                    U_s_load,
                    A_s_ufac_i,
                    np.abs(Theta_star_HBR_d_t[t] - Theta_ex_d_t[t])
                ) for t in range(24*365)
            ])
            H, C, M = dc.get_season_array_d_t(house.region)
            # (9)-補正
            Cf = np.logical_and(C, load.L_CS_d_t_i[:5, :] > 0)
            assert Cf.shape == (5, 24*365)
            L_star_CS_d_t_i[Cf] += delta_L_uf2room_d_t_i[:5, :][Cf]
            # (8)-補正
            Hf = np.logical_and(H, load.L_H_d_t_i[:5, :] > 0)
            assert Hf.shape == (5, 24*365)
            L_star_H_d_t_i[Hf] -= delta_L_uf2room_d_t_i[:5, :][Hf]

            # 床下空調 新ロジック 調査用出力ファイル
            new_ufac_df.update_df({
                "L_H_d_t_1": load.L_H_d_t_i[0],   "L_H_d_t_2": load.L_H_d_t_i[1],   "L_H_d_t_3": load.L_H_d_t_i[2],   "L_H_d_t_4": load.L_H_d_t_i[3],   "L_H_d_t_5": load.L_H_d_t_i[4],
                "L_CS_d_t_1": load.L_CS_d_t_i[0], "L_CS_d_t_2": load.L_CS_d_t_i[1], "L_CS_d_t_3": load.L_CS_d_t_i[2], "L_CS_d_t_4": load.L_CS_d_t_i[3], "L_CS_d_t_5": load.L_CS_d_t_i[4],
                "L_CL_d_t_1": load.L_CL_d_t_i[0], "L_CL_d_t_2": load.L_CL_d_t_i[1], "L_CL_d_t_3": load.L_CL_d_t_i[2], "L_CL_d_t_4": load.L_CL_d_t_i[3], "L_CL_d_t_5": load.L_CL_d_t_i[4],
                "L_star_CS_d_t_1": L_star_CS_d_t_i[0], "L_star_CS_d_t_2": L_star_CS_d_t_i[1], "L_star_CS_d_t_3": L_star_CS_d_t_i[2], "L_star_CS_d_t_4": L_star_CS_d_t_i[3], "L_star_CS_d_t_5": L_star_CS_d_t_i[4],
                "L_star_H_d_t_1": L_star_H_d_t_i[0],  "L_star_H_d_t_2": L_star_H_d_t_i[1],   "L_star_H_d_t_3": L_star_H_d_t_i[2],   "L_star_H_d_t_4": L_star_H_d_t_i[3],   "L_star_H_d_t_5": L_star_H_d_t_i[4],
            })

        ####################################################################################################################
        if ac_setting.type in [
                計算モデル.ダクト式セントラル空調機,
                計算モデル.RAC活用型全館空調_潜熱評価モデル
            ]:
            # (33)
            L_star_CL_d_t = dc.get_L_star_CL_d_t(L_star_CL_d_t_i)
            # (32)
            L_star_CS_d_t = dc.get_L_star_CS_d_t(L_star_CS_d_t_i)
            # (31)
            L_star_CL_max_d_t = dc.get_L_star_CL_max_d_t(L_star_CS_d_t)
            # (30)
            L_star_dash_CL_d_t = dc.get_L_star_dash_CL_d_t(L_star_CL_max_d_t, L_star_CL_d_t)
            # (29)
            L_star_dash_C_d_t = dc.get_L_star_dash_C_d_t(L_star_CS_d_t, L_star_dash_CL_d_t)
            # (28)
            SHF_dash_d_t = dc.get_SHF_dash_d_t(L_star_CS_d_t, L_star_dash_C_d_t)
            # (27)
            Q_hs_max_C_d_t = dc.get_Q_hs_max_C_d_t_2024(ac_setting.type, q_hs_rtd_C(), cool_CRAC.input_C_af)
            # (26)
            Q_hs_max_CL_d_t = dc.get_Q_hs_max_CL_d_t(Q_hs_max_C_d_t, SHF_dash_d_t, L_star_dash_CL_d_t)
            # (25)
            Q_hs_max_CS_d_t = dc.get_Q_hs_max_CS_d_t(Q_hs_max_C_d_t, SHF_dash_d_t)
            # (24) デフロストに関する暖房出力補正係数
            C_df_H_d_t = climate.get_C_df_H_d_t()
            # (23)
            Q_hs_max_H_d_t = dc.get_Q_hs_max_H_d_t_2024(ac_setting.type, q_hs_rtd_H(), C_df_H_d_t, heat_CRAC.input_C_af)

        elif ac_setting.type in [
                計算モデル.RAC活用型全館空調_現行省エネ法RACモデル,
                計算モデル.電中研モデル
            ]:
            # (24)　デフロストに関する暖房出力補正係数
            C_df_H_d_t = climate.get_C_df_H_d_t()
            _logger.debug(f'C_df_H_d_t: {C_df_H_d_t}')

            # 最大暖房能力比
            q_r_max_H = rac.get_q_r_max_H(heat_CRAC.q_max, heat_CRAC.q_rtd)
            _logger.debug(f'q_r_max_H: {q_r_max_H}')  # here

            # 最大暖房出力比
            Q_r_max_H_d_t = rac.calc_Q_r_max_H_d_t(cool_CRAC.q_rtd, q_r_max_H, Theta_ex_d_t)
            _logger.NDdebug("Q_r_max_H_d_t", Q_r_max_H_d_t)  # here

            # 最大暖房出力
            Q_max_H_d_t = rac.calc_Q_max_H_d_t(Q_r_max_H_d_t, heat_CRAC.q_rtd, Theta_ex_d_t, h_ex_d_t, heat_CRAC.input_C_af)
            _logger.NDdebug("Q_max_H_d_t", Q_max_H_d_t)
            Q_hs_max_H_d_t = Q_max_H_d_t

            # 最大冷房能力比
            q_r_max_C = rac.get_q_r_max_C(cool_CRAC.q_max, cool_CRAC.q_rtd)
            _logger.debug(f"q_r_max_C: {q_r_max_C}")

            # 最大冷房出力比
            Q_r_max_C_d_t = rac.calc_Q_r_max_C_d_t(q_r_max_C, cool_CRAC.q_rtd, Theta_ex_d_t)
            _logger.NDdebug("Theta_ex_d_t", Theta_ex_d_t)
            _logger.NDdebug("Q_r_max_C_d_t", Q_r_max_C_d_t)

            # 最大冷房出力
            Q_max_C_d_t = rac.calc_Q_max_C_d_t(Q_r_max_C_d_t, cool_CRAC.q_rtd, cool_CRAC.input_C_af)
            _logger.NDdebug("Q_max_C_d_t", Q_max_C_d_t)
            Q_hs_max_C_d_t = Q_max_C_d_t

            # 冷房負荷最小顕熱比
            SHF_L_min_c = rac.get_SHF_L_min_c()

            # 最大冷房潜熱負荷
            L_max_CL_d_t = rac.get_L_max_CL_d_t(np.sum(load.L_CS_d_t_i, axis=0), SHF_L_min_c)

            # 補正冷房潜熱負荷
            L_dash_CL_d_t = rac.get_L_dash_CL_d_t(L_max_CL_d_t, np.sum(load.L_CL_d_t_i, axis=0))
            L_dash_C_d_t = rac.get_L_dash_C_d_t(np.sum(load.L_CS_d_t_i, axis=0), L_dash_CL_d_t)

            # 冷房負荷補正顕熱比
            SHF_dash_d_t = rac.get_SHF_dash_d_t(np.sum(load.L_CS_d_t_i, axis=0), L_dash_C_d_t)

            # 最大冷房顕熱出力, 最大冷房潜熱出力
            Q_max_CS_d_t = rac.get_Q_max_CS_d_t(Q_max_C_d_t, SHF_dash_d_t)
            Q_max_CL_d_t = rac.get_Q_max_CL_d_t(Q_max_C_d_t, SHF_dash_d_t, L_dash_CL_d_t)
            Q_hs_max_C_d_t = Q_max_C_d_t
            Q_hs_max_CL_d_t = Q_max_CL_d_t
            Q_hs_max_CS_d_t = Q_max_CS_d_t
        else:
            raise Exception('設備機器の種類の入力が不正です。')
        ####################################################################################################################

        # (20)　負荷バランス時の熱源機の入口における絶対湿度
        X_star_hs_in_d_t = dc.get_X_star_hs_in_d_t(X_star_NR_d_t)

        # (19)　負荷バランス時の熱源機の入口における空気温度
        Theta_star_hs_in_d_t = dc.get_Theta_star_hs_in_d_t(Theta_star_NR_d_t)

        # (18)　熱源機の出口における空気温度の最低値
        X_hs_out_min_C_d_t = dc.get_X_hs_out_min_C_d_t(X_star_hs_in_d_t, Q_hs_max_CL_d_t, V_dash_supply_d_t_i)

        # (22)　熱源機の出口における要求絶対湿度
        X_req_d_t_i = dc.get_X_req_d_t_i(X_star_HBR_d_t, L_star_CL_d_t_i, V_dash_supply_d_t_i, house.region)

        # (21)　熱源機の出口における要求空気温度
        Theta_req_d_t_i = dc.get_Theta_req_d_t_i(Theta_sur_d_t_i, Theta_star_HBR_d_t, V_dash_supply_d_t_i,
                            L_star_H_d_t_i, L_star_CS_d_t_i, l_duct_i, house.region)

        # NOTE: 床下空調を使用する(旧・新 両ロジックとも) 対象居室のみ損失分を補正する
        if new_ufac.new_ufac_flg == 床下空調ロジック.変更する:
            # 期待される床下温度を事前に計算(本計算は後で行う)
            Theta_uf_d_t_2023 = calc_Theta_uf_d_t_2023(
                L_star_H_d_t_i, L_star_CS_d_t_i,
                house.A_A, house.A_MR, house.A_OR, skin.r_A_ufac,
                V_dash_supply_d_t_i, Theta_ex_d_t,
                H=H, C=C, M=M,
            )
            # θuf_supply を逆算(二分探索)
            _, _, Theta_uf_supply_d_t  \
                = algo.calc_Theta(  # 新床下空調-1st
                    region = house.region,
                    A_A = house.A_A,
                    A_MR = house.A_MR,
                    A_OR = house.A_OR,
                    Q = skin.Q,
                    r_A_ufvnt = skin.r_A_ufac,  # 床下換気ではなく床下空調のため
                    underfloor_insulation = skin.underfloor_insulation,
                    Theta_sa_d_t = Theta_uf_d_t_2023,  # ★
                    Theta_ex_d_t = Theta_ex_d_t,
                    # 熱源機出口温度から吹き出し温度を計算する
                    V_sa_d_t_A = np.sum(V_dash_supply_d_t_i[:2, :], axis=0),  # i=1,2
                    H_OR_C = "",
                    L_dash_H_R_d_t_i = load.L_dash_H_R_d_t_i,
                    L_dash_CS_R_d_t_i = load.L_dash_CS_R_d_t_i,
                    calc_backwards = True,  # 従来の θuf 用計算式を借りて θuf_supply計算する
                    new_ufac = new_ufac,
                    new_ufac_df = new_ufac_df,
                    # 式(21)の逆算でも、地盤応答は目標床下温度ではなく
                    # 前時刻の実床下温度から更新する。
                    Theta_uf_ground_feedback_d_t=(
                        Theta_uf_ground_feedback_d_t
                    ),
                )

            # 対象居室 i=1,2(1階居室)の損失分を補正する
            Theta_req_d_t_i = np.vstack([
                    np.tile(Theta_uf_supply_d_t, (2, 1)),
                    Theta_req_d_t_i[2:, :]
                ])
            assert np.shape(Theta_req_d_t_i)==(5, 8760), "想定外の行列数"

            match (q_hs_rtd_H(), q_hs_rtd_C()):
                case (None, None):
                    raise Exception("どちらかのみを前提")
                case (_, None):
                    Theta_in_H = Theta_in_d_t[0]
                    Theta_req_d_t_i = np.clip(Theta_req_d_t_i, Theta_in_H, None)
                case (None, _):
                    Theta_in_C = Theta_in_d_t[0]
                    Theta_req_d_t_i = np.clip(Theta_req_d_t_i, None, Theta_in_C)
                case (_, _):
                    raise Exception("どちらかのみを前提")

            new_ufac_df.update_df({
                "Theta_uf_d_t_2023": Theta_uf_d_t_2023,
                "Theta_req_d_t_1": Theta_req_d_t_i[0], "Theta_req_d_t_2": Theta_req_d_t_i[1], "Theta_req_d_t_3": Theta_req_d_t_i[2], "Theta_req_d_t_4": Theta_req_d_t_i[3], "Theta_req_d_t_5": Theta_req_d_t_i[4],
            })

        elif skin.underfloor_air_conditioning_air_supply:
            for i in range(2):  # 1F居室のみ(i=0,1)損失分を補正
                # CHECK: 床下温度が i(部屋) で変わるが問題ないか
                Theta_uf_d_t, Theta_g_surf_d_t, *others  \
                    = algo.calc_Theta(  # 旧床下空調-1st
                        house.region, house.A_A, house.A_MR, house.A_OR, skin.Q, skin.r_A_ufac, skin.underfloor_insulation,
                        Theta_req_d_t_i[i], Theta_ex_d_t, V_dash_supply_d_t_i[i],
                        '', load.L_H_d_t_i, load.L_CS_d_t_i)

                match ac_setting:
                    case HeatingAcSetting():
                        mask = Theta_req_d_t_i[i] > Theta_uf_d_t
                    case CoolingAcSetting():
                        mask = Theta_req_d_t_i[i] < Theta_uf_d_t
                    case _:
                        raise ValueError

                Theta_req_d_t_i[i] = np.where(mask,
                                    # 熱源機出口 -> 居室床下までの温度低下分を見込む
                                    Theta_req_d_t_i[i] + (Theta_req_d_t_i[i] - Theta_uf_d_t),
                                    Theta_req_d_t_i[i])

            assert np.shape(Theta_req_d_t_i)==(5, 8760), "想定外の行列数です"

        # (15)　熱源機の出口における絶対湿度
        X_hs_out_d_t = dc.get_X_hs_out_d_t(X_NR_d_t, X_req_d_t_i, V_dash_supply_d_t_i, X_hs_out_min_C_d_t, L_star_CL_d_t_i, house.region)

        # 式(14)(46)(48)の条件に合わせてTheta_NR_d_tを初期化
        Theta_NR_d_t = np.zeros(24 * 365)

        # (17)　冷房時の熱源機の出口における空気温度の最低値
        Theta_hs_out_min_C_d_t = dc.get_Theta_hs_out_min_C_d_t(Theta_star_hs_in_d_t, Q_hs_max_CS_d_t, V_dash_supply_d_t_i)

        # (16)　暖房時の熱源機の出口における空気温度の最高値
        Theta_hs_out_max_H_d_t = dc.get_Theta_hs_out_max_H_d_t(Theta_star_hs_in_d_t, Q_hs_max_H_d_t, V_dash_supply_d_t_i)

        # L_star_H_d_t_i，L_star_CS_d_t_iの暖冷房区画1～5を合算し0以上だった場合の順序で計算
        # (14)　熱源機の出口における空気温度
        Theta_hs_out_d_t = dc.get_Theta_hs_out_d_t(ac_setting.VAV, Theta_req_d_t_i, V_dash_supply_d_t_i,
                                                L_star_H_d_t_i, L_star_CS_d_t_i, house.region, Theta_NR_d_t,
                                                Theta_hs_out_max_H_d_t, Theta_hs_out_min_C_d_t)

        # (43)　暖冷房区画𝑖の吹き出し風量
        V_supply_d_t_i_before = dc.get_V_supply_d_t_i(L_star_H_d_t_i, L_star_CS_d_t_i, Theta_sur_d_t_i, l_duct_i, Theta_star_HBR_d_t
                                                    , V_vent_g_i, V_dash_supply_d_t_i, ac_setting.VAV, house.region, Theta_hs_out_d_t)
        V_supply_d_t_i = jjj_vsupcap.cap_V_supply_d_t_i(v_supply_cap_dto, V_supply_d_t_i_before, V_dash_supply_d_t_i
                                                    , V_vent_g_i, house.region, V_hs_dsgn_H, V_hs_dsgn_C, print_exec=True)

        # (41)　暖冷房区画𝑖の吹き出し温度
        Theta_supply_d_t_i = dc.get_Thata_supply_d_t_i(Theta_sur_d_t_i, Theta_hs_out_d_t, Theta_star_HBR_d_t, l_duct_i,
                                                       V_supply_d_t_i, L_star_H_d_t_i, L_star_CS_d_t_i, house.region)
        _logger.NDdebug("Theta_supply_d_t_1", Theta_supply_d_t_i[0])
        _logger.NDdebug("Theta_supply_d_t_2", Theta_supply_d_t_i[1])
        _logger.NDdebug("Theta_supply_d_t_3", Theta_supply_d_t_i[2])
        _logger.NDdebug("Theta_supply_d_t_4", Theta_supply_d_t_i[3])
        _logger.NDdebug("Theta_supply_d_t_5", Theta_supply_d_t_i[4])

        # 実行条件: 床下新空調ロジックのみ
        if new_ufac.new_ufac_flg == 床下空調ロジック.変更する:
            # θuf の本計算
            Theta_uf_d_t, Theta_g_surf_d_t, *others  \
                = algo.calc_Theta(  # 新床下空調-2nd
                    region = house.region,
                    A_A = house.A_A,
                    A_MR = house.A_MR,
                    A_OR = house.A_OR,
                    Q = skin.Q,
                    r_A_ufvnt = skin.r_A_ufac,  # 床下換気ではなく床下空調のため
                    underfloor_insulation = skin.underfloor_insulation,
                    Theta_sa_d_t = Theta_hs_out_d_t,  # ★
                    Theta_ex_d_t = Theta_ex_d_t,
                    # 熱源機出口温度から吹き出し温度を計算する
                    V_sa_d_t_A = np.sum(V_dash_supply_d_t_i[:2, :], axis=0),  # i=1,2
                    H_OR_C = "",
                    L_dash_H_R_d_t_i = load.L_dash_H_R_d_t_i,
                    L_dash_CS_R_d_t_i = load.L_dash_CS_R_d_t_i,
                    calc_backwards = False,  # ここでは θuf の従来計算のみ
                    new_ufac = new_ufac,
                    new_ufac_df = new_ufac_df
                )

            # 床下・床上の熱貫流分だけ 目標床下温度からわずかな中和がある
            Theta_supply_d_t_i  \
                = np.vstack([
                    # NOTE: i=1,2(1階居室)は床下を通して出口温度が中和されたものになる
                    np.tile(Theta_uf_d_t, (2, 1)),
                    # CHECK: i=3,4,5(2階居室)は床下通さないので中和がなく高温なのは問題ないか
                    Theta_supply_d_t_i[2:, :]
                ])
            assert np.shape(Theta_supply_d_t_i)==(5, 8760), "想定外の行列数です"

            new_ufac_df.update_df({
                "Theta_hs_out_d_t": Theta_hs_out_d_t,
                "Theta_uf_d_t": Theta_uf_d_t,
                "Theta_supply_d_t_1": Theta_supply_d_t_i[0], "Theta_supply_d_t_2": Theta_supply_d_t_i[1], "Theta_supply_d_t_3": Theta_supply_d_t_i[2], "Theta_supply_d_t_4": Theta_supply_d_t_i[3], "Theta_supply_d_t_5": Theta_supply_d_t_i[4]
            })
        elif skin.underfloor_air_conditioning_air_supply == True:
            for i in range(2):  #i=0,1
                Theta_uf_d_t, Theta_g_surf_d_t, *others  \
                    = algo.calc_Theta(  # 旧床下空調-2nd
                        house.region, house.A_A, house.A_MR, house.A_OR, skin.Q, skin.r_A_ufac, skin.underfloor_insulation,
                        Theta_supply_d_t_i[i], Theta_ex_d_t, V_dash_supply_d_t_i[i],
                        '', load.L_H_d_t_i, load.L_CS_d_t_i)

                match ac_setting:
                    case HeatingAcSetting():
                        mask = Theta_supply_d_t_i[i] > Theta_uf_d_t
                    case CoolingAcSetting():
                        mask = Theta_supply_d_t_i[i] < Theta_uf_d_t
                    case _:
                        raise ValueError

                Theta_supply_d_t_i[i] = np.where(mask, Theta_uf_d_t, Theta_supply_d_t_i[i])

        _logger.NDdebug("Theta_supply_d_t_1", Theta_supply_d_t_i[0])
        _logger.NDdebug("Theta_supply_d_t_2", Theta_supply_d_t_i[1])
        _logger.NDdebug("Theta_supply_d_t_3", Theta_supply_d_t_i[2])
        _logger.NDdebug("Theta_supply_d_t_4", Theta_supply_d_t_i[3])
        _logger.NDdebug("Theta_supply_d_t_5", Theta_supply_d_t_i[4])

        # (46) 暖冷房区画𝑖の実際の居室の室温
        if new_ufac.new_ufac_flg == 床下空調ロジック.変更する:
            HCM = np.array(climate.get_HCM_d_t())
            A_s_ufac_i, _ = jjj_ufac_dc.get_A_s_ufac_i(house.A_A, house.A_MR, house.A_OR)
            Theta_HBR_d_t_i = np.hstack([
                get_Theta_HBR_i(
                    Theta_star_HBR = Theta_star_HBR_d_t[t],
                    V_supply_i = V_supply_d_t_i[:, t:t+1],
                    Theta_supply_i = Theta_supply_d_t_i[:, t:t+1],
                    U_prt = U_prt,
                    A_prt_i = A_prt_i.reshape(-1,1)[:5, :],
                    Q = skin.Q,
                    A_HCZ_i = A_HCZ_i.reshape(-1,1),
                    L_star_H_i = L_star_H_d_t_i[:, t:t+1],
                    L_star_CS_i = L_star_CS_d_t_i[:, t:t+1],
                    HCM = HCM[t],
                    A_s_ufac_i = A_s_ufac_i[:5, :],
                    Theta_uf = Theta_uf_d_t[t],
                ) for t in range(24*365)
            ])
        else:
            # 改変なし元式
            Theta_HBR_d_t_i  \
                = dc.get_Theta_HBR_d_t_i(
                    Theta_star_HBR_d_t, V_supply_d_t_i, Theta_supply_d_t_i,
                    U_prt, A_prt_i, skin.Q, A_HCZ_i,
                    L_star_H_d_t_i, L_star_CS_d_t_i, house.region)

        # (48) 実際の非居室の室温
        if new_ufac.new_ufac_flg == 床下空調ロジック.変更する:
            Theta_NR_d_t = np.array([
                get_Theta_NR(
                    Theta_star_NR = Theta_star_NR_d_t[t],
                    Theta_star_HBR = Theta_star_HBR_d_t[t],
                    Theta_HBR_i = Theta_HBR_d_t_i[:, t:t+1],
                    A_NR = A_NR,
                    V_vent_l_NR = V_vent_l_NR_d_t[t],
                    V_dash_supply_i = V_dash_supply_d_t_i[:, t:t+1],
                    V_supply_i = V_supply_d_t_i[:, t:t+1],
                    U_prt = U_prt,
                    A_prt_i = A_prt_i.reshape(-1,1),
                    Q = skin.Q,
                    Theta_uf = Theta_uf_d_t[t],
                    r_A_NR_1F = r_A_NR_uf_1F
                ) for t in range(24*365)
            ])
        else:
            # 改変なし元式
            Theta_NR_d_t  \
                = dc.get_Theta_NR_d_t(
                    Theta_star_NR_d_t, Theta_star_HBR_d_t, Theta_HBR_d_t_i,
                    A_NR, V_vent_l_NR_d_t, V_dash_supply_d_t_i, V_supply_d_t_i,
                    U_prt, A_prt_i, skin.Q)

    ### 熱繰越 / 非熱繰越 の分岐が終了 -> 以降、共通の処理 ###

    # NOTE: 繰越の有無によってCSV出力が異ならないよう df_output の処理は以降に限定する
    _logger.NDdebug("Theta_HBR_d_t_1", Theta_HBR_d_t_i[0])
    _logger.NDdebug("Theta_HBR_d_t_2", Theta_HBR_d_t_i[1])
    _logger.NDdebug("Theta_HBR_d_t_3", Theta_HBR_d_t_i[2])
    _logger.NDdebug("Theta_HBR_d_t_4", Theta_HBR_d_t_i[3])
    _logger.NDdebug("Theta_HBR_d_t_5", Theta_HBR_d_t_i[4])
    _logger.NDdebug("Theta_NR_d_t", Theta_NR_d_t)

    if carryover_heat_dto.carry_over_heat == 過剰熱量繰越計算.行う:
        df_carryover_output = df_carryover_output.assign(
            carryovers_i_1 = carryovers[0],
            carryovers_i_2 = carryovers[1],
            carryovers_i_3 = carryovers[2],
            carryovers_i_4 = carryovers[3],
            carryovers_i_5 = carryovers[4]
        )
        match (q_hs_rtd_H(), q_hs_rtd_C()):
            case (None, None):
                raise Exception("q_hs_rtd_H, q_hs_rtd_C はどちらかのみを前提")
            case (_, None):
                df_carryover_output.to_csv(
                    case_name + jjj_consts.version_info() + '_H_carryover_output.csv',
                    encoding = 'cp932')
            case (None, _):
                df_carryover_output.to_csv(
                    case_name + jjj_consts.version_info() + '_C_carryover_output.csv',
                    encoding = 'cp932')
            case (_, _):
                raise Exception("q_hs_rtd_H, q_hs_rtd_C はどちらかのみを前提")

    """ 熱損失・熱取得を含む負荷バランス時の熱負荷 - 熱損失・熱取得を含む負荷バランス時(2) """
    df_output = df_output.assign(
        L_star_CS_d_t_i_1 = L_star_CS_d_t_i[0],
        L_star_CS_d_t_i_2 = L_star_CS_d_t_i[1],
        L_star_CS_d_t_i_3 = L_star_CS_d_t_i[2],
        L_star_CS_d_t_i_4 = L_star_CS_d_t_i[3],
        L_star_CS_d_t_i_5 = L_star_CS_d_t_i[4]
    )
    df_output = df_output.assign(
        L_star_H_d_t_i_1 = L_star_H_d_t_i[0],
        L_star_H_d_t_i_2 = L_star_H_d_t_i[1],
        L_star_H_d_t_i_3 = L_star_H_d_t_i[2],
        L_star_H_d_t_i_4 = L_star_H_d_t_i[3],
        L_star_H_d_t_i_5 = L_star_H_d_t_i[4]
    )

    """ 最大暖冷房能力 """
    df_output = df_output.assign(
        # NOTE: タイプ毎に出力する変数の数を変えないようIFなどの分岐はしない
        # 以下タイプ(1, 3)
        L_star_CL_d_t = L_star_CL_d_t if "L_star_CL_d_t" in locals() else None,  # (33)
        L_star_CS_d_t = L_star_CS_d_t if "L_star_CS_d_t" in locals() else None,  # (32)
        L_star_dash_CL_d_t = L_star_dash_CL_d_t if "L_star_dash_CL_d_t" in locals() else None,  # (30)
        L_star_dash_C_d_t = L_star_dash_C_d_t if "L_star_dash_C_d_t" in locals() else None,   # (29)
        # 以下タイプ(2, 4)
        C_df_H_d_t = C_df_H_d_t if "C_df_H_d_t" in locals() else None,  # (24)
        Q_r_max_H_d_t = Q_r_max_H_d_t if "Q_r_max_H_d_t" in locals() else None,
        Q_r_max_C_d_t = Q_r_max_C_d_t if "Q_r_max_C_d_t" in locals() else None,
        L_max_CL_d_t = L_max_CL_d_t if "L_max_CL_d_t" in locals() else None,
        L_dash_CL_d_t = L_dash_CL_d_t if "L_dash_CL_d_t" in locals() else None,
        L_dash_C_d_t  = L_dash_C_d_t if "L_dash_C_d_t" in locals() else None,
    )
    df_output3 = df_output3.assign(
        # 以下タイプ(2, 4)
        q_r_max_H = q_r_max_H if "q_r_max_+H" in locals() else None,
        q_r_max_C = q_r_max_C if "q_r_max_C" in locals() else None,
        SHF_L_min_c = SHF_L_min_c if "SHF_L_min_c" in locals() else None,
    )
    df_output['SHF_dash_d_t'] = SHF_dash_d_t
    df_output = df_output.assign(
        Q_hs_max_C_d_t  = Q_hs_max_C_d_t,
        Q_hs_max_CL_d_t = Q_hs_max_CL_d_t,
        Q_hs_max_CS_d_t = Q_hs_max_CS_d_t,
        Q_hs_max_H_d_t  = Q_hs_max_H_d_t,
    )

    """ 熱源機の出口 - 負荷バランス時 """
    df_output['X_star_hs_in_d_t'] = X_star_hs_in_d_t
    df_output['Theta_star_hs_in_d_t'] = Theta_star_hs_in_d_t

    """ 熱源機の出口 - 熱源機の出口 """
    df_output['X_star_hs_in_d_t'] = X_star_hs_in_d_t
    df_output['Theta_star_hs_in_d_t'] = Theta_star_hs_in_d_t
    df_output['X_hs_out_min_C_d_t'] = X_hs_out_min_C_d_t
    df_output = df_output.assign(
        X_req_d_t_1 = X_req_d_t_i[0],
        X_req_d_t_2 = X_req_d_t_i[1],
        X_req_d_t_3 = X_req_d_t_i[2],
        X_req_d_t_4 = X_req_d_t_i[3],
        X_req_d_t_5 = X_req_d_t_i[4]
    )
    df_output = df_output.assign(
        Theta_req_d_t_1 = Theta_req_d_t_i[0],
        Theta_req_d_t_2 = Theta_req_d_t_i[1],
        Theta_req_d_t_3 = Theta_req_d_t_i[2],
        Theta_req_d_t_4 = Theta_req_d_t_i[3],
        Theta_req_d_t_5 = Theta_req_d_t_i[4]
    )
    df_output['X_hs_out_d_t'] = X_hs_out_d_t
    df_output = df_output.assign(
        Theta_hs_out_min_C_d_t = Theta_hs_out_min_C_d_t,
        Theta_hs_out_max_H_d_t = Theta_hs_out_max_H_d_t,
        Theta_hs_out_d_t = Theta_hs_out_d_t,
    )

    """吹出口 - 吹出口"""
    # NOTE: 2024/02/14 WG の話で出力してほしいデータになりました
    df_output = df_output.assign(
        V_supply_d_t_1_before = V_supply_d_t_i_before[0] if V_supply_d_t_i_before is not None else None,
        V_supply_d_t_2_before = V_supply_d_t_i_before[1] if V_supply_d_t_i_before is not None else None,
        V_supply_d_t_3_before = V_supply_d_t_i_before[2] if V_supply_d_t_i_before is not None else None,
        V_supply_d_t_4_before = V_supply_d_t_i_before[3] if V_supply_d_t_i_before is not None else None,
        V_supply_d_t_5_before = V_supply_d_t_i_before[4] if V_supply_d_t_i_before is not None else None,
    )
    df_output = df_output.assign(
        V_supply_d_t_1 = V_supply_d_t_i[0],
        V_supply_d_t_2 = V_supply_d_t_i[1],
        V_supply_d_t_3 = V_supply_d_t_i[2],
        V_supply_d_t_4 = V_supply_d_t_i[3],
        V_supply_d_t_5 = V_supply_d_t_i[4]
    )
    df_output = df_output.assign(
        Theta_supply_d_t_1 = Theta_supply_d_t_i[0],
        Theta_supply_d_t_2 = Theta_supply_d_t_i[1],
        Theta_supply_d_t_3 = Theta_supply_d_t_i[2],
        Theta_supply_d_t_4 = Theta_supply_d_t_i[3],
        Theta_supply_d_t_5 = Theta_supply_d_t_i[4]
    )

    """ 吹出口 - 実際 """
    df_output = df_output.assign(
        Theta_HBR_d_t_1 = Theta_HBR_d_t_i[0],
        Theta_HBR_d_t_2 = Theta_HBR_d_t_i[1],
        Theta_HBR_d_t_3 = Theta_HBR_d_t_i[2],
        Theta_HBR_d_t_4 = Theta_HBR_d_t_i[3],
        Theta_HBR_d_t_5 = Theta_HBR_d_t_i[4],
        Theta_NR_d_t = Theta_NR_d_t
    )

    """ 吹出口 - 熱源機の出口 """
    # L_star_H_d_t_i，L_star_CS_d_t_iの暖冷房区画1～5を合算し0以下だった場合の為に再計算
    # (14)　熱源機の出口における空気温度
    Theta_hs_out_d_t = dc.get_Theta_hs_out_d_t(ac_setting.VAV, Theta_req_d_t_i, V_dash_supply_d_t_i,
                                            L_star_H_d_t_i, L_star_CS_d_t_i, house.region, Theta_NR_d_t,
                                            Theta_hs_out_max_H_d_t, Theta_hs_out_min_C_d_t)
    df_output['Theta_hs_out_d_t'] = Theta_hs_out_d_t

    """ 吹出口 - 吹出口 """
    # (42)　暖冷房区画𝑖の吹き出し絶対湿度
    X_supply_d_t_i = dc.get_X_supply_d_t_i(X_star_HBR_d_t, X_hs_out_d_t, L_star_CL_d_t_i, house.region)
    df_output = df_output.assign(
        X_supply_d_t_1 = X_supply_d_t_i[0],
        X_supply_d_t_2 = X_supply_d_t_i[1],
        X_supply_d_t_3 = X_supply_d_t_i[2],
        X_supply_d_t_4 = X_supply_d_t_i[3],
        X_supply_d_t_5 = X_supply_d_t_i[4]
    )

    """ 熱源機の入口 - 熱源機の風量の計算 """
    # (35)　熱源機の風量のうちの全般換気分
    V_hs_vent_d_t = dc.get_V_hs_vent_d_t(V_vent_g_i, ac_setting.general_ventilation)  # 従来式通り
    df_output['V_hs_vent_d_t'] = V_hs_vent_d_t

    # (34)　熱源機の風量
    V_hs_supply_d_t = dc.get_V_hs_supply_d_t(V_supply_d_t_i)
    df_output['V_hs_supply_d_t'] = V_hs_supply_d_t

    """ 熱源機の入口 - 熱源機の入口 """
    # (13)　熱源機の入口における絶対湿度
    X_hs_in_d_t = dc.get_X_hs_in_d_t(X_NR_d_t)
    df_output['X_hs_in_d_t'] = X_hs_in_d_t

    # (12)　熱源機の入口における空気温度
    Theta_hs_in_d_t = dc.get_Theta_hs_in_d_t(Theta_NR_d_t)
    df_output['Theta_hs_in_d_t'] = Theta_hs_in_d_t

    """ まとめ - 実際の暖冷房負荷 """
    # (7)　間仕切りの熱取得を含む実際の冷房潜熱負荷
    if carryover_heat_dto.carry_over_heat == 過剰熱量繰越計算.行う:
        L_dash_CL_d_t_i = np.clip(
            dc.get_L_dash_CL_d_t_i(V_supply_d_t_i, X_HBR_d_t_i, X_supply_d_t_i, house.region), # 従来式
            0, None)
    else:
        L_dash_CL_d_t_i = dc.get_L_dash_CL_d_t_i(V_supply_d_t_i, X_HBR_d_t_i, X_supply_d_t_i, house.region)
    df_output = df_output.assign(
        L_dash_CL_d_t_1 = L_dash_CL_d_t_i[0],
        L_dash_CL_d_t_2 = L_dash_CL_d_t_i[1],
        L_dash_CL_d_t_3 = L_dash_CL_d_t_i[2],
        L_dash_CL_d_t_4 = L_dash_CL_d_t_i[3],
        L_dash_CL_d_t_5 = L_dash_CL_d_t_i[4]
    )
    # (6)　間仕切りの熱取得を含む実際の冷房顕熱負荷
    if carryover_heat_dto.carry_over_heat == 過剰熱量繰越計算.行う:
        L_dash_CS_d_t_i = np.clip(
            dc.get_L_dash_CS_d_t_i(V_supply_d_t_i, Theta_supply_d_t_i, Theta_HBR_d_t_i, house.region), # 従来式
            0, None)
    else:
        L_dash_CS_d_t_i = dc.get_L_dash_CS_d_t_i(V_supply_d_t_i, Theta_supply_d_t_i, Theta_HBR_d_t_i, house.region)
    df_output = df_output.assign(
        L_dash_CS_d_t_1 = L_dash_CS_d_t_i[0],
        L_dash_CS_d_t_2 = L_dash_CS_d_t_i[1],
        L_dash_CS_d_t_3 = L_dash_CS_d_t_i[2],
        L_dash_CS_d_t_4 = L_dash_CS_d_t_i[3],
        L_dash_CS_d_t_5 = L_dash_CS_d_t_i[4]
    )
    # (5)　間仕切りの熱損失を含む実際の暖房負荷
    if carryover_heat_dto.carry_over_heat == 過剰熱量繰越計算.行う:
        L_dash_H_d_t_i = np.clip(
            dc.get_L_dash_H_d_t_i(V_supply_d_t_i, Theta_supply_d_t_i, Theta_HBR_d_t_i, house.region), # 従来式
            0, None)
    else:
        L_dash_H_d_t_i = dc.get_L_dash_H_d_t_i(V_supply_d_t_i, Theta_supply_d_t_i, Theta_HBR_d_t_i, house.region)
    df_output = df_output.assign(
        L_dash_H_d_t_1 = L_dash_H_d_t_i[0],
        L_dash_H_d_t_2 = L_dash_H_d_t_i[1],
        L_dash_H_d_t_3 = L_dash_H_d_t_i[2],
        L_dash_H_d_t_4 = L_dash_H_d_t_i[3],
        L_dash_H_d_t_5 = L_dash_H_d_t_i[4]
    )

    """ まとめ - 未処理負荷 """
    # (4)　冷房設備機器の未処理冷房潜熱負荷
    Q_UT_CL_d_t_i = dc.get_Q_UT_CL_d_t_i(L_star_CL_d_t_i, L_dash_CL_d_t_i)
    df_output = df_output.assign(
        Q_UT_CL_d_t_1 = Q_UT_CL_d_t_i[0],
        Q_UT_CL_d_t_2 = Q_UT_CL_d_t_i[1],
        Q_UT_CL_d_t_3 = Q_UT_CL_d_t_i[2],
        Q_UT_CL_d_t_4 = Q_UT_CL_d_t_i[3],
        Q_UT_CL_d_t_5 = Q_UT_CL_d_t_i[4]
    )
    # (3)　冷房設備機器の未処理冷房顕熱負荷
    Q_UT_CS_d_t_i = dc.get_Q_UT_CS_d_t_i(L_star_CS_d_t_i, L_dash_CS_d_t_i)
    df_output = df_output.assign(
        Q_UT_CS_d_t_1 = Q_UT_CS_d_t_i[0],
        Q_UT_CS_d_t_2 = Q_UT_CS_d_t_i[1],
        Q_UT_CS_d_t_3 = Q_UT_CS_d_t_i[2],
        Q_UT_CS_d_t_4 = Q_UT_CS_d_t_i[3],
        Q_UT_CS_d_t_5 = Q_UT_CS_d_t_i[4]
    )
    # (2)　暖房設備機器等の未処理暖房負荷
    Q_UT_H_d_t_i = dc.get_Q_UT_H_d_t_i(L_star_H_d_t_i, L_dash_H_d_t_i)
    df_output = df_output.assign(
        Q_UT_H_d_t_1 = Q_UT_H_d_t_i[0],
        Q_UT_H_d_t_2 = Q_UT_H_d_t_i[1],
        Q_UT_H_d_t_3 = Q_UT_H_d_t_i[2],
        Q_UT_H_d_t_4 = Q_UT_H_d_t_i[3],
        Q_UT_H_d_t_5 = Q_UT_H_d_t_i[4]
    )

    """ まとめ - 一次エネルギー """
    match ac_setting:
        case HeatingAcSetting():
            # 暖房: 未処理暖房負荷の設計一次エネルギー消費量相当値
            alpha_UT_H_A: float = get_alpha_UT_H_A(house.region)
            Q_UT_H_A_d_t: np.ndarray = np.sum(Q_UT_H_d_t_i, axis=0)
            E_UT_d_t = Q_UT_H_A_d_t * alpha_UT_H_A
            df_output['E_UT_H_d_t'] = E_UT_d_t
        case CoolingAcSetting():
            # (1)　冷房設備の未処理冷房負荷の設計一次エネルギー消費量相当値
            E_UT_d_t = dc.get_E_C_UT_d_t(Q_UT_CL_d_t_i, Q_UT_CS_d_t_i, house.region)
            df_output['E_UT_C_d_t'] = E_UT_d_t
        case _:
            raise ValueError("ac_setting must be HeatingAcSetting or CoolingAcSetting")

    if export_outputs:
        # 床下空調新ロジック調査用変数の出力
        if new_ufac.new_ufac_flg == 床下空調ロジック.変更する:
            filename = case_name + jjj_consts.version_info() + flg_char() + "_output_uf.csv"
            # ネスト関数内で更新されているデータフレーム
            new_ufac_df.export_to_csv(filename)

        match(q_hs_rtd_H(), q_hs_rtd_C()):
            case(None, None):
                raise Exception("q_hs_rtd_H, q_hs_rtd_C はどちらかのみを前提")
            case(_, None):
                df_output3.to_csv(case_name + jjj_consts.version_info() + '_H_output3.csv', encoding = 'cp932')
                df_output2.to_csv(case_name + jjj_consts.version_info() + '_H_output4.csv', encoding = 'cp932')
                df_output.to_csv(case_name  + jjj_consts.version_info() + '_H_output5.csv', encoding = 'cp932')
            case(None, _):
                df_output3.to_csv(case_name + jjj_consts.version_info() + '_C_output3.csv', encoding = 'cp932')
                df_output2.to_csv(case_name + jjj_consts.version_info() + '_C_output4.csv', encoding = 'cp932')
                df_output.to_csv(case_name  + jjj_consts.version_info() + '_C_output5.csv', encoding = 'cp932')
            case(_, _):
                raise Exception("q_hs_rtd_H, q_hs_rtd_C はどちらかのみを前提")

    Theta_uf_actual_d_t = (
        Theta_uf_d_t
        if new_ufac.new_ufac_flg == 床下空調ロジック.変更する
        else None
    )
    Theta_uf_M_actual_d_t = None
    if new_ufac.new_ufac_flg == 床下空調ロジック.変更する:
        Theta_uf_for_ground_d_t = (
            Theta_uf_actual_d_t
            if Theta_uf_ground_feedback_d_t is None
            else Theta_uf_ground_feedback_d_t
        )
        response_d_t = calc_sum_Theta_dash_g_surf_A_m_d_t(
            Theta_uf_for_ground_d_t,
            Theta_ex_d_t,
            skin.underfloor_insulation,
            Theta_g_avg=Theta_g_avg,
        )
        Theta_uf_M_actual_d_t = calc_intermediate_floor_temperature(
            np.sum(V_dash_supply_d_t_i[:2, :], axis=0),
            Theta_star_HBR_d_t,
            Theta_ex_d_t,
            response_d_t,
            U_s_supply,
            A_s_ufac_A,
            phi,
            L_uf,
            jjj_consts.R_g,
            Phi_A_0,
            Theta_g_avg,
        )

    return E_UT_d_t, \
            Theta_hs_out_d_t, Theta_hs_in_d_t, \
            X_hs_out_d_t, X_hs_in_d_t, V_hs_supply_d_t, V_hs_vent_d_t, \
            Theta_uf_actual_d_t, Theta_uf_M_actual_d_t


def _solve_shared_ground_feedback(
        heating_args: dict,
        cooling_args: dict,
        H: np.ndarray,
        C: np.ndarray) -> tuple[np.ndarray, int, float]:
    """暖房・冷房を一つの年間床下温度履歴で反復計算する。"""
    heating_iteration_args = {
        **{key: value for key, value in heating_args.items()
           if key != "new_ufac_df"},
        "export_outputs": False,
    }
    cooling_iteration_args = {
        **{key: value for key, value in cooling_args.items()
           if key != "new_ufac_df"},
        "export_outputs": False,
    }

    heating = _calc_Q_UT_A_once(
        **heating_iteration_args,
        new_ufac_df=UfVarsDataFrame(),
    )
    cooling = _calc_Q_UT_A_once(
        **cooling_iteration_args,
        new_ufac_df=UfVarsDataFrame(),
    )
    assert heating[7] is not None
    assert cooling[7] is not None
    assert heating[8] is not None
    Theta_uf_feedback_d_t = merge_annual_floor_temperature(
        heating[7], cooling[7], heating[8], H, C
    )

    max_delta = float("inf")
    for iteration_count in range(1, GROUND_FEEDBACK_MAX_ITERATIONS + 1):
        heating = _calc_Q_UT_A_once(
            **heating_iteration_args,
            new_ufac_df=UfVarsDataFrame(),
            Theta_uf_ground_feedback_d_t=Theta_uf_feedback_d_t,
        )
        cooling = _calc_Q_UT_A_once(
            **cooling_iteration_args,
            new_ufac_df=UfVarsDataFrame(),
            Theta_uf_ground_feedback_d_t=Theta_uf_feedback_d_t,
        )
        assert heating[7] is not None
        assert cooling[7] is not None
        assert heating[8] is not None
        Theta_uf_updated_d_t = merge_annual_floor_temperature(
            heating[7], cooling[7], heating[8], H, C
        )
        max_delta = float(np.max(np.abs(
            Theta_uf_updated_d_t - Theta_uf_feedback_d_t
        )))
        Theta_uf_feedback_d_t = Theta_uf_updated_d_t
        if max_delta <= GROUND_FEEDBACK_TOLERANCE:
            return Theta_uf_feedback_d_t, iteration_count, max_delta

    raise RuntimeError(
        "暖房・冷房共通の地盤応答フィードバックが収束しませんでした: "
        f"iterations={GROUND_FEEDBACK_MAX_ITERATIONS}, "
        f"max_delta={max_delta:.6g} K"
    )


# NOTE: section4_2 の同名の関数の改変版
@jjj_cloning
@inject
def calc_Q_UT_A(
        case_name: CaseName,
        climateFile: ClimateFile,
        house: HouseInfo,
        ac_setting: ActiveAcSetting,
        skin: OuterSkin,
        heat_CRAC: HeatCRACSpec,
        cool_CRAC: CoolCRACSpec,
        new_ufac: UnderfloorAc,
        new_ufac_df: UfVarsDataFrame,
        v_min_heat_input: HeatMinVolumeInput,
        v_min_cool_input: CoolMinVolumeInput,
        V_hs_dsgn_H: VHS_DSGN_H,
        V_hs_dsgn_C: VHS_DSGN_C,
        v_supply_cap_dto: VSupplyCapDto,
        carryover_heat_dto: CarryoverHeatDto,
        load: Load_DTI,
        annual_ground_feedback_context: AnnualGroundFeedbackContext = None):
    """地盤応答に前時刻の実際の床下温度を反映して未処理負荷を計算する。

    ベクトル計算の構成を維持したまま、1回目で得た実際の床下温度を
    次の計算の地盤応答へ渡す。calc_sum_Theta_dash_g_surf_A_m_d_t 内では
    時刻tの応答に時刻t-1の熱流が使われるため、当時刻値の循環参照は生じない。
    """
    args = dict(
        case_name=case_name,
        climateFile=climateFile,
        house=house,
        ac_setting=ac_setting,
        skin=skin,
        heat_CRAC=heat_CRAC,
        cool_CRAC=cool_CRAC,
        new_ufac=new_ufac,
        new_ufac_df=new_ufac_df,
        v_min_heat_input=v_min_heat_input,
        v_min_cool_input=v_min_cool_input,
        V_hs_dsgn_H=V_hs_dsgn_H,
        V_hs_dsgn_C=V_hs_dsgn_C,
        v_supply_cap_dto=v_supply_cap_dto,
        carryover_heat_dto=carryover_heat_dto,
        load=load,
    )

    if new_ufac.new_ufac_flg != 床下空調ロジック.変更する:
        result = _calc_Q_UT_A_once(**args)
        return result[:7]

    if annual_ground_feedback_context is not None:
        context = annual_ground_feedback_context
        if context.Theta_uf_feedback_d_t is None:
            common_args = {
                **{key: value for key, value in args.items()
                   if key not in {
                       "ac_setting", "V_hs_dsgn_H", "V_hs_dsgn_C",
                       "new_ufac_df",
                   }},
            }
            heating_args = {
                **common_args,
                "ac_setting": context.heat_ac_setting,
                "V_hs_dsgn_H": context.V_hs_dsgn_H,
                "V_hs_dsgn_C": 0.0,
                "new_ufac_df": UfVarsDataFrame(),
            }
            cooling_args = {
                **common_args,
                "ac_setting": context.cool_ac_setting,
                "V_hs_dsgn_H": 0.0,
                "V_hs_dsgn_C": context.V_hs_dsgn_C,
                "new_ufac_df": UfVarsDataFrame(),
            }
            H, C, _ = dc.get_season_array_d_t(house.region)
            feedback, iteration_count, max_delta = \
                _solve_shared_ground_feedback(
                    heating_args, cooling_args, H, C
                )
            context.Theta_uf_feedback_d_t = feedback
            _logger.info(
                "暖房・冷房共通の地盤応答フィードバック: "
                f"iterations={iteration_count}, max_delta={max_delta:.6g} K"
            )

        result = _calc_Q_UT_A_once(
            **args,
            Theta_uf_ground_feedback_d_t=context.Theta_uf_feedback_d_t,
        )
        return result[:7]

    # 反復中の調査用CSVに中間結果を混ぜない。収束後の1回だけ
    # 呼出側から渡されたDataFrameと出力ファイルへ記録する。
    iteration_args = {
        **{key: value for key, value in args.items() if key != "new_ufac_df"},
        "export_outputs": False,
    }
    result = _calc_Q_UT_A_once(
        **iteration_args,
        new_ufac_df=UfVarsDataFrame(),
    )
    Theta_uf_feedback_d_t = result[7]
    assert Theta_uf_feedback_d_t is not None

    converged = False
    max_delta = float("inf")
    iteration_count = 0
    for iteration_count in range(1, GROUND_FEEDBACK_MAX_ITERATIONS + 1):
        updated = _calc_Q_UT_A_once(
            **iteration_args,
            new_ufac_df=UfVarsDataFrame(),
            Theta_uf_ground_feedback_d_t=Theta_uf_feedback_d_t,
        )
        Theta_uf_updated_d_t = updated[7]
        assert Theta_uf_updated_d_t is not None
        max_delta = float(np.max(np.abs(
            Theta_uf_updated_d_t - Theta_uf_feedback_d_t
        )))
        result = updated
        Theta_uf_feedback_d_t = Theta_uf_updated_d_t
        if max_delta <= GROUND_FEEDBACK_TOLERANCE:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            "地盤応答の床下温度フィードバックが収束しませんでした: "
            f"iterations={iteration_count}, max_delta={max_delta:.6g} K"
        )

    _logger.info(
        "地盤応答の床下温度フィードバック: "
        f"iterations={iteration_count}, max_delta={max_delta:.6g} K"
    )

    # 収束した前時刻床下温度を使って最終値を1回だけ出力する。
    result = _calc_Q_UT_A_once(
        **args,
        Theta_uf_ground_feedback_d_t=Theta_uf_feedback_d_t,
    )
    return result[:7]
