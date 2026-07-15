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

        …12475 tokens truncated…  L_star_dash_CL_d_t = L_star_dash_CL_d_t if "L_star_dash_CL_d_t" in locals() else None,  # (30)
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
