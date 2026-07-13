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
        load: Load_DTI):
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
    # 地盤の不易層温度と助走計算による吸熱応答成分の合計 (床下→地盤 熱損失計算用)
    # Theta_ex_d_t に依存するが ループ内では変わらないため事前に計算する
    Theta_g_avg = climate.get_Theta_g_avg()
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
        U_s_load = algo.get_U_s_vert(house.region, skin.Q)  # 標準住戸: 約0.542 W/(m2・K)
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
        #print("Theta_uf_d_t[0] 床下…9218 tokens truncated…         # θuf の本計算
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
                    U_s_override = 0.0  # Excel benchmark omits the direct floor term here.
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

    return E_UT_d_t, \
            Theta_hs_out_d_t, Theta_hs_in_d_t, \
            X_hs_out_d_t, X_hs_in_d_t, V_hs_supply_d_t, V_hs_vent_d_t

