# ============================================================================
# 第四章 暖冷房設備
# 第二節 ダクト式セントラル空調機
# Ver.05（エネルギー消費性能計算プログラム（住宅版）Ver.02.01～）
# ============================================================================

import numpy as np

from functools import lru_cache

import datetime

import pyhees.section3_1 as ld

from pyhees.section3_2_8 import \
    get_r_env

from pyhees.section3_1 import \
    get_A_NR

from pyhees.section3_1_d import \
    get_V_A

from pyhees.section3_1_e import \
    get_U_s, get_ro_air, get_c_p_air, \
    calc_Theta, get_r_A_uf_i, calc_A_s_ufvnt_i

from pyhees.section4_3 import \
    get_C_af_H, \
    get_C_af_C

from pyhees.section4_7_i import \
    get_A_A_R

from pyhees.section11_1 import \
    get_Theta_ex, \
    get_X_ex, \
    load_climate, \
    get_climate_df

from pyhees.section11_2 import \
    calc_I_s_d_t

from pyhees.section11_3 import \
    load_schedule, \
    get_schedule_ac

from pyhees.section11_5 import \
    calc_h_ex, \
    get_J

from pyhees.section11_6 import \
    get_table_7

# JJJ
import jjjexperiment.constants as jjj_consts
from jjjexperiment.common import *
from jjjexperiment.logger import LimitedLoggerAdapter as _logger, log_res
from jjjexperiment.inputs.options import *
from jjjexperiment.inputs.di_container import *

@jjj_cloned
# 未処理負荷と機器の計算に必要な変数を取得
def calc_Q_UT_A(A_A, A_MR, A_OR, r_env, mu_H, mu_C, q_hs_rtd_H, q_hs_rtd_C, V_hs_dsgn_H, V_hs_dsgn_C, Q,
             VAV, general_ventilation, duct_insulation, region, L_H_d_t_i, L_CS_d_t_i, L_CL_d_t_i):
    """

    Args:
      A_A: param A_MR:
      A_OR: param A_env:
      mu_H: param mu_C:
      q_hs_rtd_H: param q_hs_rtd_C:
      V_hs_dsgn_H: param V_hs_dsgn_C:
      Q: param VAV:
      general_ventilation: param duct_insulation:
      region: param L_H_d_t_i:
      L_CS_d_t_i: param L_CL_d_t_i:
      A_MR:
      r_env(float): 床面積の合計に対する外皮の部位の面積の合計の比 (-)
      mu_C:
      q_hs_rtd_C:
      V_hs_dsgn_C:
      VAV:
      duct_insulation:
      L_H_d_t_i:
      L_CL_d_t_i:

    Returns:

    """
    raise NotImplementedError("代わりにJJJ改変版を使用する")

    # 外気条件
    climate = load_climate(region)
    X_ex_d_t = get_X_ex(climate)
    Theta_ex_d_t = get_Theta_ex(climate)
    J_d_t = calc_I_s_d_t(0, 0, get_climate_df(climate))
    h_ex_d_t = calc_h_ex(X_ex_d_t, Theta_ex_d_t)

    A_HCZ_i = np.array([ld.get_A_HCZ_i(i, A_A, A_MR, A_OR) for i in range(1, 6)])
    A_HCZ_R_i = [ld.get_A_HCZ_R_i(i) for i in range(1, 6)]

    A_NR = get_A_NR(A_A, A_MR, A_OR)

    # (67)
    L_wtr = get_L_wtr()

    # (66d)
    n_p_NR_d_t = calc_n_p_NR_d_t(A_NR)

    # (66c)
    n_p_OR_d_t = calc_n_p_OR_d_t(A_OR)

    # (66b)
    n_p_MR_d_t = calc_n_p_MR_d_t(A_MR)

    # (66a)
    n_p_d_t = get_n_p_d_t(n_p_MR_d_t, n_p_OR_d_t, n_p_NR_d_t)

    # 人体発熱
    q_p_H = get_q_p_H()
    q_p_CS = get_q_p_CS()
    q_p_CL = get_q_p_CL()

    # (65d)
    w_gen_NR_d_t = calc_w_gen_NR_d_t(A_NR)

    # (65c)
    w_gen_OR_d_t = calc_w_gen_OR_d_t(A_OR)

    # (65b)
    w_gen_MR_d_t = calc_w_gen_MR_d_t(A_MR)

    # (65a)
    w_gen_d_t = get_w_gen_d_t(w_gen_MR_d_t, w_gen_OR_d_t, w_gen_NR_d_t)

    # (64d)
    q_gen_NR_d_t = calc_q_gen_NR_d_t(A_NR)

    # (64c)
    q_gen_OR_d_t = calc_q_gen_OR_d_t(A_OR)

    # (64b)
    q_gen_MR_d_t = calc_q_gen_MR_d_t(A_MR)

    # (64a)
    q_gen_d_t = get_q_gen_d_t(q_gen_MR_d_t, q_gen_OR_d_t, q_gen_NR_d_t)

    # (63)
    V_vent_l_NR_d_t = get_V_vent_l_NR_d_t()
    V_vent_l_OR_d_t = get_V_vent_l_OR_d_t()
    V_vent_l_MR_d_t = get_V_vent_l_MR_d_t()
    V_vent_l_d_t = get_V_vent_l_d_t(V_vent_l_MR_d_t, V_vent_l_OR_d_t, V_vent_l_NR_d_t)

    # (62)
    V_vent_g_i = get_V_vent_g_i(A_HCZ_i, A_HCZ_R_i)

    # (61)
    U_prt = get_U_prt()

    # (60)
    A_prt_i = get_A_prt_i(A_HCZ_i, r_env, A_MR, A_NR, A_OR)

    # (59)
    Theta_SAT_d_t = get_Theta_SAT_d_t(Theta_ex_d_t, J_d_t)

    # (58)
    l_duct_ex_i = get_l_duct_ex_i(A_A)

    # (57)
    l_duct_in_i = get_l_duct_in_i(A_A)

    # (56)
    l_duct_i = get_l_duct__i(l_duct_in_i, l_duct_ex_i)

    # (51)
    X_star_HBR_d_t = get_X_star_HBR_d_t(X_ex_d_t, region)

    # (50)
    Theta_star_HBR_d_t = get_Theta_star_HBR_d_t(Theta_ex_d_t, region)

    # (55)
    Theta_attic_d_t = get_Theta_attic_d_t(Theta_SAT_d_t, Theta_star_HBR_d_t)

    # (54)
    Theta_sur_d_t_i = get_Theta_sur_d_t_i(Theta_star_HBR_d_t, Theta_attic_d_t, l_duct_in_i, l_duct_ex_i, duct_insulation)

    # (40)
    Q_hat_hs_d_t = calc_Q_hat_hs_d_t(Q, A_A, V_vent_l_d_t, V_vent_g_i, mu_H, mu_C, J_d_t, q_gen_d_t, n_p_d_t, q_p_H,
                                     q_p_CS, q_p_CL, X_ex_d_t, w_gen_d_t, Theta_ex_d_t, L_wtr, region)

    # (39)
    V_hs_min = get_V_hs_min(V_vent_g_i)

    # (38)
    Q_hs_rtd_C = get_Q_hs_rtd_C(q_hs_rtd_C)

    # (37)
    Q_hs_rtd_H = get_Q_hs_rtd_H(q_hs_rtd_H)

    # (36)
    V_dash_hs_supply_d_t = get_V_dash_hs_supply_d_t(V_hs_min, V_hs_dsgn_H, V_hs_dsgn_C, Q_hs_rtd_H, Q_hs_rtd_C, Q_hat_hs_d_t, region)

    # (45)
    r_supply_des_i = get_r_supply_des_i(A_HCZ_i)

    # (44)
    V_dash_supply_d_t_i = get_V_dash_supply_d_t_i(r_supply_des_i, V_dash_hs_supply_d_t, V_vent_g_i)

    # (53)
    X_star_NR_d_t = get_X_star_NR_d_t(X_star_HBR_d_t, L_CL_d_t_i, L_wtr, V_vent_l_NR_d_t, V_dash_supply_d_t_i, region)

    # (52)
    Theta_star_NR_d_t = get_Theta_star_NR_d_t(Theta_star_HBR_d_t, Q, A_NR, V_vent_l_NR_d_t, V_dash_supply_d_t_i, U_prt,
                                              A_prt_i, L_H_d_t_i, L_CS_d_t_i, region)

    # (49)
    X_NR_d_t = get_X_NR_d_t(X_star_NR_d_t)

    # (47)
    X_HBR_d_t_i = get_X_HBR_d_t_i(X_star_HBR_d_t)

    # (11)
    Q_star_trs_prt_d_t_i = get_Q_star_trs_prt_d_t_i(U_prt, A_prt_i, Theta_star_HBR_d_t, Theta_star_NR_d_t)

    # (10)
    L_star_CL_d_t_i = get_L_star_CL_d_t_i(L_CS_d_t_i, L_CL_d_t_i, region)

    # (9)
    L_star_CS_d_t_i = get_L_star_CS_d_t_i(L_CS_d_t_i, Q_star_trs_prt_d_t_i, region)

    # (8)
    L_star_H_d_t_i = get_L_star_H_d_t_i(L_H_d_t_i, Q_star_trs_prt_d_t_i, region)

    # (33)
    L_star_CL_d_t = get_L_star_CL_d_t(L_star_CL_d_t_i)

    # (32)
    L_star_CS_d_t = get_L_star_CS_d_t(L_star_CS_d_t_i)

    # (31)
    L_star_CL_max_d_t = get_L_star_CL_max_d_t(L_star_CS_d_t)

    # (30)
    L_star_dash_CL_d_t = get_L_star_dash_CL_d_t(L_star_CL_max_d_t, L_star_CL_d_t)

    # (29)
    L_star_dash_C_d_t = get_L_star_dash_C_d_t(L_star_CS_d_t, L_star_dash_CL_d_t)

    # (28)
    SHF_dash_d_t = get_SHF_dash_d_t(L_star_CS_d_t, L_star_dash_C_d_t)

    # (27)
    Q_hs_max_C_d_t = get_Q_hs_max_C_d_t(q_hs_rtd_C)

    # (26)
    Q_hs_max_CL_d_t = get_Q_hs_max_CL_d_t(Q_hs_max_C_d_t, SHF_dash_d_t, L_star_dash_CL_d_t)

    # (25)
    Q_hs_max_CS_d_t = get_Q_hs_max_CS_d_t(Q_hs_max_C_d_t, SHF_dash_d_t)

    # (24)
    C_df_H_d_t = get_C_df_H_d_t(Theta_ex_d_t, h_ex_d_t)

    # (23)
    Q_hs_max_H_d_t = get_Q_hs_max_H_d_t(q_hs_rtd_H, C_df_H_d_t)

    # (20)
    X_star_hs_in_d_t = get_X_star_hs_in_d_t(X_star_NR_d_t)

    # (19)
    Theta_star_hs_in_d_t = get_Theta_star_hs_in_d_t(Theta_star_NR_d_t)

    # (18)
    X_hs_out_min_C_d_t = get_X_hs_out_min_C_d_t(X_star_hs_in_d_t, Q_hs_max_CL_d_t, V_dash_supply_d_t_i)

    # (22)
    X_req_d_t_i = get_X_req_d_t_i(X_star_HBR_d_t, L_star_CL_d_t_i, V_dash_supply_d_t_i, region)

    # (21)
    Theta_req_d_t_i = get_Theta_req_d_t_i(Theta_sur_d_t_i, Theta_star_HBR_d_t, V_dash_supply_d_t_i,
                        L_star_H_d_t_i, L_star_CS_d_t_i, l_duct_i, region)

    # (15)
    X_hs_out_d_t = get_X_hs_out_d_t(X_NR_d_t, X_req_d_t_i, V_dash_supply_d_t_i, X_hs_out_min_C_d_t, L_star_CL_d_t_i, region)

    # 式(14)(46)(48)の条件に合わせてTheta_NR_d_tを初期化
    Theta_NR_d_t = np.zeros(24 * 365)

    # (17)
    Theta_hs_out_min_C_d_t = get_Theta_hs_out_min_C_d_t(Theta_star_hs_in_d_t, Q_hs_max_CS_d_t, V_dash_supply_d_t_i)

    # (16)
    Theta_hs_out_max_H_d_t = get_Theta_hs_out_max_H_d_t(Theta_star_hs_in_d_t, Q_hs_max_H_d_t, V_dash_supply_d_t_i)

    # L_star_H_d_t_i，L_star_CS_d_t_iの暖冷房区画1～5を合算し0以上だった場合の順序で計算
    # (14)
    Theta_hs_out_d_t = get_Theta_hs_out_d_t(VAV, Theta_req_d_t_i, V_dash_supply_d_t_i,
                                            L_star_H_d_t_i, L_star_CS_d_t_i, region, Theta_NR_d_t,
                                            Theta_hs_out_max_H_d_t, Theta_hs_out_min_C_d_t)

    # (43)
    V_supply_d_t_i = get_V_supply_d_t_i(L_star_H_d_t_i, L_star_CS_d_t_i, Theta_sur_d_t_i, l_duct_i, Theta_star_HBR_d_t,
                                                    V_vent_g_i, V_dash_supply_d_t_i, VAV, region, Theta_hs_out_d_t)

    # (41)
    Theta_supply_d_t_i = get_Thata_supply_d_t_i(Theta_sur_d_t_i, Theta_hs_out_d_t, Theta_star_HBR_d_t, l_duct_i,
                                                   V_supply_d_t_i, L_star_H_d_t_i, L_star_CS_d_t_i, region)

    # (46)
    Theta_HBR_d_t_i = get_Theta_HBR_d_t_i(Theta_star_HBR_d_t, V_supply_d_t_i, Theta_supply_d_t_i, U_prt, A_prt_i, Q,
                                             A_HCZ_i, L_star_H_d_t_i, L_star_CS_d_t_i, region)

    # (48)
    Theta_NR_d_t = get_Theta_NR_d_t(Theta_star_NR_d_t, Theta_star_HBR_d_t, Theta_HBR_d_t_i, A_NR, V_vent_l_NR_d_t,
                                        V_dash_supply_d_t_i, V_supply_d_t_i, U_prt, A_prt_i, Q)

     # L_star_H_d_t_i，L_star_CS_d_t_iの暖冷房区画1～5を合算し0以下だった場合の為に再計算
     # (14)
    Theta_hs_out_d_t = get_Theta_hs_out_d_t(VAV, Theta_req_d_t_i, V_dash_supply_d_t_i,
                                            L_star_H_d_t_i, L_star_CS_d_t_i, region, Theta_NR_d_t,
                                            Theta_hs_out_max_H_d_t, Theta_hs_out_min_C_d_t)

    # (42)
    X_supply_d_t_i = get_X_supply_d_t_i(X_star_HBR_d_t, X_hs_out_d_t, L_star_CL_d_t_i, region)

    # (35)
    V_hs_vent_d_t = get_V_hs_vent_d_t(V_vent_g_i, general_ventilation)

    # (34)
    V_hs_supply_d_t = get_V_hs_supply_d_t(V_supply_d_t_i)

    # (13)
    X_hs_in_d_t = get_X_hs_in_d_t(X_NR_d_t)

    # (12)
    Theta_hs_in_d_t = get_Theta_hs_in_d_t(Theta_NR_d_t)

    # (7)
    L_dash_CL_d_t_i = get_L_dash_CL_d_t_i(V_supply_d_t_i, X_HBR_d_t_i, X_supply_d_t_i, region)

    # (6)
    L_dash_CS_d_t_i = get_L_dash_CS_d_t_i(V_supply_d_t_i, Theta_supply_d_t_i, Theta_HBR_d_t_i, region)

    # (5)
    L_dash_H_d_t_i = get_L_dash_H_d_t_i(V_supply_d_t_i, Theta_supply_d_t_i, Theta_HBR_d_t_i, region)

    # (4)
    Q_UT_CL_d_t_i = get_Q_UT_CL_d_t_i(L_star_CL_d_t_i, L_dash_CL_d_t_i)

    # (3)
    Q_UT_CS_d_t_i = get_Q_UT_CS_d_t_i(L_star_CS_d_t_i, L_dash_CS_d_t_i)

    # (2)
    Q_UT_H_d_t_i = get_Q_UT_H_d_t_i(L_star_H_d_t_i, L_dash_H_d_t_i)

    # (1)
    E_C_UT_d_t = get_E_C_UT_d_t(Q_UT_CL_d_t_i, Q_UT_CS_d_t_i, region)

    return E_C_UT_d_t, Q_UT_H_d_t_i, Q_UT_CS_d_t_i, Q_UT_CL_d_t_i, Theta_hs_out_d_t, Theta_hs_in_d_t, \
           X_hs_out_d_t, X_hs_in_d_t, V_hs_supply_d_t, V_hs_vent_d_t, C_df_H_d_t


# ============================================================================
# 5 暖房エネルギー消費量
# ============================================================================

# ============================================================================
# 5.1 消費電力量
# ============================================================================

# 4_2_aで実装

# ============================================================================
# 5.2 ガス消費量
# ============================================================================

# 日付dの時刻tにおける1時間当たりのガス消費量 (MJ/h)
def get_E_G_H_d_t():
    """ガス消費量
    ガス消費量は0とする

    Args:

    Returns:
      ndarray: ガス消費量

    """
    # ガス消費量は0とする
    return np.zeros(24 * 365)

# ============================================================================
# 5.3 灯油消費量
# ============================================================================

# 日付dの時刻tにおける1時間当たりの灯油消費量 (MJ/h)
def get_E_K_H_d_t():
    """灯油消費量
    灯油消費量は0とする

    Args:

    Returns:
      ndarray: 灯油消費量

    """
    # 灯油消費量は0とする
    return np.zeros(24 * 365)

# ============================================================================
# 5.4 その他の燃料による一次エネルギ―消費量
# ============================================================================

# 日付dの時刻tにおける1時間当たりのその他の燃料による一次エネルギー消費量 (MJ/h)
def get_E_M_H_d_t():
    """その他の燃料による一次エネルギー消費量

    Args:

    Returns:
      ndarray: その他の燃料による一次エネルギー消費量

    """
    # その他の燃料による一次エネルギー消費量は0とする
    return np.zeros(24 * 365)


# ============================================================================
# 6 冷房エネルギー消費量
# ============================================================================

# ============================================================================
# 6.1 消費電力量
# ============================================================================

# 4_2_aで実装

# ============================================================================
# 6.2 ガス消費量
# ============================================================================

# 日付dの時刻tにおける1時間当たりのガス消費量 (MJ/h)
def get_E_G_C_d_t():
    """ """
    # ガス消費量は0とする
    return np.zeros(24 * 365)

# ============================================================================
# 6.3 灯油消費量
# ============================================================================

# 日付dの時刻tにおける1時間当たりの灯油消費量 (MJ/h)
def get_E_K_C_d_t():
    """ """
    # 灯油消費量は0とする
    return np.zeros(24 * 365)

# ============================================================================
# 6.4 その他の燃料による一次エネルギ―消費量
# ============================================================================

# 日付dの時刻tにおける1時間当たりのその他の燃料による一次エネルギー消費量 (MJ/h)
def get_E_M_C_d_t():
    """ """
    # その他の燃料による一次エネルギー消費量は0とする
    return np.zeros(24 * 365)


# ============================================================================
# 7 冷房設備の未処理冷房負荷の設計一次エネルギー消費量相当値
# ============================================================================

def get_E_C_UT_d_t(Q_UT_CL_d_t_i, Q_UT_CS_d_t_i, region):
    """(1)

    Args:
      Q_UT_CL_d_t_i: 日付dの時刻tにおける1時間当たりの暖冷房区画iに設置された冷房機器の未処理冷房潜熱負荷（MJ/h）
      Q_UT_CS_d_t_i: 日付dの時刻tにおける1時間当たりの暖冷房区画iに設置された冷房機器の未処理冷房顕熱負荷（MJ/h）
      region: 地域区分

    Returns:
      日付dの時刻tにおける1時間当たりの冷房設備の未処理冷房負荷の設計一次エネルギー消費量相当値（MJ/h）

    """
    # 暖房設備の未処理冷房負荷を未処理暖房負荷の設計一次エネルギー消費量相当値に換算する係数α_(UT,H)（-）を取得
    from pyhees.section4_1 import \
       get_alpha_UT_H_A

    region = 7 if region == 8 else region

    alpha_UT_H_A = get_alpha_UT_H_A(region)

    # 冷房設備の未処理冷房負荷を未処理冷房負荷の設計一次エネルギー消費量相当値に換算する係数（-）
    alpha_UT_C = alpha_UT_H_A

    return np.sum(alpha_UT_C * (Q_UT_CL_d_t_i + Q_UT_CS_d_t_i), axis=0)


# ============================================================================
# 8 未処理負荷
# ============================================================================

# メモ： i=1-5のみ i>=6 の場合はどこで計算するのか要確認
@log_res(['Q_UT_H_d_t_i'])
def get_Q_UT_H_d_t_i(L_star_H_d_t_i, L_dash_H_d_t_i):
    """(2)

    Args:
      L_star_H_d_t_i: 日付dの時刻tにおける暖冷房区画iの1時間当たりの熱損失を含む負荷バランス時の暖房負荷（MJ/h）
      L_dash_H_d_t_i: 日付dの時刻tにおける暖冷房区画iの1時間当たりの間仕切りの熱損失を含む実際の暖房負荷（MJ/h）

    Returns:
      日付dの時刻tにおける1時間当たりの暖冷房区画iに設置された暖房設備機器等の未処理暖房負荷（MJ/h）

    """
    return np.clip(L_star_H_d_t_i[:5] - L_dash_H_d_t_i[:5], 0, None)

# メモ： i=1-5のみ i>=6 の場合はどこで計算するのか要確認
@log_res(['Q_UT_CS_d_t_i'])
def get_Q_UT_CS_d_t_i(L_star_CS_d_t_i, L_dash_CS_d_t_i):
    """(3)

    Args:
      L_star_CS_d_t_i: 日付dの時刻tにおける暖冷房区画iの1時間当たりの熱損失を含む負荷バランス時の冷房顕熱負荷（MJ/h）
      L_dash_CS_d_t_i: 日付dの時刻tにおける暖冷房区画iの1時間当たりの間仕切りの熱損失を含む実際の冷房顕熱負荷（MJ/h）

    Returns:
      日付dの時刻tにおける1時間当たりの暖冷房区画iに設置された冷房機器の未処理冷房顕熱負荷（MJ/h）

    """
    return np.clip(L_star_CS_d_t_i[:5] - L_dash_CS_d_t_i[:5], 0, None)

# メモ： i=1-5のみ i>=6 の場合はどこで計算するのか要確認
@log_res(['Q_UT_CL_d_t_i'])
def get_Q_UT_CL_d_t_i(L_star_CL_d_t_i, L_dash_CL_d_t_i):
    """(4)

    Args:
      L_star_CL_d_t_i: 日付dの時刻tにおける暖冷房区画iの1時間当たりの熱損失を含む負荷バランス時の冷房潜熱負荷（MJ/h）
      L_dash_CS_d_t_i: 日付dの時刻tにおける暖冷房区画iの1時間当たりの間仕切りの熱損失を含む実際の冷房潜熱負荷（MJ/h）
      L_dash_CL_d_t_i: returns: 日付dの時刻tにおける1時間当たりの暖冷房区画iに設置された冷房機器の未処理冷房潜熱負荷（MJ/h）

    Returns:
      日付dの時刻tにおける1時間当たりの暖冷房区画iに設置された冷房機器の未処理冷房潜熱負荷（MJ/h）

    """
    return np.clip(L_star_CL_d_t_i[:5] - L_dash_CL_d_t_i[:5], 0, None)

# メモ： i=1-5のみ i>=6 の場合はどこで計算するのか要確認
@log_res(['L_dash_H_d_t_i'])
def get_L_dash_H_d_t_i(V_supply_d_t_i, Theta_supply_d_t_i, Theta_HBR_d_t_i, region):
    """(5-1)(5-2)(5-3)

    Args:
      V_supply_d_t_i: 日付dの時刻tにおける暖冷房区画iの吹き出し風量（m3/h）
      Theta_supply_d_t_i: 日付dの時刻tにおける暖冷房区画iの吹き出し温度（℃）
      Theta_HBR_d_t_i: 日付dの時刻tにおける暖冷房区画iの実際の居室の室温（℃）
      region: 地域区分

    Returns:
      日付dの時刻tにおける暖冷房区画iの1時間当たりの間仕切りの熱損失を含む実際の暖房負荷（MJ/h）

    """
    c_p_air = get_c_p_air()
    rho_air = get_rho_air()
    H, C, M = get_season_array_d_t(region)

    L_dash_H_d_t_i = np.zeros((5, 24 * 365))

    # 暖房期 (5-1)
    L_dash_H_d_t_i[:, H] = c_p_air * rho_air * V_supply_d_t_i[:, H] * (Theta_supply_d_t_i[:, H] - Theta_HBR_d_t_i[:, H]) * 10 ** -6

    # 冷房期 (5-2)
    L_dash_H_d_t_i[:, C] = 0.0

    # 中間期 (5-3)
    L_dash_H_d_t_i[:, M] = 0.0

    return L_dash_H_d_t_i


@log_res(['L_dash_CS_d_t_i'])
def get_L_dash_CS_d_t_i(V_supply_d_t_i, Theta_supply_d_t_i, Theta_HBR_d_t_i, region):
    """(6-1)(6-2)(6-3)

    Args:
      V_supply_d_t_i: 日付dの時刻tにおける暖冷房区画iの吹き出し風量（m3/h）
      Theta_supp…19089 tokens truncated…る内部発熱
def get_table_4():
    """ """
    return [
        (66.9,66.9,66.9,66.9,66.9,66.9,123.9,383.6,323.2,307.3,134.8,66.9,286.7,271.2,66.9,66.9,236.9,288.6,407.8,383.1,423.1,339.1,312.9,278),
        (66.9,66.9,66.9,66.9,66.9,66.9,66.9,66.9,440.5,443.3,515.1,488.9,422.9,174.4,66.9,66.9,237.8,407.8,383.1,326.8,339.1,339.1,312.9,66.9),
        (18,18,18,18,18,18,18,18,18,398.2,18,18,18,18,18,18,18,18,53,53,115.5,103,258.3,137.3),
        (18,18,18,18,18,18,18,18,35.5,654.3,223,223,53,18,18,18,93,93,55.5,18,270,168.8,270,18),
        (41.5,41.5,41.5,41.5,41.5,41.5,126.1,249.9,158.3,191.3,117.5,41.5,42.5,89,41.5,41.5,105.8,105.8,112.1,118.5,155.7,416.1,314.8,174.9),
        (41.5,41.5,41.5,41.5,41.5,41.5,41.5,281.3,311,269.5,100.4,106.7,98.5,55.8,41.5,41.5,158.4,171.3,82.7,101.4,99.5,255.1,232.1,157.8),
    ]


# 日付dの時刻tにおける内部発湿
def get_w_gen_d_t(w_gen_MR_d_t, w_gen_OR_d_t, w_gen_NR_d_t):
    """(65a)

    Args:
      w_gen_MR_d_t: 日付dの時刻tにおける主たる居室の内部発湿（W）
      w_gen_OR_d_t: 日付dの時刻tにおけるその他の居室の内部発湿（W）
      w_gen_NR_d_t: 日付dの時刻tにおける非居室の内部発湿（W）

    Returns:
      日付dの時刻tにおける内部発湿（W）

    """
    return  w_gen_MR_d_t + w_gen_OR_d_t + w_gen_NR_d_t


def calc_w_gen_MR_d_t(A_MR):
    """(65b)

    Args:
      A_MR: 主たる居室の床面積（m2）

    Returns:

    """
    w_gen_MR_R_d_t = get_w_gen_MR_R_d_t()

    return w_gen_MR_R_d_t * (A_MR / 29.81)


def calc_w_gen_OR_d_t(A_OR):
    """(65c)

    Args:
      A_OR: その他の居室の床面積（m2）

    Returns:

    """
    w_gen_OR_R_d_t = get_w_gen_OR_R_d_t()

    return w_gen_OR_R_d_t * (A_OR / 51.34)


def calc_w_gen_NR_d_t(A_NR):
    """(65d)

    Args:
      A_NR: 非居室の床面積（m2）

    Returns:

    """
    w_gen_NR_R_d_t = get_w_gen_NR_R_d_t()

    return w_gen_NR_R_d_t * (A_NR / 38.93)


# 日付dの時刻tにおける標準住戸の主たる居室の内部発湿（W）
def get_w_gen_MR_R_d_t():
    """:return: 日付dの時刻tにおける標準住戸の主たる居室の内部発湿（W）"""
    schedule = load_schedule()
    schedule_ac = get_schedule_ac(schedule)

    table_5 = get_table_5()


    # 全日平日とみなした24時間365日の標準住戸における内部発湿
    tmp_a = np.tile(table_5[0], 365)

    # 全日休日とみなした24時間365日の標準住戸における内部発湿
    tmp_b = np.tile(table_5[1], 365)

    # 時間単位に展開した生活パターン
    schedule_extend = np.repeat(np.array(schedule_ac), 24)

    w_gen_MR_R_d_t = tmp_a * (schedule_extend == '平日') \
                    + tmp_b * (schedule_extend == '休日')

    return w_gen_MR_R_d_t


# 日付dの時刻tにおける標準住戸のその他の居室の内部発湿（W）
def get_w_gen_OR_R_d_t():
    """:return: 日日付dの時刻tにおける標準住戸のその他の居室の内部発湿（W）"""
    schedule = load_schedule()
    schedule_ac = get_schedule_ac(schedule)

    table_5 = get_table_5()

    # 全日平日とみなした24時間365日の標準住戸における内部発湿
    tmp_a = np.tile(table_5[2], 365)

    # 全日休日とみなした24時間365日の標準住戸における内部発湿
    tmp_b = np.tile(table_5[3], 365)

    # 時間単位に展開した生活パターン
    schedule_extend = np.repeat(np.array(schedule_ac), 24)

    w_gen_OR_R_d_t = tmp_a * (schedule_extend == '平日') \
                      + tmp_b * (schedule_extend == '休日')

    return w_gen_OR_R_d_t


# 日付dの時刻tにおける標準住戸の非居室の内部発湿（W）
def get_w_gen_NR_R_d_t():
    """:return: 日付dの時刻tにおける標準住戸の非居室の内部発湿（W）"""
    schedule = load_schedule()
    schedule_ac = get_schedule_ac(schedule)

    table_5 = get_table_5()

    # 全日平日とみなした24時間365日の標準住戸における内部発湿
    tmp_a = np.tile(table_5[4], 365)

    # 全日休日とみなした24時間365日の標準住戸における内部発湿
    tmp_b = np.tile(table_5[5], 365)

    # 時間単位に展開した生活パターン
    schedule_extend = np.repeat(np.array(schedule_ac), 24)

    w_gen_NR_R_d_t = tmp_a * (schedule_extend == '平日') \
                      + tmp_b * (schedule_extend == '休日')

    return w_gen_NR_R_d_t


# 標準住戸における内部発熱
def get_table_5():
    """ """
    return [
        (0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ]


# ============================================================================
# 13.2.6 人体発熱および在室人数
# ============================================================================

# 暖房期における人体からの1人当たりの顕熱発熱量（W/人）
def get_q_p_H():
    """ """
    return 79.0


# 冷房期における人体からの1人当たりの顕熱発熱量（W/人）
def get_q_p_CS():
    """ """
    return 51.0


# 冷房期における人体からの1人当たりの潜熱発熱量（W/人）
def get_q_p_CL():
    """ """
    return 40.0


# 日付dの時刻tにおける在室人数（人）
def get_n_p_d_t(n_p_MR_d_t, n_p_OR_d_t, n_p_NR_d_t):
    """(66a)

    Args:
      q_gen_MR_d_t: 日付dの時刻tにおける主たる居室の在室人数（人）
      q_gen_OR_d_t: 日付dの時刻tにおけるその他の居室の在室人数（人）
      q_gen_NR_d_t: 日付dの時刻tにおける非居室の在室人数（人）
      n_p_MR_d_t: param n_p_OR_d_t:
      n_p_NR_d_t: returns: 日付dの時刻tにおける在室人数（人）
      n_p_OR_d_t:

    Returns:
      日付dの時刻tにおける在室人数（人）

    """
    return n_p_MR_d_t + n_p_OR_d_t + n_p_NR_d_t


def calc_n_p_MR_d_t(A_MR):
    """(66b)

    Args:
      A_MR: 主たる居室の床面積（m2）

    Returns:

    """
    n_p_MR_R_d_t = get_n_p_MR_R_d_t()

    return  n_p_MR_R_d_t * (A_MR / 29.81)


def calc_n_p_OR_d_t(A_OR):
    """(66c)

    Args:
      A_OR: その他の居室の床面積（m2）

    Returns:

    """
    n_p_OR_R_d_t = get_n_p_OR_R_d_t()

    return  n_p_OR_R_d_t * (A_OR / 51.34)


def calc_n_p_NR_d_t(A_NR):
    """(66d)

    Args:
      A_NR: 非居室の床面積（m2）

    Returns:

    """
    n_p_NR_R_d_t = get_n_p_NR_R_d_t()

    return  n_p_NR_R_d_t * (A_NR / 38.93)


# 日付dの時刻tにおける標準住戸の主たる居室の在室人数（W）
def get_n_p_MR_R_d_t():
    """:return: 日付dの時刻tにおける標準住戸の主たる居室の在室人数（W）"""
    schedule = load_schedule()
    schedule_ac = get_schedule_ac(schedule)

    table_6 = get_table_6()

    # 全日平日とみなした24時間365日の標準住戸における在室人数
    tmp_a = np.tile(table_6[0], 365)

    # 全日休日とみなした24時間365日の標準住戸における在室人数
    tmp_b = np.tile(table_6[1], 365)

    # 時間単位に展開した生活パターン
    schedule_extend = np.repeat(np.array(schedule_ac), 24)

    n_p_MR_R_d_t = tmp_a * (schedule_extend == '平日') \
                    + tmp_b * (schedule_extend == '休日')

    return n_p_MR_R_d_t


# 日付dの時刻tにおける標準住戸のその他の居室の在室人数（W）
def get_n_p_OR_R_d_t():
    """:return: 日日付dの時刻tにおける標準住戸のその他の居室の在室人数（W）"""
    schedule = load_schedule()
    schedule_ac = get_schedule_ac(schedule)

    table_6 = get_table_6()

    # 全日平日とみなした24時間365日の標準住戸における在室人数
    tmp_a = np.tile(table_6[2], 365)

    # 全日休日とみなした24時間365日の標準住戸における在室人数
    tmp_b = np.tile(table_6[3], 365)

    # 時間単位に展開した生活パターン
    schedule_extend = np.repeat(np.array(schedule_ac), 24)

    n_p_OR_R_d_t = tmp_a * (schedule_extend == '平日') \
                      + tmp_b * (schedule_extend == '休日')

    return n_p_OR_R_d_t


# 日付dの時刻tにおける標準住戸の非居室の在室人数（W）
def get_n_p_NR_R_d_t():
    """:return: 日付dの時刻tにおける標準住戸の非居室の在室人数（W）"""
    schedule = load_schedule()
    schedule_ac = get_schedule_ac(schedule)

    table_6 = get_table_6()

    # 全日平日とみなした24時間365日の標準住戸における在室人数
    tmp_a = np.tile(table_6[4], 365)

    # 全日休日とみなした24時間365日の標準住戸における在室人数
    tmp_b = np.tile(table_6[5], 365)

    # 時間単位に展開した生活パターン
    schedule_extend = np.repeat(np.array(schedule_ac), 24)

    n_p_NR_R_d_t = tmp_a * (schedule_extend == '平日') \
                      + tmp_b * (schedule_extend == '休日')

    return n_p_NR_R_d_t


# 標準住戸における内部発熱
def get_table_6():
    """ """
    return [
        (0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 1, 1, 0, 0, 1, 2, 2, 3, 3, 2, 1, 1),
        (0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 2, 2, 1, 0, 0, 2, 3, 3, 4, 2, 2, 1, 0),
        (4, 4, 4, 4, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 2, 3),
        (4, 4, 4, 4, 4, 4, 4, 3, 1, 2, 2, 2, 1, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 3),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ]


# ============================================================================
# 13.3 使い方
# ============================================================================

# ============================================================================
# 13.3.1 暖冷房期間
# ============================================================================

# 暖冷房期間
def get_season_array(region):
    """

    Args:
      region:

    Returns:

    """
    table_7 = get_table_7()[region-1]

    H = get_period_array(table_7[0], table_7[1])
    C = get_period_array(table_7[2], table_7[3])
    M = np.logical_and(np.logical_not(H), np.logical_not(C))

    return H, C, M


# 暖冷房期間を24*365の配列で返す
def get_season_array_d_t(region):
    """

    Args:
      region:

    Returns:

    """
    H, C, M = get_season_array(region)

    H = np.repeat(H, 24)
    C = np.repeat(C, 24)
    M = np.repeat(M, 24)

    return H, C, M


def get_period_array(p1, p2):
    """指定月日期間のみTrueのndarrayを作成する

    指定月日期間のみTrueのndarrayを作成する。
    開始月日が終了月日が逆転する場合は、年をまたいだとみなす。

    Args:
      p1: 開始月日をtuple指定 例) 1月2日 であれば (1,2)
      p2: 終了月日をtuple指定 例) 5月4日 であれば (5,4)

    Returns:
      p1からp2の間はTrueである365の長さのndarray

    """

    if p1 is None or p2 is None:
        return np.zeros(365, dtype=bool)

    d_base = datetime.date(2018, 1, 1)   #年初
    d1 = (datetime.date(2018, p1[0], p1[1]) - d_base).days   #年初からの日数1
    d2 = (datetime.date(2018, p2[0], p2[1]) - d_base).days   #年初からの日数2

    if d1 < d2:
        # d1からd2の間はTrue
        # 例) 7月10日～8月31日
        arr = np.zeros(365, dtype=bool)
        arr[d1:d2+1] = True
    else:
        # d1からd2の間はFalse
        # 例) 9月24日～6月7日
        arr = np.ones(365, dtype=bool)
        arr[d2+1:d1] = False

    return arr


# ============================================================================
# 13.3.2 設定温度・設定絶対湿度
# ============================================================================

# 暖房時の設定温度 (℃)
def get_Theta_set_H():
    """ """
    return 20.0


# 冷房時の設定温度 (℃)
def get_Theta_set_C():
    """ """
    return 27.0


# 冷房時の設定絶対湿度(空気温度27℃、60%の時の絶対湿度とする) (kg/kg(DA))
def get_X_set_C():
    """ """
    return 0.013425743


# ============================================================================
# 13.4 暖房負荷・冷房負荷
# ============================================================================


# ============================================================================
# 13.5 空気および水の物性値
# ============================================================================

# WARN: section3_1_e.get_c_p_air() と単位が異なるので注意
# 空気の比熱 (J/Kg・K)
def get_c_p_air():
    """ """
    return 1006.0
    # NOTE: 先生提供資料では 1005 を使用


# 空気の密度 (kg/m3)
def get_rho_air():
    """ """
    return 1.2


# 水の蒸発潜熱 (kJ/kg) (67)
def get_L_wtr():
    """ """
    Theta = get_Theta()
    return 2500.8 - 2.3668 * Theta


# 冷房時を仮定した温度 (℃)
def get_Theta():
    """ """
    return 27


# ============================================================================
# デバッグ用コード
# ============================================================================
if __name__ == '__main__':
    from pyhees.section2_2 import get_E_H_d_t, get_E_C_d_t
    from pyhees.section3_1 import get_Q
    from pyhees.section3_2 import calc_r_env, get_Q_dash, get_mu_H, get_mu_C
    from pyhees.section4_1 import calc_heating_load, calc_cooling_load, calc_E_UT_H_d_t
    from pyhees.section4_2_a import get_q_hs_H_d_t, get_q_hs_C_d_t
    import pyhees.section4_2_b as dc_spec
    import pandas as pd
    import numpy as np

    region = 6
    A_A = 120.08
    A_MR = 29.81
    A_OR = 51.34
    A_env = 307.51

    mode_H = '住戸全体を連続的に暖房する方式'
    mode_C = '住戸全体を連続的に冷房する方式'

    H_A = {
        'type': 'ダクト式セントラル空調機',
        'duct_insulation': '全てもしくは一部が断熱区画外である',
        'VAV': False,
        'general_ventilation': True,
        'EquipmentSpec': '入力しない'
    }
    C_A = {
        'type': 'ダクト式セントラル空調機',
        'duct_insulation': '全てもしくは一部が断熱区画外である',
        'VAV': False,
        'general_ventilation': True,
        'EquipmentSpec': '入力しない'
    }

    sol_region = None,
    NV_MR = 0
    NV_OR = 0
    TS = False
    r_A_ufvnt = None
    HEX = None
    underfloor_insulation = None
    spec_MR = None
    spec_OR = None
    mode_MR = None
    mode_OR = None
    C_MR = None
    C_OR = None
    SHC = None
    HW = None
    CG = None
    spec_HS = None
    heating_flag_d = None
    duct_insulation = '全てもしくは一部が断熱区画外である'
    VAV = False
    general_ventilation = True
    EquipmentSpec = '入力しない'

    # 床面積の合計に対する外皮の部位の面積の合計の比
    r_env = calc_r_env(
        method='当該住戸の外皮の部位の面積等を用いて外皮性能を評価する方法',
        A_env=A_env,
        A_A=A_A
    )

    # セントラル暖房機器の仕様
    q_hs_rtd_H = dc_spec.get_q_hs_rtd_H(region, A_A)
    q_hs_mid_H = dc_spec.get_q_hs_mid_H(q_hs_rtd_H)
    q_hs_min_H = dc_spec.get_q_hs_min_H(q_hs_rtd_H)
    P_hs_rtd_H = dc_spec.get_P_hs_rtd_H(q_hs_rtd_H)
    V_fan_rtd_H = dc_spec.get_V_fan_rtd_H(q_hs_rtd_H)
    V_fan_mid_H = dc_spec.get_V_fan_mid_H(q_hs_mid_H)
    P_fan_rtd_H = dc_spec.get_P_fan_rtd_H(V_fan_rtd_H)
    P_fan_mid_H = dc_spec.get_P_fan_mid_H(V_fan_mid_H)
    P_hs_mid_H = np.nan
    V_hs_dsgn_H = dc_spec.get_V_fan_dsgn_H(V_fan_rtd_H)

    # セントラル冷房機器の仕様
    q_hs_rtd_C = dc_spec.get_q_hs_rtd_C(region, A_A)
    q_hs_mid_C = dc_spec.get_q_hs_mid_C(q_hs_rtd_C)
    q_hs_min_C = dc_spec.get_q_hs_min_C(q_hs_rtd_C)
    P_hs_rtd_C = dc_spec.get_P_hs_rtd_C(q_hs_rtd_C)
    V_fan_rtd_C = dc_spec.get_V_fan_rtd_C(q_hs_rtd_C)
    V_fan_mid_C = dc_spec.get_V_fan_mid_C(q_hs_mid_C)
    P_fan_rtd_C = dc_spec.get_P_fan_rtd_C(V_fan_rtd_C)
    P_fan_mid_C = dc_spec.get_P_fan_mid_C(V_fan_mid_C)
    P_hs_mid_C = np.nan
    V_hs_dsgn_C = dc_spec.get_V_fan_dsgn_C(V_fan_rtd_C)

    # 外皮平均熱貫流率(UA値)
    U_A = np.array([0.87, 0.60, 0.40])
    # 日射取得係数
    eta_A_H = np.array([4.3, 3.0, 2.0])
    eta_A_C = np.array([2.8, 2.0, 1.0])

    for i in range(3):
        mu_H = get_mu_H(eta_A_H[i], r_env)
        mu_C = get_mu_C(eta_A_C[i], r_env)

        # 熱損失係数（換気による熱損失を含まない）
        Q_dash = get_Q_dash(U_A[i], r_env)
        # 熱損失係数
        Q = get_Q(Q_dash)

        # 暖房負荷の取得
        L_H_d_t_i, _ = calc_heating_load(region, sol_region, A_A, A_MR, A_OR, Q, mu_H, mu_C, NV_MR, NV_OR, TS, r_A_ufvnt,
                                         HEX, underfloor_insulation, mode_H, mode_C, spec_MR, spec_OR, mode_MR, mode_OR, SHC)

        # 冷房負荷の取得
        L_CS_d_t_i, L_CL_d_t_i = calc_cooling_load(region, A_A, A_MR, A_OR, Q, mu_H, mu_C, NV_MR, NV_OR, r_A_ufvnt,
                                                   underfloor_insulation, mode_C, mode_H, mode_MR, mode_OR, TS, HEX)

        # 未処理冷房負荷の一次エネ相当および熱源機の入口における空気温度
        E_C_UT_d_t, _, _, _, Theta_hs_out_d_t, Theta_hs_in_d_t, \
        X_hs_out_d_t, X_hs_in_d_t, V_hs_supply_d_t, _, C_df_H_d_t = calc_Q_UT_A(A_A, A_MR, A_OR, r_env, mu_H, mu_C,
                                                                               q_hs_rtd_H, q_hs_rtd_C, V_hs_dsgn_H,
                                                                               V_hs_dsgn_C, Q, VAV, general_ventilation,
                                                                               duct_insulation, region, L_H_d_t_i,
                                                                               L_CS_d_t_i, L_CL_d_t_i)

        # 1時間当たりの熱源機の平均暖房能力
        q_hs_H_d_t = get_q_hs_H_d_t(Theta_hs_out_d_t, Theta_hs_in_d_t, V_hs_supply_d_t, C_df_H_d_t, region)

        # 1時間当たりの熱源機の平均冷房能力
        q_hs_CS_d_t, q_hs_CL_d_t = get_q_hs_C_d_t(Theta_hs_out_d_t, Theta_hs_in_d_t, X_hs_out_d_t, X_hs_in_d_t, V_hs_supply_d_t, region)
        q_hs_C_d_t = q_hs_CS_d_t + q_hs_CL_d_t

        # 未処理暖房負荷の一次エネ相当
        E_UT_H_d_t = calc_E_UT_H_d_t(region, A_A, A_MR, A_OR, r_env, mu_H, mu_C, Q, mode_H, H_A, spec_MR, spec_OR, spec_HS,
                                     mode_MR, mode_OR, CG, L_H_d_t_i, L_CS_d_t_i, L_CL_d_t_i)

        # 1時間当たりの暖房設備の設計一次エネルギー消費量
        E_H_d_t = get_E_H_d_t(region, sol_region, A_A, A_MR, A_OR, r_env, mu_H, mu_C, Q, mode_H, H_A, spec_MR, spec_OR,
                              spec_HS, mode_MR, mode_OR, HW, CG, SHC, heating_flag_d, L_H_d_t_i, L_CS_d_t_i, L_CL_d_t_i)

        # 1時間当たりの冷房設備の設計一次エネルギー消費量
        E_C_d_t = get_E_C_d_t(region, A_A, A_MR, A_OR, r_env, mu_H, mu_C, Q, C_A, C_MR, C_OR,
                              L_H_d_t_i, L_CS_d_t_i, L_CL_d_t_i, mode_C)

        df = pd.DataFrame({
            'Theta_hs_in_d_t':Theta_hs_in_d_t,
            'q_hs_H_d_t':q_hs_H_d_t,
            'q_hs_C_d_t':q_hs_C_d_t,
            'E_H_d_t':E_H_d_t,
            'E_C_d_t':E_C_d_t,
            'E_UT_H_d_t':E_UT_H_d_t,
            'E_C_UT_d_t':E_C_UT_d_t
        })

        df.to_csv('新法_Calc_condition_' + str(i + 1) +'.csv')

