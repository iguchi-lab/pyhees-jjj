import pyhees.section4_2 as dc
# JJJ
from jjjexperiment.common import *

# NOTE: θ*NR,d,t は、室温の計算時に、床下からの貫流分を考慮(2025/03)
@jjj_cloning  # section4_2/get_Theta_star_NR_d_t
def get_Theta_star_NR(
        Theta_star_HBR: float,
        Q: float,
        A_NR: float,
        V_vent_l_NR: float,
        V_dash_supply_A: float,
        U_prt: float,
        A_prt_A: float,
        L_H_NR_A: float,
        L_CS_NR_A: float,
        Theta_NR: float,
        Theta_uf: float,
        HCM: JJJ_HCM,  # regionの代替
        r_A_NR_1F_excl_bath: float  # 非居室の1F(浴室除く)面積比 [-]
    ) -> float:
    """(52-1)(52-2)(52-3)
    Args:
        Theta_star_HBR: 負荷バランス時の居室の室温 [℃]
        Q: 当該住戸の熱損失係数 [W/(m2・K)]
        A_NR: 非居室の床面積 [m2]
        V_vent_l_NR: 非居室の局所換気量 [m3/h]
        V_dash_supply_A: 暖冷房区画(i=1~5)のVAV調整前の吹き出し風量の合計 [m3/h]
        U_prt: 間仕切りの熱貫流率 [W/(m2・K)]
        A_prt_A: 暖冷房区画(i=1~5)から見た非居室の間仕切りの面積の合計 [m2]
        L_H_NR_A: 暖冷房区画(i=6~12)の1時間当たりの暖房負荷の合計 [MJ/h]
        L_CS_NR_A: 暖冷房区画(i=6~12)の1時間当たりの冷房顕熱負荷の合計 [MJ/h]
        Theta_NR: 非居室の室温 [℃]
        Theta_uf: 床下温度 [℃]
        HCM: 暖冷房期間
    Returns:
        Theta_star_NR: 単時点版 負荷バランス時の非居室の室温 [℃]
    """
    # NOTE: 新床下空調ロジックのみで使用可能

    # vectorize可能
    c_p_air = dc.get_c_p_air()
    rho_air = dc.get_rho_air()
    U_s = dc.get_U_s()  # U_s_vert でないチェック済み

    A_s_ufvnt_NR = A_NR
    # Excel benchmark applies the full non-room area at this stage.

    k1 = (Q - 0.35 * 0.5 * 2.4) * A_NR \
        + c_p_air * rho_air * V_vent_l_NR / 3600  \
        + c_p_air * rho_air * V_dash_supply_A / 3600  \
        + U_prt * A_prt_A

    k2 = U_s * A_s_ufvnt_NR

    match HCM:
        case JJJ_HCM.H:
            # (52-1)
            Theta_star_NR  \
                = (k1 * Theta_star_HBR  \
                   + k2 * Theta_uf  \
                   - L_H_NR_A * 1e+6 / 3600)  \
                / (k1 + k2)
            return Theta_star_NR

        case JJJ_HCM.C:
            # (52-2)
            Theta_star_NR  \
                = (k1 * Theta_star_HBR  \
                    + k2 * Theta_uf  \
                    #冷房計算の符号を修正　250501 IGUCHI
                    + L_CS_NR_A * 1e+6 / 3600)  \
                / (k1 + k2)
            return Theta_star_NR

        case JJJ_HCM.M:
            # (52-3)
            Theta_star_NR = Theta_star_HBR
            return Theta_star_NR

        case _: raise ValueError('HCMの値が不正です')
