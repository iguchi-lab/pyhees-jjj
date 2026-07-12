import numpy as np

import pyhees.section4_2 as dc
# JJJ
from jjjexperiment.inputs.di_container import *
from jjjexperiment.common import *
from jjjexperiment.inputs.options import *

# shape(5,1)の引数があるので hstack & broadcast で利用
@jjj_cloning
def get_Theta_HBR_i(
        Theta_star_HBR: float,
        V_supply_i: Array5x1,
        Theta_supply_i: Array5x1,
        U_prt: float,
        A_prt_i: Array5x1,
        Q: float,
        A_HCZ_i: Array5x1,
        L_star_H_i: Array5x1,
        L_star_CS_i: Array5x1,
        HCM: JJJ_HCM,
        A_s_ufac_i: Array5x1,
        Theta_uf: float,
        U_s_override: float = None,
    ) -> Array5x1:
    """単時点版 (46-1)(46-2)(46-3) の床下空調 補正
    """
    # TODO: 事前条件: 新床下空調ロジック 有効時のみ

    assert V_supply_i.shape == (5, 1), '想定外の行列数'
    assert Theta_supply_i.shape == (5, 1), '想定外の行列数'
    assert A_prt_i.shape == (5, 1), '想定外の行列数'
    assert A_HCZ_i.shape == (5, 1), '想定外の行列数'
    assert L_star_H_i.shape == (5, 1), '想定外の行列数'
    assert L_star_CS_i.shape == (5, 1), '想定外の行列数'
    assert A_s_ufac_i.shape == (5, 1), '想定外の行列数'

    CRV = dc.get_c_p_air() * dc.get_rho_air() * V_supply_i
    assert CRV.shape == (5, 1), '想定外の行列数'

    U_s = dc.get_U_s()  # U_s_vert でないチェック済み

    if U_s_override is not None:
        U_s = U_s_override

    match HCM:
        # 暖房期 (46-1)
        case JJJ_HCM.H:
            Theta_HBR_i = Theta_star_HBR  \
                + (CRV * (Theta_supply_i - Theta_star_HBR)
                    + U_s * A_s_ufac_i * (Theta_uf - Theta_star_HBR) * 3600  ##
                    - L_star_H_i * 1e+6)  \
                    / (CRV
                        + (U_prt * A_prt_i + Q * A_HCZ_i) * 3600
                        + U_s * A_s_ufac_i * 3600)  ##
            return np.clip(Theta_HBR_i, Theta_star_HBR, None)

        # 冷房期 (46-2)
        case JJJ_HCM.C:
            Theta_HBR_i = Theta_star_HBR  \
                - (CRV * (Theta_star_HBR - Theta_supply_i)
                    + U_s * A_s_ufac_i * (Theta_star_HBR - Theta_uf) * 3600  ##
                    - L_star_CS_i * 1e+6)  \
                    / (CRV
                        + (U_prt * A_prt_i + Q * A_HCZ_i) * 3600
                        + U_s * A_s_ufac_i * 3600)  ##
            return np.clip(Theta_HBR_i, None, Theta_star_HBR)

        # 中間期 (46-3)
        case JJJ_HCM.M:
            return np.full((5, 1), Theta_star_HBR)

        case _:
            raise ValueError('HCMの値が不正です')

@jjj_cloning  # section4_2/get_Theta_NR_d_t
def get_Theta_NR(
        Theta_star_NR: float,
        Theta_star_HBR: float,
        Theta_HBR_i: Array5x1,
        A_NR: float,
        V_vent_l_NR: float,
        V_dash_supply_i: Array5x1,
        V_supply_i: Array5x1,
        U_prt: float,
        A_prt_i: Array5x1,
        Q: float,
        Theta_uf: float,
        r_A_NR_1F: float
    ) -> float:
    """単時点版 (48a)(48b) 床下空調 補正アリ
    """
    # TODO: 事前条件: 新床下空調ロジック 有効時のみ

    assert Theta_HBR_i.shape == (5, 1), '想定外の行列数'
    assert V_dash_supply_i.shape == (5, 1), '想定外の行列数'
    assert V_supply_i.shape == (5, 1), '想定外の行列数'
    assert A_prt_i.shape == (5, 1), '想定外の行列数'

    c_p_air = dc.get_c_p_air()
    rho_air = dc.get_rho_air()
    # (48d)
    k_dash_prt_i = c_p_air * rho_air * (V_dash_supply_i / 3600) + U_prt * A_prt_i
    k_dash_prt_A = np.sum(k_dash_prt_i)
    # (48c)
    k_prt_i = c_p_air * rho_air * (V_supply_i / 3600) + U_prt * A_prt_i
    k_prt_A = np.sum(k_prt_i)

    # NOTE: VAVなしのとき V',supply=V,supply -> k',prt,i=k,prt,i

    # (48b) [J/(K・s)]
    # Only the first-floor non-room area in contact with the underfloor space.
    A_s_ufvnt_NR = A_NR * r_A_NR_1F

    k_evp = (Q - 0.35 * 0.5 * 2.4) * A_NR + c_p_air * rho_air * (V_vent_l_NR / 3600)
    # (48a)
    U_s = dc.get_U_s()  # U_s_vert でないチェック済み

    #Theta_NR  \
    #    = ((k_evp + k_prt_A) * Theta_star_NR  \
    #        + U_s * A_NR * Theta_uf  \
    #        - k_dash_prt_A * (Theta_star_HBR - Theta_star_NR)  \
    #        + np.sum(k_prt_i * (Theta_HBR_i - Theta_star_NR)))  \
    #    / (k_evp
    #       + U_s * A_NR
    #       + k_prt_A)

    Theta_NR  \
        = ((k_evp + k_prt_A) * Theta_star_NR  \
            + U_s * A_s_ufvnt_NR * Theta_uf  \
            - k_dash_prt_A * (Theta_star_HBR - Theta_star_NR)  \
            + np.sum(k_prt_i * (Theta_HBR_i - Theta_star_NR)))  \
        / (k_evp
           + U_s * A_s_ufvnt_NR
           + k_prt_A)

    # NOTE: 非床下空調の式は定義しない(オリジナルを使用)
    return Theta_NR
