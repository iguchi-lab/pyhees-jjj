import numpy as np

from pyhees.section3_1 import get_A_HCZ_R_i
import pyhees.section3_1_e as algo
import pyhees.section4_2 as dc
# JJJ
from jjjexperiment.common import *
from jjjexperiment.inputs.options import *

from jjjexperiment.logger import LimitedLoggerAdapter as _logger, log_res

def get_r_A_uf_i() -> Array12x1:
    """暖冷房区画iの床面積のうち床下空間に接する床面積の割合 (-)
    """
    r_A_uf_i = np.array([algo.get_r_A_uf_i(i) for i in range(1, 13)])  \
                .reshape(-1, 1)
    assert r_A_uf_i.shape == (12, 1)
    return r_A_uf_i


def get_r_A_NR_uf_1F() -> float:
    '''Return the ratio of first-floor non-room area contacting underfloor air.

    Zones 6, 7, 8, and 9 contact the underfloor space. Zone 9 uses its
    partial contact ratio. The standard-area ratio is invariant when all
    non-room zones are scaled to the dwelling non-room area.
    '''
    A_NR_1F = sum(algo.get_r_A_uf_i(i) * get_A_HCZ_R_i(i) for i in [6, 7, 8, 9])
    # 標準住戸の非居室合計面積
    A_NR_R = sum(get_A_HCZ_R_i(i) for i in range(6, 13))
    return A_NR_1F / A_NR_R


def calc_L_flr1st_area_apportioned_d_t(
        Q_hat_hs_d_t: np.ndarray,
        A_s_ufvnt_HCZ_1F: float,
        A_HCZ_A: float,
        cooling: bool,
    ) -> np.ndarray:
    """式(40)用の1階負荷を空調対象室の床面積比で按分する。

    暖房時は ``L_H,1F = Q_hat_hs,H * A_s,uf,HCZ,1F / A_HCZ,A``、
    冷房時は顕熱出力を用い、床下温度の熱収支に合わせて負値で返す。

    Args:
        Q_hat_hs_d_t: 式(40)の補正前の熱源機出力 [MJ/h]
        A_s_ufvnt_HCZ_1F: 1階空調対象室の床下接触面積 [m2]
        A_HCZ_A: 空調対象室（主たる居室・その他居室）の床面積合計 [m2]
        cooling: 冷房時はTrue

    Returns:
        面積按分した1階の暖房負荷、または冷房顕熱負荷 [MJ/h]
    """
    assert Q_hat_hs_d_t.shape == (24 * 365,)
    assert np.all(Q_hat_hs_d_t >= 0), "熱源機出力は0以上"
    assert A_HCZ_A > 0, "空調対象室の床面積合計は正の値"
    assert 0 <= A_s_ufvnt_HCZ_1F <= A_HCZ_A, \
        "1階空調対象室面積は空調対象室の床面積合計以下"

    L_flr1st_d_t = Q_hat_hs_d_t * A_s_ufvnt_HCZ_1F / A_HCZ_A
    return -L_flr1st_d_t if cooling else L_flr1st_d_t


def get_A_s_ufac_i(
        A_A: float,
        A_MR: float,
        A_OR: float
    ) -> tuple[Array12x1, float]:
    """
    Returns:
        tuple(Array12x1, float):
            - A_s_ufac_i: 暖冷房区画iの床下空調有効面積[m2]
            - r_A_s_ufac: 面積全体における床下空調部分の床面積の比率[-]
    """
    r_A_ufac = 1.0  # 床下空調利用時の有効率
    A_s_ufac_i \
        = np.array([
            # 床面積 * 床下空調部分割合 * 有効率
            algo.calc_A_s_ufvnt_i(i, r_A_ufac, A_A, A_MR, A_OR)
            for i in range(1, 13)
        ]).reshape(-1, 1)
    assert np.shape(A_s_ufac_i) == (12, 1)

    # 面積全体における 床下空調部分の床面積の割合
    r_A_s_ufac = np.sum(A_s_ufac_i) / A_A

    return A_s_ufac_i, r_A_s_ufac

# 計算方法はいくつかあるので使い分けを確認する
# ex. pyhees.section4_2.calc_Theta_uf
# ex. pyhees.section4_2.calc_Theta_uf_d_t_2023


def calc_Theta_uf(
        q_hs_rtd_H: float,
        q_hs_rtd_C: float,
        L_flr1st: float,
        A_s_ufvnt_HCZ_1F: float,
        U_s_supply: float,
        U_s_load: float,
        Theta_in: float,
        Theta_ex: float,
        V_flr1st: float,
        ) -> float:
    """床下空間の温度 (40-4) 単時点

    Args:
        L_flr1st: 1階空調対象室の1時間当たりの暖房負荷（冷房は負値） [MJ/h]
        A_s_ufvnt_HCZ_1F: 1階空調対象室の床下接触面積 [m2]
        U_s_supply: 床下から室へ熱を供給する無断熱床の熱貫流率 [W/m2K]
        U_s_load: 通常の負荷計算で見込んだ床の熱貫流率 [W/m2K]
        Theta_in: 室温 [℃]
        Theta_ex: 外気温度 [℃]
        V_flr1st: 1階空調対象室への吹出風量 [m3/h]

    Returns:
        床下空間の温度 [℃]
    """
    ro_air = dc.get_ro_air()  # 空気密度 [kg/m3]
    c_p_air = algo.get_c_p_air()  # 空気の比熱 [kJ/kgK]
    H_floor = 0.7  # 床の温度差係数(-) 損失として

    # TODO: sympy の方程式で記述できればコードの意味が理解しやすくなる
    b = ro_air * c_p_air * V_flr1st \
        + U_s_supply * A_s_ufvnt_HCZ_1F * 3.6

    match (q_hs_rtd_H, q_hs_rtd_C):
        case (None, None):
            raise Exception("どちらかのみを前提")

        case (_, None):  # 暖房期
            delta_Theta = max(Theta_in - Theta_ex, 0)
            # 元の負荷計算に含まれる一般床断熱の損失を差し引く。
            a2 = U_s_load * A_s_ufvnt_HCZ_1F * delta_Theta * H_floor * 3.6
            assert L_flr1st >= 0, "暖房期の負荷は正の値"
            Theta_uf = (L_flr1st * 1e+3 - a2 + Theta_in * b) / b
            return Theta_uf

        case (None, _):  # 冷房期
            delta_Theta = max(Theta_ex - Theta_in, 0)
            # 元の負荷計算に含まれる一般床断熱の損失を差し引く。
            a2 = U_s_load * A_s_ufvnt_HCZ_1F * delta_Theta * H_floor * 3.6
            assert L_flr1st <= 0, "冷房期の負荷は負の値"
            Theta_uf = (L_flr1st * 1e+3 + a2 + Theta_in * b) / b
            return Theta_uf

        case (_, _):
            raise Exception("どちらかのみを前提")


# vectorizeできなのいのでhstack-broadcastで対応 (A_s_ufac_iが強制でfloatになるため)
def calc_delta_L_room2uf_i(
        U_s_load: float,
        A_s_ufac_i: Array5x1,
        delta_Theta: float
    ) -> Array5x1:
    """床下空間から居室全体への熱損失 [MJ/h]

    Args:
        U_s_load: 通常の負荷計算で見込んだ床の熱貫流率 [W/m2・K]

    """
    assert A_s_ufac_i.ndim == 2
    assert delta_Theta >= 0, "温度差は正を前提に計算"

    H_floor = 0.7  # 床下空調でなく意図しない熱移動の分なので通常の遮蔽係数(0.7)となる

    # 元の負荷計算に含まれる一般床断熱の損失を除く。
    delta_L_uf2room = U_s_load * A_s_ufac_i * delta_Theta * H_floor * 3.6 / 1000  # [W] -> [MJ/h]

    # NOTE: L_H_d_t_i, L_CS_d_t_i に含まれている通常(非床下空調)の床下ロス部分(室内→床下→屋外)
    # 下記の補正を追加する前にコチラを引くことでイコールフッティングできます
    assert delta_L_uf2room.ndim == 2
    return delta_L_uf2room  # TODO: i=1~5でトリムするか検討


def calc_delta_L_uf2outdoor(
        phi: float,
        L_uf: float,
        delta_Theta: float
    ) -> float:
    """床下空間から外気への熱損失 [MJ/h]
    Args:
        phi: 土間床等の外気に接する床の熱貫流率 [W/m2K]
        L_uf: 土間床等の外気に接する床の周辺部の長さ [m]
        delta_Theta: 床下空間と外気の温度差 [℃]
    """
    return phi * L_uf * delta_Theta * 3.6 / 1000  # [W] -> [MJ/h]


def calc_delta_L_uf2gnd(
        q_hs_rtd_H: float,
        q_hs_rtd_C: float,
        A_s_ufvnt_A: float,
        R_g: float,
        Phi_A_0: float,
        Theta_uf: float,
        sum_Theta_dash_g_surf_A_m: float,
        Theta_g_avg: float
    ) -> float:
    """床下空間から地盤への熱損失 [MJ/h]
    Args:
        A_s_ufvnt_A: 空気を供給する床下空間に接する床の面積 [m2]
        R_g: 地盤またはそれを覆う基礎の表面熱伝達抵抗 [m2・K/W]
        Phi_A_0: 吸熱応答係数の初項
        Theta_uf: 床下空間の温度 [℃]
        sum_Theta_dash_g_surf_A_m: 指数項mの吸熱応答の項別成分の合計 [℃]
        Theta_g_avg: 地盤の不易層温度 [℃]
    """
    # CHECK: θ'g_surf_A_d_t の値に不一致アリ
    delta_Theta = Theta_uf - sum_Theta_dash_g_surf_A_m - Theta_g_avg

    # 暖冷房期 判別
    match(q_hs_rtd_H, q_hs_rtd_C):
        case (None, None):
            raise Exception("q_hs_rtd_H, q_hs_rtd_C はどちらかのみを前提")
        case (_, None):  # 暖房期
            delta_Theta = 1 * delta_Theta
        case (None, _):  # 冷房期
            delta_Theta = -1 * delta_Theta
        case (_, _):
            raise Exception("q_hs_rtd_H, q_hs_rtd_C はどちらかのみを前提")

    return (A_s_ufvnt_A / R_g) / (1 + Phi_A_0 / R_g)  \
        * delta_Theta * 3.6 / 1000  # [W] -> [MJ/h]

