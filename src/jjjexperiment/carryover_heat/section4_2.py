import numpy as np

import pyhees.section4_2 as dc
# JJJ
from jjjexperiment.common import *
import jjjexperiment.carryover_heat as jjj_carryover_heat
from jjjexperiment.logger import log_res

def calc_carryover(
        H: np.bool,
        C: np.bool,
        A_HCZ_i: Array5,
        Theta_HBR_i: Array5x1,
        Theta_star_HBR: float,
    ) -> Array5x1:
    """過剰熱量 [MJ] (一時点)

    Args:
        H: 暖房期
        C: 冷房期
        A_HCZ_i: 暖冷房区画iの床面積 [m2]
        Theta_HBR_i: (時点)暖冷房区画iの実際の居室の室温 [℃]
        Theta_star_HBR: (時点)負荷バランス時の居室の室温 [℃]
    """
    # 事前条件: 次数チェック
    assert A_HCZ_i.shape == (5,), "A_HCZ_iの次数が想定外"
    assert Theta_HBR_i.shape == (5, 1), "Theta_HBR_iの次数が想定外"

    # 熱容量の取得
    A_HCZ_i = A_HCZ_i.reshape(-1,1)  # (5,) -> (5, 1)
    cbri = jjj_carryover_heat.get_C_BR_i(A_HCZ_i)

    # NOTE('25/04): キャップはかけない
    # 温度が過剰の時に得をするだけでなく、未達の時に損する分を考慮するとのこと
    match (H, C):
        case (np.True_, np.True_):
            raise ValueError("想定外の季節")
        case (np.True_, np.False_):
            # temperature_diff = np.clip(Theta_HBR_i - Theta_star_HBR, 0, None)
            temperature_diff = Theta_HBR_i - Theta_star_HBR
        case (np.False_, np.True_):
            # temperature_diff = np.clip(Theta_star_HBR - Theta_HBR_i, 0, None)
            temperature_diff = Theta_star_HBR - Theta_HBR_i
        case (np.False_, np.False_):
            # 空調なし -> 過剰熱量なし
            return np.zeros((5, 1))
        case _:
            raise ValueError("判別に失敗")

    # 事後条件:
    assert np.all(0 <= cbri), "居室の熱容量は負にならない"
    # assert np.all(0 <= temperature_diff), "ここで温度差は正になるよう算出"
    # NOTE: 負の過剰熱量を仕様上許容しています

    return cbri * temperature_diff / 1_000_000  # J/h -> MJ/h

# NOTE:
# MATRIX[:, t] -> Shape(5, )
# MATRIX[:, t:t+1] -> Shape(5, 1)

def get_L_star_H_i_2024(
        H: bool,
        L_H_i: Array5x1,
        Q_star_trs_prt_i: Array5x1,
        carryover: Array5x1,
    )-> Array5x1:
    """(8-1)(8-2)(8-3) -> 時点版, 過剰熱量を考慮

    Args:
        H: 暖房期
        L_H_i: (時点)暖冷房区画iの1時間当たりの 暖房負荷 [MJ/h]
        Q_star_trs_prt_i: (時点)暖冷房区画iの1時間当たりの 熱損失を含む負荷バランス時の非居室への熱移動 [MJ/h]
        carryover: 過剰熱量 [MJ]

    Returns:
        暖冷房区画iの一時点の 熱損失を含む負荷バランス時の暖房負荷 [MJ]
    """
    # 実行条件: 指定時刻tが暖房期である
    assert L_H_i.shape == (5, 1), "L_H_iの行列数が想定外"
    if not H or np.all(L_H_i <= 0):
        return np.zeros((5, 1))

    # 繰越熱量は負荷低減に働く
    L_star_H_i = L_H_i + Q_star_trs_prt_i - carryover
    L_star_H_i = np.clip(L_star_H_i, 0, None)

    # 事後条件:
    assert np.all(0 <= L_star_H_i), "負荷バランス時の暖房負荷は負にならない"
    return L_star_H_i

def get_L_star_CS_i_2024(
        C: bool,
        L_CS_i: Array5x1,
        Q_star_trs_prt_i: Array5x1,
        carryover: Array5x1,
    )-> Array5x1:
    """(9-1)(9-2)(9-3) -> 時点版, 過剰熱量を考慮

    Args:
        C: 冷房期
        L_CS_i: (時点)暖冷房区画iの1時間当たりの 冷房顕熱負荷 [MJ/h]
        Q_star_trs_prt_i: (時点)暖冷房区画iの1時間当たりの 熱損失を含む負荷バランス時の非居室への熱移動 [MJ/h]
        carryover: 過剰熱量 [MJ]

    Returns:
        暖冷房区画iの一時点の 熱損失を含む負荷バランス時の冷房顕熱負荷 [MJ]
    """
    # 実行条件: 指定時刻tが暖房期である
    assert L_CS_i.shape == (5, 1), "L_CS_iの行列数が想定外"
    if not C or np.all(L_CS_i <= 0):
        return np.zeros((5, 1))

    # 繰越熱量は負荷低減に働く
    L_star_CS_i = L_CS_i + Q_star_trs_prt_i - carryover
    L_star_CS_i = np.clip(L_star_CS_i, 0, None)

    # 事後条件:
    assert np.all(0 <= L_star_CS_i), "負荷バランス時の冷房顕熱負荷は負にならない"
    return L_star_CS_i

# NOTE: d_t 長の配列を引数に渡さず、時点のみの引数にするメリット
# 意図しない前後インデックスの利用・上書きがないことが保証される
# デメリットは、クライアントコードに知識が必要になる(時点の選択)

def get_Theta_HBR_i_2023(
        isFirst: bool, H: bool, C: bool, M: bool,
        Theta_star_HBR: float,
        V_supply_i: Array5x1,
        Theta_supply_i: Array5x1,
        U_prt: float,
        A_prt_i: Array5x1,
        Q: float,
        A_HCZ_i: Array5x1,
        L_star_H_i: Array5x1,
        L_star_CS_i: Array5x1,
        Theta_HBR_before_i: Array5x1
        ) -> Array5x1:
    """(46-1)(46-2)(46-3) -> 時点版, 過剰熱量を考慮

    Args:
        Theta_star_HBR: (時点)負荷バランス時の居室の室温 [℃]
        V_supply_i: 日付dの時刻tにおける暖冷房区画iの吹き出し風量 [m3/h]
        Theta_supply_i: 日付dの時刻tにおける負荷バランス時の居室の室温 [℃]
        U_prt: 間仕切りの熱貫流率 [W/(m2・K)]
        A_prt_i: 暖冷房区画iから見た非居室の間仕切りの面積 [m2]
        Q: 当該住戸の熱損失係数 [W/(m2・K)]
        A_HCZ_i: 暖冷房区画iの 床面積 [m2]
        L_star_H_i: (時点)暖冷房区画iの 間仕切りの熱取得を含む実際の暖房負荷 [MJ/h]
        L_star_CS_i: (時点)暖冷房区画iの 間仕切りの熱取得を含む実際の冷房顕熱負荷 [MJ/h]
        Theta_HBR_before_i: (※前時点)暖冷房区画iの 実際の居室の室温 [℃]

    Returns:
        (時点)暖冷房区画iの 過剰熱量繰越を考慮した 実際の居室の室温 [℃]
    """
    # 事前条件: 次数チェック
    assert V_supply_i.shape == (5, 1), "V_supply_iの行列数が想定外"
    assert Theta_supply_i.shape == (5, 1), "Theta_supply_iの行列数が想定外"
    assert A_prt_i.shape == (5, 1), "A_prt_iの行列数が想定外"
    assert A_HCZ_i.shape == (5, 1), "A_HCZ_iの行列数が想定外"
    assert L_star_H_i.shape == (5, 1), "L_star_H_iの行列数が想定外"
    assert L_star_CS_i.shape == (5, 1), "L_star_CS_iの行列数が想定外"
    assert Theta_HBR_before_i.shape == (5, 1), "Theta_HBR_before_iの行列数が想定外"

    # NOTE: t = 1時間区切りなので必ずhで整理

    # 熱容量(間仕切り) [J/(K・h)]
    c_prt = U_prt * A_prt_i * 3600

    # 熱損失 [J/(K・h)]
    heat_loss = Q * A_HCZ_i * 3600

    # 熱容量(居室) [J/K]
    cbri = jjj_carryover_heat.get_C_BR_i(A_HCZ_i)

    # NOTE: 250625 梅本様より絶対値を外すよう指摘に対応しました
    carryover_theta_diff = Theta_HBR_before_i - Theta_star_HBR

    # 熱繰越による熱容量[J] (1/1/0:00 なしとする)
    carryover_capacity = 0 if isFirst else cbri * carryover_theta_diff

    # 熱容量(空調空気) [J/(K・h)]
    c_ac_air = dc.get_c_p_air() * dc.get_rho_air() * V_supply_i

    ac_theta_diff = np.abs(Theta_supply_i - Theta_star_HBR)  # [K]
    ac_capacity = c_ac_air * ac_theta_diff  # [J/h]

    if (H and C):
        raise ValueError("想定外の季節")
    elif H:  # 暖房期 (46-1)
        load_capacity = L_star_H_i * 10**6  # [MJ/h] -> [J/h]
        Theta_HBR_i = Theta_star_HBR \
                    + (ac_capacity + carryover_capacity - load_capacity) \
                    / (c_ac_air + c_prt + heat_loss + cbri)
        # NOTE: 負荷バランス時の居室の室温で下限を設定 -> しない
        # Theta_HBR_i = np.clip(Theta_HBR_i, Theta_star_HBR, None)
    elif C:  # 冷房期 (46-2)
        load_capacity = L_star_CS_i * 10**6  # [MJ/h] -> [J/h]
        Theta_HBR_i = Theta_star_HBR \
                    -1 * (ac_capacity + carryover_capacity - load_capacity) \
                    / (c_ac_air + c_prt + heat_loss + cbri)
        # NOTE: 負荷バランス時の居室の室温で下限を設定 -> しない
        # Theta_HBR_i = np.clip(Theta_HBR_i, None, Theta_star_HBR)
    elif M:  # 中間期 (46-3)
        Theta_HBR_i = Theta_star_HBR * np.ones((5, 1))

    # 事後条件:
    assert Theta_HBR_i.shape == (5, 1), "Theta_HBR_iの行列数が想定外"
    return Theta_HBR_i

def get_Theta_NR_2023(
        isFirst: bool, H: bool, C: bool, M: bool,
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
        Theta_NR_before: float
        ) -> float:
    """ (48a)(48b)(48c)(48d) -> 時点版, 過剰熱量を考慮

    Args:
        Theta_star_NR: (時点)実際の非居室の室温 [℃]
        Theta_star_HBR: (時点)負荷バランス時の居室の室温 [℃]
        Theta_HBR_i: (時点)暖冷房区画iの 実際の居室の室温 [℃]
        A_NR: 非居室の床面積 [m2]
        V_vent_l_NR: (時点)非居室の局所換気量 [m3/h]
        V_dash_supply_i: (時点)暖冷房区画iの VAV調整前の吹き出し風量 [m3/h]
        V_supply_i: (時点)暖冷房区画iの 吹き出し風量 [m3/h]
        U_prt: 間仕切りの熱貫流率 [W/(m2・K)]
        A_prt_i: 暖冷房区画iから見た非居室の間仕切りの面積 [m2]
        Q: 当該住戸の熱損失係数 [W/(m2・K)]
        Theta_NR_before: (※前時点)非居室の室温 [℃]

    Returns:
        (時点)過剰熱量繰越を考慮した 実際の非居室の室温 [℃]
    """
    # 事前条件: 次数チェック
    assert Theta_HBR_i.shape == (5, 1), "Theta_HBR_iの行列数が想定外"
    assert V_dash_supply_i.shape == (5, 1), "V_dash_supply_iの行列数が想定外"
    assert V_supply_i.shape == (5, 1), "V_supply_iの行列数が想定外"
    assert A_prt_i.shape == (5, 1), "A_prt_iの行列数が想定外"

    # 熱容量(間仕切り) [W/K]
    c_prt = U_prt * A_prt_i

    c_p_air = dc.get_c_p_air()  # [J/(kg・K)]
    rho_air = dc.get_rho_air()  # [kg/m3]

    # (48d) [W/K]
    k_prt_dash_i = c_p_air * rho_air * (V_dash_supply_i / 3600) + c_prt
    # (48c) [W/K]
    k_prt_i = c_p_air * rho_air * (V_supply_i / 3600) + c_prt
    # (48b) [W/K]
    k_evp = (Q - 0.35 * 0.5 * 2.4) * A_NR \
        + c_p_air * rho_air * (V_vent_l_NR / 3600)

    val1 = -1 * np.sum(k_prt_dash_i) * (Theta_star_HBR - Theta_star_NR)
    val2 = np.sum(k_prt_i * (Theta_HBR_i - Theta_star_NR))
    val3 = k_evp + np.sum(k_prt_i)

    # 過剰熱量発生条件
    H = H and (Theta_NR_before >= Theta_star_NR)
    C = C and (Theta_NR_before <= Theta_star_NR)

    # H,C のチェック
    if (H and C):
        raise ValueError("想定外の季節")
    # 暖房期に 過剰熱量が有効
    elif H:
        ac_theta_diff = Theta_NR_before - Theta_star_NR
        assert ac_theta_diff >= 0, f"想定外の温度差 {Theta_NR_before} >= {Theta_star_NR}"
    # 冷房期に 過剰熱量が有効
    elif C:
        ac_theta_diff = Theta_NR_before - Theta_star_NR
        assert ac_theta_diff <= 0, f"想定外の温度差 {Theta_NR_before} <= {Theta_star_NR}"
    else:
        ac_theta_diff = 0  # 使用されないが定義は必要
        pass

    # (48a) NOTE: isFirst と過剰熱量無効のとき元式と一致するべき
    C_NR = 0 if isFirst or not (H or C)  \
        else jjj_carryover_heat.get_C_NR(A_NR) / 3600
    Theta_NR = Theta_star_NR + (val1 + val2 + ac_theta_diff * C_NR) / (val3 + C_NR)

    # TODO: Theta_NR が単増加してしまう問題がある
    # -> 過剰熱量持越しの追い空調の停止条件を追加することが考えられる
    # 今は、下のキャップロジックで仮対応しています

    # 空調されている部屋以上に過剰熱量繰越が効くことはないため
    if H:
        Theta_NR = np.clip(Theta_NR, None, Theta_star_HBR)
    elif C:
        Theta_NR = np.clip(Theta_NR, Theta_star_HBR, None)

    # NOTE: axis オプションによる次数の変化
    # 次数を意識せずにfloatに総計するなら axisなしがよい
    # np.sum(k_prt_i, axis=0) -> shape(5,1) -> shape(1,)
    # np.sum(k_prt_i) -> shape(5,1) -> float

    # 事後条件: 次数チェック
    assert isinstance(Theta_NR, float), "Theta_NRの次数が想定外"
    return Theta_NR
