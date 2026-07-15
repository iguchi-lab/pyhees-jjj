import typing
import numpy as np
import pandas as pd

from jjjexperiment.denchu.inputs.heating import DenchuCatalogSpecification as H_CatalogSpec, RealInnerCondition as H_RealInnerCondition
from jjjexperiment.denchu.inputs.cooling import DenchuCatalogSpecification as C_CatalogSpec, RealInnerCondition as C_RealInnerCondition
from jjjexperiment.denchu.utils import *

from jjjexperiment.logger import LimitedLoggerAdapter as _logger, log_res

ATM_AIR_PRESSURE = 1013.25
""" 大気圧[hPa] """
BF = 0.2  # NOTE: 論文内では0.2に固定
""" Bypass-Factor """

class Spec:
    """ 機器特性(メーカー公表値)をまとめるクラス
    """
    def __init__(self,
                P_rac_min, P_rac_rtd, P_rac_max,
                q_rac_min, q_rac_rtd, q_rac_max,
                V_inner, V_outer, name=""):
        self._name = name
        self._p_rac_min, self._p_rac_rtd, self._p_rac_max = P_rac_min, P_rac_rtd, P_rac_max
        self._q_rac_min, self._q_rac_rtd, self._q_rac_max = q_rac_min, q_rac_rtd, q_rac_max
        self._V_inner, self._V_outer = V_inner, V_outer

    @property
    def name(self):
        """ 機器名 """; return self._name
    @property
    def P_rac_min(self):
        """ JIS条件での消費電力(最小)[W] """; return self._p_rac_min
    @property
    def P_rac_rtd(self):
        """ JIS条件での消費電力(定格)[W] """; return self._p_rac_rtd
    @property
    def P_rac_max(self):
        """ JIS条件での消費電力(最大)[W] """; return self._p_rac_max

    # NOTE: 室内から除去するエネルギー = 冷房能力 (論文より)
    @property
    def q_rac_min(self):
        """ JIS条件での熱処理能力(最小)[kW] """; return self._q_rac_min
    @property
    def q_rac_rtd(self):
        """ JIS条件での熱処理能力(定格)[kW] """; return self._q_rac_rtd
    @property
    def q_rac_max(self):
        """ JIS条件での熱処理能力(最大)[kW] """; return self._q_rac_max

    # NOTE: evp,cnd は冷暖で逆になるため室内機・室外機の識別子にはならない
    # よって inner, outer で表現する
    @property
    def V_inner(self):
        """ 室内機の吸込み空気量[m3/min] """; return self._V_inner
    @property
    def V_outer(self):
        """ 室外機の吸込み空気量[m3/min] """; return self._V_outer

    def outputDataFrame(self, data_frame):
        """csv出力用のデータフレーム"""
        data_frame.loc[Spec.P_rac_min.__doc__] = [self.P_rac_min]
        data_frame.loc[Spec.P_rac_rtd.__doc__] = [self.P_rac_rtd]
        data_frame.loc[Spec.P_rac_max.__doc__] = [self.P_rac_max]
        data_frame.loc[Spec.q_rac_min.__doc__] = [self.q_rac_min]
        data_frame.loc[Spec.q_rac_rtd.__doc__] = [self.q_rac_rtd]
        data_frame.loc[Spec.q_rac_max.__doc__] = [self.q_rac_max]
        data_frame.loc[Spec.V_inner.__doc__] = [self.V_inner]
        data_frame.loc[Spec.V_outer.__doc__] = [self.V_outer]
        return data_frame

class Condition():
    """ エアコンの使用条件をまとめるクラス
    """
    def __init__(self, T_ein, T_cin, X_ein, X_cin):
        self._T_ein, self._T_cin = T_ein, T_cin
        self._X_ein, self._X_cin = X_ein, X_cin

    # NOTE: ein, cin が部屋の内・外かどうかは冷房・暖房によって変わる
    @property
    def T_ein(self):
        """ 蒸発器と接する空気の温度[℃] """; return self._T_ein
    @property
    def T_cin(self):
        """ 凝集器と接する空気の温度[℃] """; return self._T_cin

    @property
    def X_ein(self):
        """ 蒸発器と接する空気の絶対湿度[g/kgDA] """; return self._X_ein
    @property
    def X_cin(self):
        """ 蒸発器と接する空気の絶対湿度[g/kgDA] """; return self._X_cin

    def outputDataFrame(self, data_frame):
        """csv出力用のデータフレーム"""
        data_frame.loc[Condition.T_ein.__doc__] = [self.T_ein]
        data_frame.loc[Condition.T_cin.__doc__] = [self.T_cin]
        data_frame.loc[Condition.X_ein.__doc__] = [self.X_ein]
        data_frame.loc[Condition.X_cin.__doc__] = [self.X_cin]
        return data_frame

def get_DataFrame_denchu_modeling_consts(
        spec: H_CatalogSpec | C_CatalogSpec,
        R2: float, R1: float, R0: float,
        real_inner: H_RealInnerCondition | C_RealInnerCondition,
        Pc: float) -> pd.DataFrame:
        """ 電柱研モデル化の定数確認用データフレーム
            Args:
                Pc: ファン等消費電力Pc [W]
        """
        df = pd.DataFrame(index=['電中研モデリング定数'], columns=['値'])
        df = spec.outputDataFrame(df)
        df = pd.concat([
            df,
            pd.DataFrame(
                [[R2], [R1], [R0]],
                columns=df.columns,
                index=['R2', 'R1', 'R0'],
            ),
        ])
        df.loc['ファン等消費電力Pc [W]'] = [Pc]
        df.loc['実際の使用状況におけるエアコン吸込み空気の 温度 [℃]'] = [real_inner.Theta_rac_real_inner]
        df.loc['実際の使用状況におけるエアコン吸込み空気の 相対温度 [%]'] = [real_inner.RH_rac_real_inner]
        return df

@log_res(['T_evp_C', 'T_cnd_C'])
def calc_reibai_phase_T_C(q: float, P: float, spec: C_CatalogSpec)-> typing.Tuple[float, float]:
    """ q, P [kW]で統一
    """
    M_evp = (1-BF) * m3ph_to_kgDAps(spec.V_rac_inner * 60, spec.T_evp)
    Ca_ein = get_Ca(spec.X_evp / 1000)  # [kJ/(kgDA*K)]
    T_evp = spec.T_evp - q / (M_evp*Ca_ein)

    # 水蒸気保持できなくなった場合
    if is_over_saturated(T_evp, spec.X_evp):
        T_evp = avoid_over_saturated(T_evp, spec.X_evp)

    M_cnd = (1-BF) * m3ph_to_kgDAps(spec.V_rac_outer * 60, spec.T_cnd)
    Ca_cin = get_Ca(spec.X_cnd / 1000)  # [kJ/(kgDA*K)]
    T_cnd = spec.T_cnd + (q+P) / (M_cnd*Ca_cin)

    return T_evp, T_cnd  # 冷媒 蒸発/凝縮 温度[℃]

@log_res(['T_evp_H', 'T_cnd_H'])
def calc_reibai_phase_T_H(q: float, P: float, spec: H_CatalogSpec)-> typing.Tuple[float, float]:
    """ q, P [kW]で統一
    """
    M_evp = (1-BF) * m3ph_to_kgDAps(spec.V_rac_outer * 60, spec.T_evp)
    Ca_ein = get_Ca(spec.X_evp / 1000)  # [kJ/(kgDA*K)]
    T_evp = spec.T_evp - (q-P) / (M_evp*Ca_ein)

    # 水蒸気保持できなくなった場合
    if is_over_saturated(T_evp, spec.X_evp):
        T_evp = avoid_over_saturated(T_evp, spec.X_evp)

    M_cnd = (1-BF) * m3ph_to_kgDAps(spec.V_rac_inner * 60, spec.T_cnd)
    Ca_cin = get_Ca(spec.X_cnd / 1000)  # [kJ/(kgDA*K)]
    T_cnd = spec.T_cnd + q / (M_cnd*Ca_cin)

    return T_evp, T_cnd  # 冷媒 蒸発/凝縮 温度[℃]

def _solve_mtx(A: np.matrix, Y: np.matrix) -> typing.Tuple[float, float]:
    mtx_ans = np.linalg.inv(A) * Y  # shape(2, 1)
    return (mtx_ans[0, 0], mtx_ans[1, 0])

def calc_R_and_Pc_C(spec: C_CatalogSpec) -> typing.Tuple[float, float, float, float]:
    """ 成績係数比Rの近似式の係数とファン等消費電力Pc[kW]
    """
    def coeffs_for_simultaneous_C(label: str, q:float, P:float, spec: C_CatalogSpec) -> typing.Tuple[float, float, float]:
        """ q, P [kW]で統一 """
        T_evp, T_cnd = calc_reibai_phase_T_C(q, P, spec)
        A = (T_cnd - T_evp) / (T_evp + 273.15)  # 冷房と異なる
        B = 1 / q
        COP = q / P; Y = 1 / COP
        return A, B, Y  # A・R' + B・Pc = Y (R'=1/R) の形にする

    @log_res(['R_minrtd_C', 'Pc_C'])
    def R_minrtd_C(spec: C_CatalogSpec) -> typing.Tuple[float, float]:
        A1, B1, Y1 = coeffs_for_simultaneous_C('min_C', spec.q_rac_min, 0.001*spec.P_rac_min, spec)
        A2, B2, Y2 = coeffs_for_simultaneous_C('rtd_C', spec.q_rac_rtd, 0.001*spec.P_rac_rtd, spec)
        mtx_A, mtx_Y = np.matrix([[A1, B1], [A2, B2]]), np.matrix([[Y1], [Y2]])
        R_minrtd_dash, Pc = _solve_mtx(mtx_A, mtx_Y)
        R_minrtd = 1 / R_minrtd_dash  # NOTE: 最小・定格時のR同一
        return R_minrtd, Pc

    R_minrtd, Pc = R_minrtd_C(spec)

    def R_max_C(Pc, q, P, spec: C_CatalogSpec) -> float:
        """ Pc, q, P [kW]で統一 """
        T_evp_max, T_cnd_max = calc_reibai_phase_T_C(q, P, spec)
        COP = q / P
        right = COP * q / (q - COP*Pc)  # (7)式右辺
        left = (T_evp_max + 273.15) / (T_cnd_max - T_evp_max)  # (7)式左辺(係数部)
        return right / left

    # NOTE: 論文より最大時のRのみ別に計算する
    R_max = R_max_C(Pc, spec.q_rac_max, 0.001*spec.P_rac_max, spec)

    Qs = np.array([spec.q_rac_min, spec.q_rac_rtd, spec.q_rac_max])
    Rs = np.array([R_minrtd, R_minrtd, R_max])
    coeffs = np.polyfit(Qs, Rs, 2)  # 二次式に近似し係数を取得

    R_poly_a2 = coeffs[0]
    R_poly_a1 = coeffs[1]
    R_poly_a0 = coeffs[2]
    _logger.info(f"a2: {R_poly_a2}\na1: {R_poly_a1}\na0: {R_poly_a0}")
    return R_poly_a2, R_poly_a1, R_poly_a0, Pc

def calc_R_and_Pc_H(spec: H_CatalogSpec) -> typing.Tuple[float, float, float, float]:
    """ 成績係数比Rの近似式の係数とファン等消費電力Pc[kW]
    """
    def coeffs_for_simultaneous_H(label: str, q:float, P:float, spec: H_CatalogSpec) -> typing.Tuple[float, float, float]:
        """ q, P [kW]で統一 """
        T_evp, T_cnd = calc_reibai_phase_T_H(q, P, spec)
        A = (T_cnd - T_evp) / (T_cnd + 273.15)  # 冷房と異なる
        B = 1 / q
        COP = q / P; Y = 1 / COP
        return A, B, Y  # A・R' + B・Pc = Y (R'=1/R) の形にする

    @log_res(['R_minrtd_H', 'Pc_H[kW]'])
    def R_minrtd_H(spec: H_CatalogSpec) -> typing.Tuple[float, float]:
        A1, B1, Y1 = coeffs_for_simultaneous_H('min_H', spec.q_rac_min, 0.001*spec.P_rac_min, spec)
        A2, B2, Y2 = coeffs_for_simultaneous_H('rtd_H', spec.q_rac_rtd, 0.001*spec.P_rac_rtd, spec)
        mtx_A, mtx_Y = np.matrix([[A1, B1], [A2, B2]]), np.matrix([[Y1], [Y2]])
        R_minrtd_dash, Pc = _solve_mtx(mtx_A, mtx_Y)
        R_minrtd = 1 / R_minrtd_dash  # NOTE: 最小・定格時のR同一
        return R_minrtd, Pc

    R_minrtd, Pc = R_minrtd_H(spec)

    def R_max_H(Pc, q, P, spec: H_CatalogSpec) -> float:
        """ Pc, q, P [kW]で統一 """
        T_evp_max, T_cnd_max = calc_reibai_phase_T_H(q, P, spec)
        COP = q / P
        right = COP * q / (q - COP*Pc)  # (7)式右辺
        left = (T_cnd_max + 273.15) / (T_cnd_max - T_evp_max)  # (7)式左辺(係数部)
        return right / left

    # NOTE: 論文より最大時のRのみ別に計算する
    R_max = R_max_H(Pc, spec.q_rac_max, 0.001*spec.P_rac_max, spec)

    Qs = np.array([spec.q_rac_min, spec.q_rac_rtd, spec.q_rac_max])
    Rs = np.array([R_minrtd, R_minrtd, R_max])

    coeffs = np.polyfit(Qs, Rs, 2)  # 二次式に近似し係数を取得
    R_poly_a2, R_poly_a1, R_poly_a0 = coeffs[0], coeffs[1], coeffs[2]
    _logger.info(f"a2: {R_poly_a2}\na1: {R_poly_a1}\na0: {R_poly_a0}")

    return R_poly_a2, R_poly_a1, R_poly_a0, Pc
