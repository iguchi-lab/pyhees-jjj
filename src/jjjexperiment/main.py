import json
from injector import inject
import pandas as pd
from datetime import datetime

from pyhees.section2_1_b import get_f_prim
from pyhees.section4_1 import calc_heating_load, calc_cooling_load, get_virtual_heating_devices
from pyhees.section4_1_a import calc_heating_mode

# ダクト式セントラル空調機
import pyhees.section4_2_a as dc_a
import pyhees.section4_2_b as dc_spec

# 床下
import pyhees.section3_1 as ld
from pyhees.section3_2 import calc_r_env, get_Q_dash, get_mu_H, get_mu_C

""" オーバーライドロジック """

import jjjexperiment.section4_2 as jjj_dc
import jjjexperiment.section4_2_a as jjj_dc_a

""" 独自ロジック """

# 電中研モデルロジック
import jjjexperiment.denchu.denchu_1
import jjjexperiment.denchu.denchu_2
import jjjexperiment.denchu.inputs.heating as jjj_denchu_heat_ipt
import jjjexperiment.denchu.inputs.cooling as jjj_denchu_cool_ipt

import jjjexperiment.inputs.di_container as jjj_ipt_di
from jjjexperiment.inputs.di_container import *
# データクラス
from jjjexperiment.inputs.common import HouseInfo, OuterSkin, HEX
from jjjexperiment.inputs.ac_setting import HeatingAcSetting, CoolingAcSetting
from jjjexperiment.inputs.heating import CRACSpecification as HeatCRACSpec
from jjjexperiment.inputs.cooling import CRACSpecification as CoolCRACSpec
# ドメインサービス
from jjjexperiment.inputs.climate_service import ClimateService
from jjjexperiment.inputs.ac_quantity_service import HeatQuantityService, CoolQuantityService

import jjjexperiment.constants as jjj_consts
from jjjexperiment.result import *
from jjjexperiment.logger import LimitedLoggerAdapter as _logger  # デバッグ用ロガー
from jjjexperiment.common import *
import jjjexperiment.underfloor_ac.inputs as jjj_ufac_ipt

# [F25-01] 最低風量の直接入力
import jjjexperiment.v_min_input as jjj_V_min_input

def calc(input_data: dict, test_mode=False):
    case_name = input_data.get('case_name', 'default')
    with open(case_name + jjj_consts.version_info() + '_input.json', 'w') as f:
        json.dump(input_data, f, indent=4)

    # グローバル定数の設定
    # NOTE: 必要最小限に留めること
    jjj_consts.set_constants(input_data)

    injector = jjj_ipt_di.create_injector_from_json(input_data, test_mode)
    # NOTE: 引数全てを型解決できるようにする必要があった
    return injector.call_with_injection(calc_main)

@inject
def calc_main(
    injector: Injector,
    # NOTE: 型解決するだけなら下記のみで充分だが、文脈の追加変更を行うため injector 自身も受取
    test_mode: jjj_ipt_di.TestMode,
    case_name: jjj_ipt_di.CaseName,
    climateFile: jjj_ipt_di.ClimateFile,
    loadFile: jjj_ipt_di.LoadFile,
    v_min_heating_input: jjj_V_min_input.inputs.heating.InputMinVolumeInput,
    v_min_cooling_input: jjj_V_min_input.inputs.cooling.InputMinVolumeInput,
    house: HouseInfo,
    skin: OuterSkin,
    hex: HEX,
    ufac: jjj_ufac_ipt.common.UnderfloorAc,
    heat_ac_setting: HeatingAcSetting,
    cool_ac_setting: CoolingAcSetting,
    heat_CRAC: HeatCRACSpec,
    cool_CRAC: CoolCRACSpec,
    heat_denchu_catalog: jjj_denchu_heat_ipt.DenchuCatalogSpecification,
    cool_denchu_catalog: jjj_denchu_cool_ipt.DenchuCatalogSpecification,
    heat_real_inner: jjj_denchu_heat_ipt.RealInnerCondition,
    cool_real_inner: jjj_denchu_cool_ipt.RealInnerCondition
    ) -> dict | None:
    print("q_rtd_C, q_rtd_H, q_max_C, q_max_H, e_rtd_C, e_rtd_H")
    print(cool_CRAC.q_rtd, heat_CRAC.q_rtd, cool_CRAC.q_max, heat_CRAC.q_max, cool_CRAC.e_rtd, heat_CRAC.e_rtd)

    _logger.info(f"q_rtd_C [w]: {cool_CRAC.q_rtd}")  # q[W] 送風機の単位[W]と同じ
    _logger.info(f"q_max_C [w]: {cool_CRAC.q_max}")
    _logger.info(f"e_rtd_C [-]: {cool_CRAC.e_rtd}")
    _logger.info(f"q_rtd_H [w]: {heat_CRAC.q_rtd}")
    _logger.info(f"q_max_H [w]: {heat_CRAC.q_max}")
    _logger.info(f"e_rtd_H [-]: {heat_CRAC.e_rtd}")

    # ドメインサービス
    climate = ClimateService(house.region, ufac, climateFile)
    heat_quantity = HeatQuantityService(heat_ac_setting, house.region, house.A_A)
    cool_quantity = CoolQuantityService(cool_ac_setting, house.region, house.A_A)

    H_MR = None
    """主たる居室暖房機器"""
    H_OR = None
    """その他居室暖房機器"""
    H_HS = None
    """温水暖房の種類"""
    C_MR = None
    """主たる居室冷房機器"""
    C_OR = None
    """その他居室冷房機器"""

    # 実質的な暖房機器の仕様を取得
    spec_MR, spec_OR = get_virtual_heating_devices(house.region, H_MR, H_OR)
    # 暖房方式及び運転方法の区分
    mode_MR, mode_OR = calc_heating_mode(region=house.region, H_MR=spec_MR, H_OR=spec_OR)

    ##### 暖房負荷の取得（MJ/h）
    L_H_d_t_i: np.ndarray
    """暖房負荷 [MJ/h]"""

    set_current_injector(injector)  # ネスト内からの利用に備える

    # L_dash_H_R_d_t_i, L_dash_CS_R_d_t_iは負荷ファイルから読み取れないため自動計算する。
    # 読み込んだ負荷と整合性が取れないため、正しい実装ではない。
    L_H_d_t_i, L_dash_H_R_d_t_i, L_dash_CS_R_d_t_i = \
        calc_heating_load(house.region, house.sol_region, house.A_A, house.A_MR, house.A_OR, skin.Q, skin.mu_H, skin.mu_C
            , skin.NV_MR, skin.NV_OR, skin.TS, skin.r_A_ufvnt, hex.to_dict(), skin.underfloor_insulation
            , heat_ac_setting.mode.name, cool_ac_setting.mode.name, spec_MR, spec_OR, mode_MR, mode_OR, skin.SHC)
    if loadFile != '-':
        load = pd.read_csv(loadFile, nrows=24 * 365)
        L_H_d_t_i = load.iloc[::,:12].T.values

    ##### 冷房負荷の取得（MJ/h）
    L_CS_d_t_i: np.ndarray
    """冷房顕熱負荷 [MJ/h]"""
    L_CL_d_t_i: np.ndarray
    """冷房潜熱負荷 [MJ/h]"""

    if loadFile == '-':
        L_CS_d_t_i, L_CL_d_t_i = \
            calc_cooling_load(house.region, house.A_A, house.A_MR, house.A_OR, skin.Q, skin.mu_H, skin.mu_C, skin.NV_MR, skin.NV_OR, skin.r_A_ufvnt
                    , skin.underfloor_insulation, cool_ac_setting.mode.name, heat_ac_setting.mode.name, mode_MR, mode_OR, skin.TS, hex.to_dict())
    else:
        load = pd.read_csv(loadFile, nrows=24 * 365)
        L_CS_d_t_i = load.iloc[::,12:24].T.values
        L_CL_d_t_i = load.iloc[::,24:].T.values

    clear_current_injector()

    # 負荷をあつめたデータクラス
    injector.binder.bind(jjj_dc.Load_DTI, to=jjj_dc.Load_DTI(L_H_d_t_i, L_CS_d_t_i, L_CL_d_t_i, L_dash_H_R_d_t_i, L_dash_CS_R_d_t_i))

    # NOTE: 出力用の下記が計算できるのは、負荷が上書きされない前提
    L_H_d_t: np.ndarray = np.sum(L_H_d_t_i, axis=0)
    """暖房負荷の全区画合計 [MJ/h]"""
    L_CS_d_t: np.ndarray = np.sum(L_CS_d_t_i, axis=0)
    """冷房顕熱負荷の全区画合計 [MJ/h]"""
    L_CL_d_t: np.ndarray = np.sum(L_CL_d_t_i, axis=0)
    """冷房潜熱負荷の全区画合計 [MJ/h]"""

    ##### 暖房消費電力の計算（kWh/h）
    print("暖房消費電力の計算")

    def get_V_hs_dsgn_H(type: 計算モデル, v_fan_rtd, q_rtd_H):
        if type in [
            計算モデル.ダクト式セントラル空調機,
            計算モデル.RAC活用型全館空調_潜熱評価モデル
        ]:
            V_fan_rtd_H = v_fan_rtd
        elif type in [
            計算モデル.RAC活用型全館空調_現行省エネ法RACモデル,
            計算モデル.電中研モデル
        ]:
            V_fan_rtd_H = dc_spec.get_V_fan_rtd_H(q_rtd_H)
        else:
            raise Exception("暖房方式が不正です。")

        return dc_spec.get_V_fan_dsgn_H(V_fan_rtd_H)

    def get_V_hs_dsgn_C(type: 計算モデル, v_fan_rtd: float, q_rtd_C: float):
        if type in [
                計算モデル.ダクト式セントラル空調機,
                計算モデル.RAC活用型全館空調_潜熱評価モデル
            ]:
            pass
        elif type in [
                計算モデル.RAC活用型全館空調_現行省エネ法RACモデル,
                計算モデル.電中研モデル
            ]:
            v_fan_rtd = dc_spec.get_V_fan_rtd_C(q_rtd_C)
        else:
            raise Exception("冷房方式が不正です。")

        return dc_spec.get_V_fan_dsgn_C(v_fan_rtd)

    def arr_summary(arr: np.ndarray):
        return {
            "MAX  ": max(arr),
            "ZEROS": arr.size - np.count_nonzero(arr),
            "AVG  ": np.average(arr[np.nonzero(arr)])
        }

    V_hs_dsgn_H: float =  \
        heat_ac_setting.V_hs_dsgn if heat_ac_setting.V_hs_dsgn > 0  \
        else get_V_hs_dsgn_H(heat_ac_setting.type, heat_quantity.V_fan_rtd, heat_CRAC.q_rtd)
    """ 暖房時の送風機の設計風量 [m3/h] """

    V_hs_dsgn_C_for_ground: float =  \
        cool_ac_setting.V_hs_dsgn if cool_ac_setting.V_hs_dsgn > 0  \
        else get_V_hs_dsgn_C(
            cool_ac_setting.type, cool_quantity.V_fan_rtd, cool_CRAC.q_rtd
        )

    annual_ground_feedback_context = jjj_dc.AnnualGroundFeedbackContext(
        heat_ac_setting=heat_ac_setting,
        cool_ac_setting=cool_ac_setting,
        V_hs_dsgn_H=V_hs_dsgn_H,
        V_hs_dsgn_C=V_hs_dsgn_C_for_ground,
    )
    injector.binder.bind(
        jjj_dc.AnnualGroundFeedbackContext,
        to=annual_ground_feedback_context,
    )

    V_hs_dsgn_C: float = 0.0  # NOTE: 暖房負荷計算時は空
    """ 冷房時の送風機の設計風量 [m3/h] """

    # 型バインド
    assert isinstance(V_hs_dsgn_H, float), "V_hs_dsgn_Hの型が不正"
    injector.binder.bind(jjj_dc.VHS_DSGN_H, to=V_hs_dsgn_H)
    assert isinstance(V_hs_dsgn_C, float), "V_hs_dsgn_Cの型が不正"
    injector.binder.bind(jjj_dc.VHS_DSGN_C, to=V_hs_dsgn_C)

    injector.binder.bind(jjj_dc.ActiveAcSetting, to=heat_ac_setting)  # 暖房負荷アクティブ

    E_UT_H_d_t: np.ndarray
    """暖房設備の未処理暖房負荷の設計一次エネルギー消費量相当値(MJ/h)"""
    E_UT_H_d_t, Theta_hs_out_d_t, Theta_hs_in_d_t, _, _, V_hs_supply_d_t, V_hs_vent_d_t = \
        injector.call_with_injection(jjj_dc.calc_Q_UT_A)
    _logger.NDdebug("V_hs_supply_d_t", V_hs_supply_d_t)
    _logger.NDdebug("V_hs_vent_d_t", V_hs_vent_d_t)

    if heat_ac_setting.type == 計算モデル.電中研モデル:
        R2, R1, R0, P_rac_fan_rtd_H = jjjexperiment.denchu.denchu_1.calc_R_and_Pc_H(heat_denchu_catalog)
        P_rac_fan_rtd_H = 1000 * P_rac_fan_rtd_H  # kW -> W
        simu_R_H = jjjexperiment.denchu.denchu_2.simu_R(R2, R1, R0)

        """ 電柱研モデルのモデリング定数の確認のためのCSV出力 """
        df_denchu_consts = jjjexperiment.denchu.denchu_1 \
            .get_DataFrame_denchu_modeling_consts(heat_denchu_catalog, R2, R1, R0, heat_real_inner, P_rac_fan_rtd_H)
        df_denchu_consts.to_csv(case_name + jjj_consts.version_info() + '_denchu_consts_H_output.csv', encoding='cp932')
        del R2, R1, R0
    else:
        P_rac_fan_rtd_H = V_hs_dsgn_H * heat_ac_setting.f_SFP
    """定格暖房能力運転時の送風機の消費電力(W)"""
    _logger.info(f"P_rac_fan_rtd_H [W]: {P_rac_fan_rtd_H}")

    # (3) 日付dの時刻tにおける1時間当たりの熱源機の平均暖房能力(W)
    q_hs_H_d_t = \
        dc_a.get_q_hs_H_d_t(Theta_hs_out_d_t, Theta_hs_in_d_t, V_hs_supply_d_t, climate.get_C_df_H_d_t(), house.region)
    # NOTE: 消費電力量計算に広く使用されており、広いスコープで定義してよいことを確認済

    HCM = climate.get_HCM_d_t()

    E_E_fan_H_d_t: Array8760
    # NOTE: 潜熱評価モデルはベース式が異なるため 最低風量・最低電力 直接入力ロジック反映から除外する
    if heat_ac_setting.type == 計算モデル.RAC活用型全館空調_潜熱評価モデル:
        print(heat_ac_setting.type)

        import jjjexperiment.latent_load.section4_2_a as jjj_latent_dc_a
        E_E_fan_H_d_t = jjj_latent_dc_a.get_E_E_fan_H_d_t(V_hs_vent_d_t, q_hs_H_d_t, heat_ac_setting.f_SFP)

    elif heat_ac_setting.type in [
        計算モデル.ダクト式セントラル空調機,
        計算モデル.RAC活用型全館空調_現行省エネ法RACモデル,
        計算モデル.電中研モデル
    ]:
        print(heat_ac_setting.type)

        # [F25-01] 最低風量・最低電力 直接入力
        match v_min_heating_input.input_V_hs_min:
            case 最低風量直接入力.入力しない:
                print(最低風量直接入力.入力しない)

                # 従来式
                E_E_fan_H_d_t = \
                    dc_a.get_E_E_fan_H_d_t(
                        # ルームエアコンファン(P_rac_fan) OR 循環ファン(P_fan)
                        P_rac_fan_rtd_H if heat_ac_setting.type == 計算モデル.RAC活用型全館空調_現行省エネ法RACモデル else heat_quantity.P_fan_rtd
                        , V_hs_vent_d_t  # 上書きナシ
                        , V_hs_supply_d_t
                        , V_hs_dsgn_H
                        , q_hs_H_d_t  # W
                        , house.region
                        , heat_ac_setting.f_SFP  # NOTE: 従来式は標準値固定だがカスタム値を反映
                        , heat_ac_setting.subtract_ventilation_power)

            case 最低風量直接入力.入力する:
                print(最低風量直接入力.入力する)

                match v_min_heating_input.input_E_E_fan_min:
                    case 最低電力直接入力.入力しない:
                        print(最低電力直接入力.入力しない)

                        # 従来式
                        E_E_fan_H_d_t = \
                            dc_a.get_E_E_fan_H_d_t(
                                # ルームエアコンファン(P_rac_fan) OR 循環ファン(P_fan)
                                P_rac_fan_rtd_H if heat_ac_setting.type == 計算モデル.RAC活用型全館空調_現行省エネ法RACモデル else heat_quantity.P_fan_rtd
                                , V_hs_vent_d_t  # 上書きアリ
                                , V_hs_supply_d_t
                                , V_hs_dsgn_H
                                , q_hs_H_d_t  # W
                                , house.region
                                , heat_ac_setting.f_SFP  # NOTE: 従来式は標準値固定だがカスタム値を反映
                                , ファン消費電力から換気分を引く.換気分を引かない)  # 最低風量を入力する場合、換気分は引かない

                    case 最低電力直接入力.入力する:
                        print(最低電力直接入力.入力する)

                        E_E_fan_min_H = v_min_heating_input.E_E_fan_min
                        E_E_fan_logic = v_min_heating_input.E_E_fan_logic

                        from jjjexperiment.v_min_input.section4_2_a import get_E_E_fan_d_t
                        E_E_fan_H_d_t = get_E_E_fan_d_t(
                                E_E_fan_logic
                                # ルームエアコンファン(P_rac_fan) OR 循環ファン(P_fan)
                                , P_rac_fan_rtd_H if heat_ac_setting.type == 計算モデル.RAC活用型全館空調_現行省エネ法RACモデル else heat_quantity.P_fan_rtd
                                , V_hs_vent_d_t  # 上書きアリ
                                , V_hs_supply_d_t
                                , V_hs_dsgn_H
                                , E_E_fan_min_H
                                , house.region
                                , for_cooling=False)
                    case _:
                        raise ValueError
            case _:
                raise ValueError
    else:
        raise ValueError

    E_E_H_d_t: np.ndarray
    """日付dの時刻tにおける1時間当たり 暖房時の消費電力量 [kWh/h]"""

    if heat_ac_setting.type in [
        計算モデル.ダクト式セントラル空調機,
        計算モデル.RAC活用型全館空調_潜熱評価モデル
    ]:
        E_E_H_d_t \
            = jjj_dc_a.calc_E_E_H_d_t_type1_and_type3(
                type = heat_ac_setting.type,
                E_E_fan_H_d_t = E_E_fan_H_d_t,
                q_hs_H_d_t = q_hs_H_d_t,
                Theta_hs_out_d_t = Theta_hs_out_d_t,
                Theta_hs_in_d_t = Theta_hs_in_d_t,
                Theta_ex_d_t = climate.get_Theta_ex_d_t(),  # 空気温度
                V_hs_supply_d_t = V_hs_supply_d_t,  # 風量
                q_hs_rtd_C = cool_quantity.q_hs_rtd,  # 定格冷房能力※

                equipment_spec = heat_ac_setting.equipment_spec,
                q_hs_min_H = heat_quantity.q_hs_min,  # 最小暖房能力
                q_hs_rtd_H = heat_quantity.q_hs_rtd,
                q_hs_mid_H = heat_quantity.q_hs_mid,
                P_hs_rtd_H = heat_quantity.P_hs_rtd,
                P_hs_mid_H = heat_quantity.P_hs_mid,
                V_fan_rtd_H = heat_quantity.V_fan_rtd,
                V_fan_mid_H = heat_quantity.V_fan_mid,
                P_fan_rtd_H = heat_quantity.P_fan_rtd,
                P_fan_mid_H = heat_quantity.P_fan_mid
            )
    elif heat_ac_setting.type == 計算モデル.RAC活用型全館空調_現行省エネ法RACモデル:
        E_E_H_d_t \
            = jjj_dc_a.calc_E_E_H_d_t_type2(
                type = heat_ac_setting.type,
                region = house.region,
                climateFile = climateFile,
                E_E_fan_H_d_t = E_E_fan_H_d_t,
                q_hs_H_d_t = q_hs_H_d_t,
                e_rtd_H = heat_CRAC.e_rtd,
                q_rtd_H = heat_CRAC.q_rtd,
                q_rtd_C = cool_CRAC.q_rtd,
                q_max_H = heat_CRAC.q_max,
                q_max_C = cool_CRAC.q_max,
                input_C_af_H = heat_CRAC.input_C_af,
                dualcompressor_H = heat_CRAC.dualcompressor
            )
    elif heat_ac_setting.type == 計算モデル.電中研モデル:
        E_E_H_d_t \
            = jjj_dc_a.calc_E_E_H_d_t_type4(
                case_name = case_name,
                type = heat_ac_setting.type,
                region = house.region,
                climateFile = climateFile,
                E_E_fan_H_d_t = E_E_fan_H_d_t,
                q_hs_H_d_t = q_hs_H_d_t,
                V_hs_supply_d_t = V_hs_supply_d_t,
                P_rac_fan_rtd_H = P_rac_fan_rtd_H,
                simu_R_H = simu_R_H,
                spec = heat_denchu_catalog,
                real_inner = heat_real_inner
            )
    else:
        raise Exception("暖房方式が不正です。")

    _logger.NDdebug("E_UT_H_d_t", E_UT_H_d_t)

    df_output2 = pd.DataFrame(index = pd.date_range(datetime(2023,1,1,1,0,0), datetime(2024,1,1,0,0,0), freq='h'))
    df_output2['Q_UT_H_d_A_t [MJ/h]'] = np.full(len(df_output2), np.nan)  # 列数維持のため
    df_output2['Theta_hs_H_out_d_t [℃]']    = Theta_hs_out_d_t
    df_output2['Theta_hs_H_in_d_t [℃]']     = Theta_hs_in_d_t
    df_output2['Theta_ex_d_t [℃]']          = climate.get_Theta_ex_d_t()
    df_output2['V_hs_supply_H_d_t [m3/h]']  = V_hs_supply_d_t
    df_output2['V_hs_vent_H_d_t [m3/h]']    = V_hs_vent_d_t
    df_output2['C_df_H_d_t [-]']            = climate.get_C_df_H_d_t()

    ##### 冷房消費電力の計算（kWh/h）
    print("冷房消費電力の計算")

    V_hs_dsgn_C: float = annual_ground_feedback_context.V_hs_dsgn_C
    """ 冷房時の送風機の設計風量 [m3/h] """

    V_hs_dsgn_H: float = 0.0  # NOTE: 冷房負荷計算時は空
    """ 暖房時の送風機の設計風量 [m3/h] """

    # 型バインド
    assert isinstance(V_hs_dsgn_H, float), "V_hs_dsgn_Hの型が不正"
    injector.binder.bind(jjj_dc.VHS_DSGN_H, to=V_hs_dsgn_H)
    assert isinstance(V_hs_dsgn_C, float), "V_hs_dsgn_Cの型が不正"
    injector.binder.bind(jjj_dc.VHS_DSGN_C, to=V_hs_dsgn_C)

    injector.binder.bind(jjj_dc.ActiveAcSetting, to=cool_ac_setting)  # 冷房負荷アクティブ

    if cool_ac_setting.type == 計算モデル.電中研モデル:
        R2, R1, R0, P_rac_fan_rtd_C = jjjexperiment.denchu.denchu_1.calc_R_and_Pc_C(cool_denchu_catalog)
        P_rac_fan_rtd_C = 1000 * P_rac_fan_rtd_C  # kW -> W
        simu_R_C = jjjexperiment.denchu.denchu_2.simu_R(R2, R1, R0)

        """ 電柱研モデルのモデリング定数の確認のためのCSV出力 """
        df_denchu_consts = jjjexperiment.denchu.denchu_1 \
            .get_DataFrame_denchu_modeling_consts(cool_denchu_catalog, R2, R1, R0, cool_real_inner, P_rac_fan_rtd_C)
        df_denchu_consts.to_csv(case_name + jjj_consts.version_info() + '_denchu_consts_C_output.csv', encoding='cp932')
        del R2, R1, R0
    else:
        P_rac_fan_rtd_C: float = V_hs_dsgn_C * cool_ac_setting.f_SFP
    """定格冷房能力運転時の送風機の消費電力(W)"""
    _logger.info(f"P_rac_fan_rtd_C [W]: {P_rac_fan_rtd_C}")

    E_UT_C_d_t: np.ndarray  # NOTE: ベースプログラムでは E_UT_C, E_C_UT が統一されていない
    """冷房設備の未処理冷房負荷の設計一次エネルギー消費量相当値(MJ/h)"""
    E_UT_C_d_t, Theta_hs_out_d_t, Theta_hs_in_d_t, X_hs_out_d_t, X_hs_in_d_t, V_hs_supply_d_t, V_hs_vent_d_t = \
        injector.call_with_injection(jjj_dc.calc_Q_UT_A)
    _logger.NDdebug("V_hs_supply_d_t", V_hs_supply_d_t)
    _logger.NDdebug("V_hs_vent_d_t", V_hs_vent_d_t)

    # (4) 日付dの時刻tにおける1時間当たりの熱源機の平均冷房能力(-)
    q_hs_CS_d_t, q_hs_CL_d_t = dc_a.get_q_hs_C_d_t(Theta_hs_out_d_t, Theta_hs_in_d_t, X_hs_out_d_t, X_hs_in_d_t, V_hs_supply_d_t, house.region)
    q_hs_C_d_t = q_hs_CS_d_t + q_hs_CL_d_t

    if cool_ac_setting.type == 計算モデル.RAC活用型全館空調_潜熱評価モデル:
        print(cool_ac_setting.type)

        import jjjexperiment.latent_load.section4_2_a as jjj_latent_dc_a
        E_E_fan_C_d_t = jjj_latent_dc_a.get_E_E_fan_C_d_t(V_hs_vent_d_t, q_hs_C_d_t, cool_ac_setting.f_SFP)

    elif cool_ac_setting.type in [
        計算モデル.ダクト式セントラル空調機,
        計算モデル.RAC活用型全館空調_現行省エネ法RACモデル,
        計算モデル.電中研モデル
    ]:
        print(cool_ac_setting.type)

        # [F25-01] 最低風量・最低電力 直接入力
        match v_min_cooling_input.input_V_hs_min:
            case 最低風量直接入力.入力しない:
                print(最低風量直接入力.入力しない)

                # 従来式
                E_E_fan_C_d_t = \
                    dc_a.get_E_E_fan_C_d_t(
                        # ルームエアコンファン(P_rac_fan) OR 循環ファン(P_fan)
                        P_rac_fan_rtd_C if cool_ac_setting.type == 計算モデル.RAC活用型全館空調_現行省エネ法RACモデル else cool_quantity.P_fan_rtd
                        , V_hs_vent_d_t  # 上書きナシ
                        , V_hs_supply_d_t
                        , V_hs_dsgn_C
                        , q_hs_C_d_t  # W
                        , house.region
                        , cool_ac_setting.f_SFP  # NOTE: 従来式は標準値固定だがカスタム値を反映
                        , cool_ac_setting.subtract_ventilation_power)

            case 最低風量直接入力.入力する:
                print(最低風量直接入力.入力する)

                match v_min_cooling_input.input_E_E_fan_min:
                    case 最低電力直接入力.入力しない:
                        print(最低電力直接入力.入力しない)

                        # 従来式
                        E_E_fan_C_d_t = \
                            dc_a.get_E_E_fan_C_d_t(
                                # ルームエアコンファン(P_rac_fan) OR 循環ファン(P_fan)
                                P_rac_fan_rtd_C if cool_ac_setting.type == 計算モデル.RAC活用型全館空調_現行省エネ法RACモデル else cool_quantity.P_fan_rtd
                                , V_hs_vent_d_t  # 上書きアリ
                                , V_hs_supply_d_t
                                , V_hs_dsgn_C
                                , q_hs_C_d_t  # W
                                , house.region
                                , cool_ac_setting.f_SFP  # NOTE: 従来式は標準値固定だがカスタム値を反映
                                , ファン消費電力から換気分を引く.換気分を引かない)  # 最低風量を入力する場合、換気分は引かない

                    case 最低電力直接入力.入力する:
                        print(最低電力直接入力.入力する)

                        E_E_fan_min_C = v_min_cooling_input.E_E_fan_min
                        E_E_fan_logic = v_min_cooling_input.E_E_fan_logic

                        from jjjexperiment.v_min_input.section4_2_a import get_E_E_fan_d_t
                        E_E_fan_C_d_t = get_E_E_fan_d_t(
                                E_E_fan_logic
                                # ルームエアコンファン(P_rac_fan) OR 循環ファン(P_fan)
                                , P_rac_fan_rtd_C if cool_ac_setting.type == 計算モデル.RAC活用型全館空調_現行省エネ法RACモデル else cool_quantity.P_fan_rtd
                                , V_hs_vent_d_t  # 上書きアリ
                                , V_hs_supply_d_t
                                , V_hs_dsgn_C
                                , E_E_fan_min_C
                                , house.region
                                , for_cooling=True)
                    case _:
                        raise ValueError
            case _:
                raise ValueError
    else:
        raise ValueError

    E_E_C_d_t: np.ndarray
    """日付dの時刻tにおける1時間当たりの冷房時の消費電力量(kWh/h)"""

    if cool_ac_setting.type in [
        計算モデル.ダクト式セントラル空調機,
        計算モデル.RAC活用型全館空調_潜熱評価モデル
    ]:
        E_E_C_d_t = jjj_dc_a.calc_E_E_C_d_t_type1_and_type3(
            type = cool_ac_setting.type,
            region = house.region,
            E_E_fan_C_d_t = E_E_fan_C_d_t,
            Theta_hs_out_d_t = Theta_hs_out_d_t,
            Theta_hs_in_d_t = Theta_hs_in_d_t,
            Theta_ex_d_t = climate.get_Theta_ex_d_t(),
            V_hs_supply_d_t = V_hs_supply_d_t,
            X_hs_out_d_t = X_hs_out_d_t,
            X_hs_in_d_t = X_hs_in_d_t,

            equipment_spec = cool_ac_setting.equipment_spec,
            q_hs_min_C =  cool_quantity.q_hs_min,
            q_hs_rtd_C =  cool_quantity.q_hs_rtd,
            q_hs_mid_C =  cool_quantity.q_hs_mid,
            P_hs_rtd_C =  cool_quantity.P_hs_rtd,
            P_hs_mid_C =  cool_quantity.P_hs_mid,
            V_fan_rtd_C = cool_quantity.V_fan_rtd,
            V_fan_mid_C = cool_quantity.V_fan_mid,
            P_fan_rtd_C = cool_quantity.P_fan_rtd,
            P_fan_mid_C = cool_quantity.P_fan_mid
        )
    elif cool_ac_setting.type == 計算モデル.RAC活用型全館空調_現行省エネ法RACモデル:
        E_E_C_d_t = jjj_dc_a.calc_E_E_C_d_t_type2(
            type = cool_ac_setting.type,
            region = house.region,
            climateFile = climateFile,
            E_E_fan_C_d_t = E_E_fan_C_d_t,
            q_hs_CS_d_t = q_hs_CS_d_t,
            q_hs_CL_d_t = q_hs_CL_d_t,
            e_rtd_C = cool_CRAC.e_rtd,
            q_rtd_C = cool_CRAC.q_rtd,
            q_max_C = cool_CRAC.q_max,
            input_C_af_C = cool_CRAC.input_C_af,
            dualcompressor_C = cool_CRAC.dualcompressor
        )
    elif cool_ac_setting.type == 計算モデル.電中研モデル:
        E_E_C_d_t = jjj_dc_a.calc_E_E_C_d_t_type4(
            case_name = case_name,
            type = cool_ac_setting.type,
            region = house.region,
            climateFile = climateFile,
            E_E_fan_C_d_t = E_E_fan_C_d_t,
            q_hs_C_d_t = q_hs_CS_d_t + q_hs_CL_d_t,
            V_hs_supply_d_t = V_hs_supply_d_t,
            P_rac_fan_rtd_C = P_rac_fan_rtd_C,
            simu_R_C = simu_R_C,
            spec = cool_denchu_catalog,
            real_inner = cool_real_inner
        )
    else:
        raise Exception("冷房方式が不正です。")

    ##### 計算結果のまとめ

    f_prim: float       = get_f_prim()                              # 電気の量 1kWh を熱量に換算する係数(kJ/kWh)
    E_H_d_t: np.ndarray = E_E_H_d_t * f_prim / 1000 + E_UT_H_d_t    #1 時間当たりの暖房設備の設計一次エネルギー消費量(MJ/h)
    E_C_d_t: np.ndarray = E_E_C_d_t * f_prim / 1000 + E_UT_C_d_t    #1 時間当たりの冷房設備の設計一次エネルギー消費量(MJ/h)
    E_H                 = np.sum(E_H_d_t)                           #1 年当たりの暖房設備の設計一次エネルギー消費量(MJ/年)
    E_C                 = np.sum(E_C_d_t)                           #1 年当たりの冷房設備の設計一次エネルギー消費量(MJ/年)

    _logger.info(f"E_H [MJ/year]: {E_H}")
    _logger.info(f"E_C [MJ/year]: {E_C}")

    print('E_H [MJ/year]: ', E_H, ', E_C [MJ/year]: ', E_C)

    df_output1 = pd.DataFrame(index = ['合計値'])
    df_output1['E_H [MJ/year]'] = E_H
    df_output1['E_C [MJ/year]'] = E_C
    df_output1.to_csv(case_name + jjj_consts.version_info() + '_output1.csv', encoding = 'cp932')

    df_output2['Theta_hs_C_out_d_t [℃]']    = Theta_hs_out_d_t
    df_output2['Theta_hs_C_in_d_t [℃]']     = Theta_hs_in_d_t
    df_output2['Theta_ex_d_t [℃]']          = climate.get_Theta_ex_d_t()
    df_output2['V_hs_supply_C_d_t [m3/h]']  = V_hs_supply_d_t
    df_output2['V_hs_vent_C_d_t [m3/h]']    = V_hs_vent_d_t
    df_output2['E_H_d_t [MJ/h]']            = E_H_d_t
    df_output2['E_C_d_t [MJ/h]']            = E_C_d_t
    df_output2['E_E_H_d_t [kWh/h]']         = E_E_H_d_t
    df_output2['E_E_C_d_t [kWh/h]']         = E_E_C_d_t
    df_output2['E_UT_H_d_t [MJ/h]']         = E_UT_H_d_t
    df_output2['E_UT_C_d_t [MJ/h]']         = E_UT_C_d_t
    df_output2['L_H_d_t [MJ/h]']            = L_H_d_t
    df_output2['L_CS_d_t [MJ/h]']           = L_CS_d_t
    df_output2['L_CL_d_t [MJ/h]']           = L_CL_d_t
    df_output2['E_E_fan_H_d_t [kWh/h]']     = E_E_fan_H_d_t
    df_output2['E_E_fan_C_d_t [kWh/h]']     = E_E_fan_C_d_t
    df_output2['q_hs_H_d_t [Wh/h]']         = q_hs_H_d_t
    df_output2['q_hs_CS_d_t [Wh/h]']        = q_hs_CS_d_t
    df_output2['q_hs_CL_d_t [Wh/h]']        = q_hs_CL_d_t
    df_output2.to_csv(case_name + jjj_consts.version_info() + '_output2.csv', encoding = 'cp932')

    # NOTE: 結合テストで確認したい値を返すのに使用します
    if test_mode:
        i = SutValues(cool_CRAC.q_rtd, heat_CRAC.q_rtd, cool_CRAC.q_max, heat_CRAC.q_max, cool_CRAC.e_rtd, heat_CRAC.e_rtd)
        r = ResultSummary(E_C, E_H)
        # NOTE: 今後の拡張を想定して既存コードが壊れにくい辞書型にしています
        return {'TInput':i, 'TValue':r}
