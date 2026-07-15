import pytest
import json
import copy
import math
from os import path

from test_utils.utils import  \
    expected_inputs, expected_result_type1, INPUT_SAMPLE_TYPE1_PATH

# JJJ
from jjjexperiment.main import calc
from jjjexperiment.logger import LimitedLoggerAdapter as _logger
from jjjexperiment.inputs.options import *


class Test既存計算維持_入力値切替_方式1:
    """ 既存計算が壊れていないことのテスト
        暖房・冷房ともに「ダクト式セントラル空調機」
    """
    with open(INPUT_SAMPLE_TYPE1_PATH, 'r') as f:
        _inputs: dict = json.load(f)

    def test_前提確認_入力値(self, expected_inputs):
        """ テストコードが想定しているインプットデータかどうか確認
        """
        result = calc(self._inputs, test_mode=True)

        assert result['TInput'].q_rtd_C == pytest.approx(expected_inputs.q_rtd_C)
        assert result['TInput'].q_rtd_H == pytest.approx(expected_inputs.q_rtd_H)
        assert result['TInput'].q_max_C == pytest.approx(expected_inputs.q_max_C)
        assert result['TInput'].q_max_H == pytest.approx(expected_inputs.q_max_H)
        assert result['TInput'].e_rtd_C == pytest.approx(expected_inputs.e_rtd_C)
        assert result['TInput'].e_rtd_H == pytest.approx(expected_inputs.e_rtd_H)

    def test_前提確認_計算結果(self, expected_result_type1):
        """ ベースとしている結果が確かであることを確認
        """
        result = calc(self._inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)

    def test_入力値入替_01(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            最大暖房出力時の熱源機の出口における空気温度の最大値の上限値
            Theta_hs_out_max_H_d_t_limit
        """
        inputs = copy.deepcopy(self._inputs)
        # NOTE: 上限域の設定なので下げて有効性をテストする
        inputs["Theta_hs_out_max_H_d_t_limit"] = 47.5

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(36577.182142974685)

    def test_入力値入替_02(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            最大冷房出力時の熱源機の出口における空気温度の最低値の下限値
            Theta_hs_out_min_C_d_t_limit
        """
        inputs = copy.deepcopy(self._inputs)
        # NOTE: 下限域の設定なので上げて有効性をテストする
        inputs["Theta_hs_out_min_C_d_t_limit"] = 80.0

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C != expected_result_type1.E_C
        assert result['TValue'].E_C == pytest.approx(94898.37812202198)

    def test_入力値入替_03(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            デフロストに関する暖房出力補正係数（ダクトセントラル空調機）
            C_df_H_d_t_defrost_ductcentral
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_df_H_d_t_defrost_ductcentral"] = 0.88

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(36136.30658681903)

    def test_入力値入替_04(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            デフロスト発生外気温度（ダクトセントラル空調機）
            defrost_temp_ductcentral
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["defrost_temp_ductcentral"] = 6.2

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(36795.17930292908)

    def test_入力値入替_05(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            デフロスト発生外気相対湿度（ダクトセントラル空調機）
            defrost_humid_ductcentral
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["defrost_humid_ductcentral"] = 77

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(36863.72831878159)

    def test_入力値入替_06(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            ダクトiの線熱損失係数
            phi_i
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["phi_i"] = 0.52

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(36897.60513882977)
        assert result['TValue'].E_C != expected_result_type1.E_C
        assert result['TValue'].E_C == pytest.approx(18141.35460878325)

    def test_入力値入替_07(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            暖房時の送風機の設計風量に関する係数
            C_V_fan_dsgn_H
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_V_fan_dsgn_H"] = 0.82

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(36088.188174138966)

    def test_入力値入替_08(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            冷房時の送風機の設計風量に関する係数
            C_V_fan_dsgn_C
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_V_fan_dsgn_C"] = 0.81

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C == pytest.approx(17811.137390839904)
        assert result['TValue'].E_C != expected_result_type1.E_C

    def test_入力値入替_09(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            ルームエアコン計算用の定数
            C_df_H_d_t_defrost_rac: デフロストに関する暖房出力補正係数
            defrost_temp_rac: デフロスト発生外気温度
            defrost_humid_rac: デフロスト発生外気相対湿度
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_df_H_d_t_defrost_rac"] = 0.88  # 0.77
        # NOTE: 下限域の設定なので上げて有効性をテストする
        inputs["defrost_temp_rac"] = 8.0  # 5.0
        # NOTE: 上限域の設定なので下げて有効性をテストする
        inputs["defrost_humid_rac"] = 60  # 80

        result = calc(inputs, test_mode=True)

        # NOTE: 上記 はダクトセントラルの計算には使用されていない
        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)

    def test_入力値入替_10(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            室内機吸い込み湿度に関する冷房出力補正係数
            C_hm_C
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_hm_C"] = 1.32

        result = calc(inputs, test_mode=True)

        # NOTE: C_hm_C はダクトセントラルの計算には使用されていない
        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)

    def test_入力値入替_11(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            定格冷房能力の最大値
            q_rtd_C_limit
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["q_rtd_C_limit"] = 3500

        result = calc(inputs, test_mode=True)

        # NOTE: q_rtd_C_limit はダクトセントラルの計算には使用されていない
        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)

    def test_入力値入替_12(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            面積の合計 [m2]
            A_A
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["A_A"] = 135.92

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(40678.98334704396)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_C == pytest.approx(18516.873982041427)
        assert result['TValue'].E_C != expected_result_type1.E_C

    def test_入力値入替_13(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            主たる居室の面積 [m2]
            A_MR
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["A_MR"] = 68.29

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(42405.99765769754)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_C == pytest.approx(22791.634076123664)
        assert result['TValue'].E_C != expected_result_type1.E_C

    def test_入力値入替_14(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            その他の居室の面積[m2]
            A_OR
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["A_OR"] = 123.456

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(49109.975894658266)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_C == pytest.approx(24974.279701328636)
        assert result['TValue'].E_C != expected_result_type1.E_C

    def test_入力値入替_15(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            地域区分
            region
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["region"] = 1

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(142538.7729598713)
        assert result['TValue'].E_C != expected_result_type1.E_C
        assert result['TValue'].E_C == pytest.approx(3665.89162743432)

    def test_入力値入替_16(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            外皮面積 [m2]
            A_env
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["A_env"] = 333.332

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(42178.032569146)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_C == pytest.approx(18259.684719593355)
        assert result['TValue'].E_C != expected_result_type1.E_C

    def test_入力値入替_17(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            外皮平均熱貫流率 UA [W/(m2・K)]
            U_A
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["U_A"] = 1.03

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(45391.65533397831)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_C == pytest.approx(16981.72640048941)
        assert result['TValue'].E_C != expected_result_type1.E_C

    def test_入力値入替_18(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            冷房期平均日射熱取得率ηAC
            eta_A_C
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["eta_A_C"] = 3.3

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C != expected_result_type1.E_C
        assert result['TValue'].E_C == pytest.approx(19875.442331444658)

    def test_入力値入替_19(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            暖房期平均日射熱取得率ηAH
            eta_A_H
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["eta_A_H"] = 3.9

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)
        assert result['TValue'].E_H == pytest.approx(40537.790968649)
        assert result['TValue'].E_H != expected_result_type1.E_H

    def test_入力値入替_20(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            床下空間を経由して外気を導入する換気方式の利用 （☐：評価しない or ☑：評価する）
            underfloor_ventilation
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["underfloor_ventilation"] = "2"

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(34284.81604847377)
        assert result['TValue'].E_C != expected_result_type1.E_C
        assert result['TValue'].E_C == pytest.approx(15307.821253970653)

    def test_入力値入替_21(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            外気が経由する床下の面積の割合 [%]
            r_A_ufvnt
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["r_A_ufvnt"] = 85.0

        result = calc(inputs, test_mode=True)

        # NOTE: r_A_ufvnt はダクトセントラルの計算には使用されていない
        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)

    def test_入力値入替_22(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            床下空間の断熱 （☐：断熱区画外 or ☑：断熱区画内）
            underfloor_insulation
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["underfloor_insulation"] = "2"

        result = calc(inputs, test_mode=True)

        # NOTE: underfloor_insulation はダクトセントラルの計算には使用されていない
        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)

    def test_入力値入替_23(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            空調空気を床下を通して給気する （☐：床下を通して給気しない or ☑：床下を通して給気する）
            underfloor_air_conditioning_air_supply
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["underfloor_air_conditioning_air_supply"] = "2"
        # NOTE: この設定のみで 他の床下関係インプットも強制されます

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(59055.42895310527)
        assert result['TValue'].E_C != expected_result_type1.E_C
        assert result['TValue'].E_C == pytest.approx(22646.390406769133)

    def test_入力値入替_24(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            地盤またはそれを覆う基礎の表面熱伝達抵抗 [(m2・K)/W]
            R_g
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["R_g"] = 0.19

        result = calc(inputs, test_mode=True)

        # NOTE: R_g はダクトセントラルの計算には使用されていない
        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)

    # NOTE: hs_CAV はオリジナルではない概念なのでテスト対象外

    def test_入力値入替_H1(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            ダクトが通過する空間 （全てもしくは一部が断熱区画外である or 全て断熱区画内である）
            H_A_duct_insulation
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["H_A"]["duct_insulation"] = 2

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(34946.68098810251)

    def test_入力値入替_R1(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            ダクトが通過する空間（全てもしくは一部が断熱区画外である or 全て断熱区画内である）
            C_A_duct_insulation
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_A"]["duct_insulation"] = 2

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C != expected_result_type1.E_C
        assert result['TValue'].E_C == pytest.approx(17295.45258203477)

    def test_入力値入替_H2(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            VAV方式 （採用しない or 採用する）
            H_A_VAV
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["H_A"]["VAV"] = 2

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(36778.616118057646)

    def test_入力値入替_R2(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            VAV方式 (採用しない or 採用する）
            C_A_VAV
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_A"]["VAV"] = 2

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C != expected_result_type1.E_C
        assert result['TValue'].E_C == pytest.approx(18239.982369748006)

    def test_入力値入替_H3(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            全般換気機能 （あり or なし）
            H_A_general_ventilation
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["H_A"]["general_ventilation"] = 全般換気機能.なし.value  # デフォルトはあり

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(37318.00958773161)

    def test_入力値入替_R3(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            全般換気機能（あり or なし）
            C_A_general_ventilation
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_A"]["general_ventilation"] = 全般換気機能.なし.value  # デフォルトはあり

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C != expected_result_type1.E_C
        assert result['TValue'].E_C == pytest.approx(18522.484263163187)

    def test_入力値入替_H4(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            設計風量 [m3/h]（入力する場合のみ）
            H_A_V_hs_dsgn_H
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["H_A"]["input_V_hs_dsgn"] = 2
        inputs["H_A"]["V_hs_dsgn"] = 1820

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(32884.75839280709)

    def test_入力値入替_R4(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            設計風量 [m3/h]（入力する場合のみ）
            C_A_V_hs_dsgn_C
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_A"]["input_V_hs_dsgn"] = 2
        inputs["C_A"]["V_hs_dsgn"] = 1650

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C != expected_result_type1.E_C
        assert result['TValue'].E_C == pytest.approx(16057.762396533733)

    def test_入力値入替_H5_方式1(self, expected_inputs, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            暖房の機器仕様を入力する1
        """
        inputs = copy.deepcopy(self._inputs)
        # 機器仕様の入力（入力しない or 定格能力試験の値を入力する or 定格能力試験と中間能力試験の値を入力する）
        inputs["H_A"]["input"] = 2  # ★ 定格能力試験の値を入力する
        # ▼ 入力する場合のみ
        inputs["H_A"]["q_hs_rtd"] = 30000.0     # 定格暖房能力試験 能力 [W]
        inputs["H_A"]["q_hs_mid"] = 15000.0     # 中間暖房能力試験 能力 [W]
        inputs["H_A"]["P_hs_rtd"] = 8570        # 定格暖房能力試験 消費電力 [W]
        inputs["H_A"]["P_hs_mid"] = 4300        # 中間暖房能力試験 消費電力 [W]
        inputs["H_A"]["V_fan_rtd"] = 50.0 * 60  # 定格暖房能力試験 風量 [m3/h]
        inputs["H_A"]["V_fan_mid"] = 25.0 * 60  # 中間暖房能力試験 風量 [m3/h]
        inputs["H_A"]["P_fan_rtd"] = 350        # 定格暖房能力試験 室内側送風機の消費電力 [W]
        inputs["H_A"]["P_fan_mid"] = 170        # 中間暖房能力試験 室内側送風機の消費電力 [W]

        result = calc(inputs, test_mode=True)

        assert result['TInput'].q_rtd_C == pytest.approx(expected_inputs.q_rtd_C)
        assert result['TInput'].q_rtd_H == pytest.approx(expected_inputs.q_rtd_H)
        assert result['TInput'].q_max_C == pytest.approx(expected_inputs.q_max_C)
        assert result['TInput'].q_max_H == pytest.approx(expected_inputs.q_max_H)
        assert result['TInput'].e_rtd_C == pytest.approx(expected_inputs.e_rtd_C)
        assert result['TInput'].e_rtd_H == pytest.approx(expected_inputs.e_rtd_H)

        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)
        assert result['TValue'].E_H == pytest.approx(82027.87069522333)
        assert result['TValue'].E_H != expected_result_type1.E_H

    def test_入力値入替_R5_方式1(self, expected_inputs, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            冷房の機器仕様を入力する1
        """
        inputs = copy.deepcopy(self._inputs)
        # 機器仕様の入力（入力しない or 定格能力試験の値を入力する or 定格能力試験と中間能力試験の値を入力する）
        inputs["C_A"]["input"] = 2
        # ▼ 入力する場合のみ
        inputs["C_A"]["q_hs_rtd"] = 30000.0     # 定格冷房能力試験 能力 [W]
        inputs["C_A"]["q_hs_mid"] = 15000.0     # 中間冷房能力試験 能力 [W]
        inputs["C_A"]["P_hs_rtd"] = 8570        # 定格冷房能力試験 消費電力 [W]
        inputs["C_A"]["P_hs_mid"] = 4300        # 中間冷房能力試験 消費電力 [W]
        inputs["C_A"]["V_fan_rtd"] = 50.0 * 60  # 定格冷房能力試験 風量 [m3/h]
        inputs["C_A"]["V_fan_mid"] = 25.0 * 60  # 中間冷房能力試験 風量 [m3/h]
        inputs["C_A"]["P_fan_rtd"] = 350        # 定格冷房能力試験 室内側送風機の消費電力 [W]
        inputs["C_A"]["P_fan_mid"] = 170        # 中間冷房能力試験 室内側送風機の消費電力 [W]

        result = calc(inputs, test_mode=True)

        assert result['TInput'].q_rtd_C == pytest.approx(expected_inputs.q_rtd_C)
        assert result['TInput'].q_rtd_H == pytest.approx(expected_inputs.q_rtd_H)
        assert result['TInput'].q_max_C == pytest.approx(expected_inputs.q_max_C)
        assert result['TInput'].q_max_H == pytest.approx(expected_inputs.q_max_H)
        assert result['TInput'].e_rtd_C == pytest.approx(expected_inputs.e_rtd_C)
        assert result['TInput'].e_rtd_H == pytest.approx(expected_inputs.e_rtd_H)

        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C != expected_result_type1.E_C
        assert result['TValue'].E_C == pytest.approx(30012.123331372208)

    def test_入力値入替_H6_方式1(self, expected_inputs, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            暖房の機器仕様を入力する2
        """
        inputs = copy.deepcopy(self._inputs)
        # 機器仕様の入力（入力しない or 定格能力試験の値を入力する or 定格能力試験と中間能力試験の値を入力する）
        inputs["H_A"]["input"] = 2
        # ▼ 入力する場合のみ
        inputs["H_A"]["q_hs_rtd"] = 27500.0     # 定格暖房能力試験 能力 [W]
        inputs["H_A"]["q_hs_mid"] = 13000.0     # 中間暖房能力試験 能力 [W]
        inputs["H_A"]["P_hs_rtd"] = 7100        # 定格暖房能力試験 消費電力 [W]
        inputs["H_A"]["P_hs_mid"] = 3450        # 中間暖房能力試験 消費電力 [W]
        inputs["H_A"]["V_fan_rtd"] = 54.0 * 60  # 定格暖房能力試験 風量 [m3/h]
        inputs["H_A"]["V_fan_mid"] = 22.0 * 60  # 中間暖房能力試験 風量 [m3/h]
        inputs["H_A"]["P_fan_rtd"] = 400        # 定格暖房能力試験 室内側送風機の消費電力 [W]
        inputs["H_A"]["P_fan_mid"] = 210        # 中間暖房能力試験 室内側送風機の消費電力 [W]

        result = calc(inputs, test_mode=True)

        assert result['TInput'].q_rtd_C == pytest.approx(expected_inputs.q_rtd_C)
        assert result['TInput'].q_rtd_H == pytest.approx(expected_inputs.q_rtd_H)
        assert result['TInput'].q_max_C == pytest.approx(expected_inputs.q_max_C)
        assert result['TInput'].q_max_H == pytest.approx(expected_inputs.q_max_H)
        assert result['TInput'].e_rtd_C == pytest.approx(expected_inputs.e_rtd_C)
        assert result['TInput'].e_rtd_H == pytest.approx(expected_inputs.e_rtd_H)

        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(73523.57117653707)

    def test_入力値入替_R6_方式1(self, expected_inputs, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            冷房の機器仕様を入力する2
        """
        # 機器仕様の入力（入力しない or 定格能力試験の値を入力する or 定格能力試験と中間能力試験の値を入力する）
        inputs = copy.deepcopy(self._inputs)
        # ▼ 入力する場合のみ
        inputs["C_A"]["input"] = 2
        inputs["C_A"]["q_hs_rtd"] = 27500.0     # 定格冷房能力試験 能力 [W]
        inputs["C_A"]["q_hs_mid"] = 13000.0     # 中間冷房能力試験 能力 [W]
        inputs["C_A"]["P_hs_rtd"] = 7100        # 定格冷房能力試験 消費電力 [W]
        inputs["C_A"]["P_hs_mid"] = 3450        # 中間冷房能力試験 消費電力 [W]
        inputs["C_A"]["V_fan_rtd"] = 54.0 * 60  # 定格冷房能力試験 風量 [m3/h]
        inputs["C_A"]["V_fan_mid"] = 22.0 * 60  # 中間冷房能力試験 風量 [m3/h]
        inputs["C_A"]["P_fan_rtd"] = 400        # 定格冷房能力試験 室内側送風機の消費電力 [W]
        inputs["C_A"]["P_fan_mid"] = 210        # 中間冷房能力試験 室内側送風機の消費電力 [W]

        result = calc(inputs, test_mode=True)

        assert result['TInput'].q_rtd_C == pytest.approx(expected_inputs.q_rtd_C)
        assert result['TInput'].q_rtd_H == pytest.approx(expected_inputs.q_rtd_H)
        assert result['TInput'].q_max_C == pytest.approx(expected_inputs.q_max_C)
        assert result['TInput'].q_max_H == pytest.approx(expected_inputs.q_max_H)
        assert result['TInput'].e_rtd_C == pytest.approx(expected_inputs.e_rtd_C)
        assert result['TInput'].e_rtd_H == pytest.approx(expected_inputs.e_rtd_H)

        assert result['TValue'].E_H == pytest.approx(expected_result_type1.E_H)
        assert result['TValue'].E_C != expected_result_type1.E_C
        assert result['TValue'].E_C == pytest.approx(27565.70865332615)

    def test_入力値入替_HEX1(self, expected_result_type1):
        """ 以前のプログラムと同じ計算結果になる
            温度交換効率（設置する場合のみ）
            etr_t
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["HEX"]["install"] = 2
        inputs["HEX"]["etr_t"] = 0.3

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type1.E_C)
        assert result['TValue'].E_H != expected_result_type1.E_H
        assert result['TValue'].E_H == pytest.approx(34617.143039980554)
