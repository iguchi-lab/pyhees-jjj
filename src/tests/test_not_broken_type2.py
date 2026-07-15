import pytest
import json
import copy
import math
from os import path

from test_utils.utils import  \
    expected_inputs, expected_result_type2, INPUT_SAMPLE_TYPE2_PATH

# JJJ
from jjjexperiment.main import calc
from jjjexperiment.inputs.options import *
from jjjexperiment.logger import LimitedLoggerAdapter as _logger


class Test既存計算維持_入力値切替_方式2:
    """ 既存計算が壊れていないことのテスト
        暖房・冷房ともに「ルームエアコンディショナ活用型全館空調」
    """
    with open(INPUT_SAMPLE_TYPE2_PATH, 'r') as f:
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

    def test_前提確認_計算結果(self, expected_result_type2):
        """ ベースとしている結果が確かであることを確認
        """
        result = calc(self._inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type2.E_H)
        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)

    def test_入力値入替_01(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            最大暖房出力時の熱源機の出口における空気温度の最大値の上限値
            Theta_hs_out_max_H_d_t_limit
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["Theta_hs_out_max_H_d_t_limit"] = 47.5

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(40703.110799509814)

    def test_入力値入替_02(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            最大冷房出力時の熱源機の出口における空気温度の最低値の下限値
            Theta_hs_out_min_C_d_t_limit
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["Theta_hs_out_min_C_d_t_limit"] = 18.2

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type2.E_H)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(15626.045801452261)

    def test_入力値入替_03(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            デフロストに関する暖房出力補正係数（ダクトセントラル空調機）
            C_df_H_d_t_defrost_ductcentral
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_df_H_d_t_defrost_ductcentral"] = 0.88

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(40259.614225256155)

    def test_入力値入替_04(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            デフロスト発生外気温度（ダクトセントラル空調機）
            defrost_temp_ductcentral
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["defrost_temp_ductcentral"] = 6.2

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(41211.39467707037)

    def test_入力値入替_05(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            デフロスト発生外気相対湿度（ダクトセントラル空調機）
            defrost_humid_ductcentral
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["defrost_humid_ductcentral"] = 77

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(41446.61835562958)

    def test_入力値入替_06(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            ダクトiの線熱損失係数
            phi_i
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["phi_i"] = 0.52

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(16766.70866796063)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(41116.4326458173)

    def test_入力値入替_07(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            暖房時の送風機の設計風量に関する係数
            C_V_fan_dsgn_H
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_V_fan_dsgn_H"] = 0.82

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(40651.396812654966)

    def test_入力値入替_08(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            冷房時の送風機の設計風量に関する係数
            C_V_fan_dsgn_C
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_V_fan_dsgn_C"] = 0.81

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type2.E_H)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(16707.71285678587)

    def test_入力値入替_09(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            デフロストに関する暖房出力補正係数（ルームエアコンディショナー）
            C_df_H_d_t_defrost_rac
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_df_H_d_t_defrost_rac"] = 0.88

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(40108.03642034368)

    def test_入力値入替_10(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            デフロスト発生外気温度（ルームエアコンディショナー）
            defrost_temp_rac
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["defrost_temp_rac"] = 5.2

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(40806.580581208946)

    def test_入力値入替_11(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            デフロスト発生外気相対湿度（ルームエアコンディショナー）
            defrost_humid_rac
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["defrost_humid_rac"] = 82

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(40362.181064378994)

    def test_入力値入替_12(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            室内機吸い込み湿度に関する冷房出力補正係数
            C_hm_C
        """

        inputs = copy.deepcopy(self._inputs)
        inputs["C_hm_C"] = 1.32

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type2.E_H)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(14070.8919648636)

    def test_入力値入替_13(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            定格冷房能力の最大値
            q_rtd_C_limit
        """

        inputs = copy.deepcopy(self._inputs)
        inputs["q_rtd_C_limit"] = 3500

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(38861.62504332851)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(18350.86553605521)

    def test_入力値入替_14(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            面積の合計 [m2]
            A_A
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["A_A"] = 135.92

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(40487.48035409603)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(16966.161721981785)

    def test_入力値入替_15(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            主たる居室の面積 [m2]
            A_MR
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["A_MR"] = 68.29

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(47568.49376799733)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(21945.402304465548)

    def test_入力値入替_16(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            その他の居室の面積[m2]
            A_OR
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["A_OR"] = 123.456

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(58421.88338474051)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(24288.367155641543)

    def test_入力値入替_17(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            地域区分
            region
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["region"] = 1

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(174261.67585289382)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(3198.754622079984)

    def test_入力値入替_18(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            外皮面積 [m2]
            A_env
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["A_env"] = 333.332

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(45431.40860402107)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(17331.948563883856)

    def test_入力値入替_19(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            外皮平均熱貫流率 UA [W/(m2・K)]
            U_A
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["U_A"] = 1.03

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(50840.21334169296)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(16022.60488034099)

    def test_入力値入替_20(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            冷房期平均日射熱取得率ηAC
            eta_A_C
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["eta_A_C"] = 3.3

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type2.E_H)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(18843.43943002122)

    def test_入力値入替_21(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            暖房期平均日射熱取得率ηAH
            eta_A_H
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["eta_A_H"] = 3.9

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(42822.54947702187)

    def test_入力値入替_22(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            床下空間を経由して外気を導入する換気方式の利用 （☐：評価しない or ☑：評価する）
            underfloor_ventilation
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["underfloor_ventilation"] = "2"

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(38220.231632402436)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(14377.917552368737)

    def test_入力値入替_23(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            外気が経由する床下の面積の割合 [%]
            r_A_ufvnt
        """
        inputs = copy.deepcopy(self._inputs)
        # WARNING: 値変更しても変わらないけど大丈夫? 100 -> 85.0
        inputs["r_A_ufvnt"] = 85.0

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type2.E_H)
        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)

    def test_入力値入替_24(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            床下空間の断熱 （☐：断熱区画外 or ☑：断熱区画内）
            underfloor_insulation
        """
        inputs = copy.deepcopy(self._inputs)
        # WARNING: 値変更しても変わらないけど大丈夫? '1' -> '2'
        inputs["underfloor_insulation"] = '2'

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type2.E_H)
        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)

    def test_入力値入替_25(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            空調空気を床下を通して給気する （☐：床下を通して給気しない or ☑：床下を通して給気する）
            underfloor_air_conditioning_air_supply
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["underfloor_air_conditioning_air_supply"] = "2"
        # NOTE: この設定のみで 他の床下関係インプットも強制されます

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(66224.9882174671)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(20997.18980253943)

    def test_入力値入替_26(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            地盤またはそれを覆う基礎の表面熱伝達抵抗 [(m2・K)/W]
            R_g
        """
        inputs = copy.deepcopy(self._inputs)
        # WARNING: 値変更しても変わらないけど大丈夫? 0.15 -> 0.19
        inputs["R_g"] = 0.19

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type2.E_H)
        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)

    def test_入力値入替_27(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            全体風量を固定する （☐：固定しない or ☑：固定する）
            hs_CAV
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["hs_CAV"] = "2"

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(17693.364531830273)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(41489.49730383161)

    def test_入力値入替_H1(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            ダクトが通過する空間 （全てもしくは一部が断熱区画外である or 全て断熱区画内である）
            H_A_duct_insulation
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["H_A"]["duct_insulation"] = 2

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(38426.4135686332)

    def test_入力値入替_H2(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            VAV方式 （採用しない or 採用する）
            H_A_VAV
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["H_A"]["VAV"] = 2

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(39667.322879561434)

    def test_入力値入替_H3(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            全般換気機能 （あり or なし）
            H_A_general_ventilation
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["H_A"]["general_ventilation"] = 全般換気機能.なし.value  # デフォルトはあり

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(41348.5747910848)

    def test_入力値入替_H4(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            設計風量 [m3/h]（入力する場合のみ）
            H_A_V_hs_dsgn_H
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["H_A"]["input_V_hs_dsgn"] = 2
        inputs["H_A"]["V_hs_dsgn"] = 1820

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(40703.594298106414)

    def test_入力値入替_H5_方式2(self, expected_inputs, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            暖房の機器仕様を入力する1
        """
        inputs = copy.deepcopy(self._inputs)

        # 暖房能力の入力（面積から能力を算出 or 性能を直接入力）
        inputs["H_A"]["input_rac_performance"] = 1  # ★ 面積から能力を算出
        # ▼ 入力する場合
        inputs["H_A"]["q_rac_rtd_H"] = 1  # 暖房定格能力 [W]
        inputs["H_A"]["q_rac_max_H"] = 1  # 暖房最大能力 [W]
        inputs["H_A"]["e_rac_rtd_H"] = 1  # 定格エネルギー効率 [-]

        # 小能力時高効率型コンプレッサー（評価しない or 搭載する）
        inputs["H_A"]["dualcompressor"] = 2

        # 設置方法の入力（設置方法を入力する or 補正係数を直接入力する）
        inputs["H_A"]["input_C_af_H2"] = 1
        # ▼ 入力する場合
        inputs["H_A"]["C_af_H2"] = 1  # 室内機吹き出し風量に関する暖房出力補正係数の入力

        # 専用チャンバーに格納される方式（該当しない or 該当する）
        inputs["H_A"]["dedicated_chamber2"] = 2
        # フィン向きが中央位置に固定される方式（該当しない or 該当する）
        inputs["H_A"]["fixed_fin_direction2"] = 2

        # ファンの比消費電力（入力しない or 入力する）
        inputs["H_A"]["input_f_SFP"] = 1
        # ▼ 入力する場合
        inputs["H_A"]["f_SFP"] = 1  # ファンの比消費電力 [W/(m3/h)]

        result = calc(inputs, test_mode=True)

        assert result['TInput'].q_max_H == pytest.approx(expected_inputs.q_max_H)
        assert result['TInput'].q_max_C == pytest.approx(expected_inputs.q_max_C)
        assert result['TInput'].q_rtd_H == pytest.approx(expected_inputs.q_rtd_H)
        assert result['TInput'].q_rtd_C == pytest.approx(expected_inputs.q_rtd_C)
        assert result['TInput'].e_rtd_H == pytest.approx(expected_inputs.e_rtd_H)
        assert result['TInput'].e_rtd_C == pytest.approx(expected_inputs.e_rtd_C)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(31465.044727934437)

    def test_入力値入替_H6_方式2(self, expected_inputs, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            暖房の機器仕様を入力する2
        """
        inputs = copy.deepcopy(self._inputs)

        # 暖房能力の入力（面積から能力を算出 or 性能を直接入力）
        inputs["H_A"]["input_rac_performance"] = 2  # ★ 性能を直接入力
        # ▼ 入力する場合のみ
        inputs["H_A"]["q_rac_rtd_H"] = 3600  # 暖房定格能力 [W]
        inputs["H_A"]["q_rac_max_H"] = 4700  # 暖房最大能力 [W]
        inputs["H_A"]["e_rac_rtd_H"] = 3.93  # 定格エネルギー効率 [-]

        # 小能力時高効率型コンプレッサー（評価しない or 搭載する）
        inputs["H_A"]["dualcompressor"] = 1

        # 設置方法の入力（設置方法を入力する or 補正係数を直接入力する）
        inputs["H_A"]["input_C_af_H2"] = 2  # ★ 補正係数を直接入力する
        # ▼ 入力する場合のみ
        inputs["H_A"]["C_af_H2"] = 0.914  # 室内機吹き出し風量に関する暖房出力補正係数の入力

        # 専用チャンバーに格納される方式（該当しない or 該当する）
        inputs["H_A"]["dedicated_chamber2"] = 1
        # フィン向きが中央位置に固定される方式（該当しない or 該当する）
        inputs["H_A"]["fixed_fin_direction2"] = 1

        # ファンの比消費電力（入力しない or 入力する）
        inputs["H_A"]["input_f_SFP"] = 2  # ★ 入力する
        # ▼ 入力する場合のみ
        inputs["H_A"]["f_SFP"] = 0.3  # ファンの比消費電力 [W/(m3/h)]

        result = calc(inputs, test_mode=True)

        assert result['TInput'].q_max_C == pytest.approx(expected_inputs.q_max_C)
        assert result['TInput'].q_max_H != expected_inputs.q_max_H
        assert result['TInput'].q_max_H == pytest.approx(4700.0)

        assert result['TInput'].q_rtd_C == pytest.approx(expected_inputs.q_rtd_C)
        assert result['TInput'].q_rtd_H != expected_inputs.q_rtd_H
        assert result['TInput'].q_rtd_H == pytest.approx(3600.0)

        assert result['TInput'].e_rtd_C == pytest.approx(expected_inputs.e_rtd_C)
        assert result['TInput'].e_rtd_H != expected_inputs.e_rtd_H
        assert result['TInput'].e_rtd_H == pytest.approx(3.93)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(40211.253920843934)

    def test_入力値入替_R1(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            ダクトが通過する空間（全てもしくは一部が断熱区画外である or 全て断熱区画内である）
            C_A_duct_insulation
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_A"]["duct_insulation"] = 2

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type2.E_H)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(15793.053177912992)

    def test_入力値入替_R2(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            VAV方式 (採用しない or 採用する）
            C_A_VAV
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_A"]["VAV"] = 2

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type2.E_H)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(15286.440101728145)

    def test_入力値入替_R3(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            全般換気機能（あり or なし）
            C_A_general_ventilation
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_A"]["general_ventilation"] = 全般換気機能.なし.value  # デフォルトはあり

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type2.E_H)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(17094.103844100133)

    def test_入力値入替_R4(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            設計風量 [m3/h]（入力する場合のみ）
            C_A_V_hs_dsgn_C
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["C_A"]["input_V_hs_dsgn"] = 2
        inputs["C_A"]["V_hs_dsgn"] = 1650

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_H == pytest.approx(expected_result_type2.E_H)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(17148.579603861243)

    def test_入力値入替_R5_方式2(self, expected_result_type2, expected_inputs):
        """ 以前のプログラムと同じ計算結果になる
            冷房の機器仕様を入力する1
        """
        inputs = copy.deepcopy(self._inputs)

        # 冷房能力の入力（面積から能力を算出 or 性能を直接入力）
        inputs["C_A"]["input_rac_performance"] = 1  # ★ 面積から能力を算出
        # ▼ 入力する場合のみ
        inputs["C_A"]["q_rac_rtd_C"] = 1  # 冷房定格能力 [W]
        inputs["C_A"]["q_rac_max_C"] = 1  # 冷房最大能力 [W]
        inputs["C_A"]["e_rac_rtd_C"] = 1  # 定格エネルギー効率 [-]

        # 小能力時高効率型コンプレッサー（評価しない or 搭載する）
        inputs["C_A"]["dualcompressor"] = 2

        # 設置方法の入力（設置方法を入力する or 補正係数を直接入力する）
        inputs["C_A"]["input_C_af_C2"] = 1  # ★ 設置方法を入力する
        # ▼ 入力する場合のみ
        inputs["C_A"]["C_af_C2"] = 1  # 室内機吹き出し風量に関する暖房出力補正係数の入力

        # 専用チャンバーに格納される方式（該当しない or 該当する）
        inputs["C_A"]["dedicated_chamber2"] = 2
        # フィン向きが中央位置に固定される方式（該当しない or 該当する）
        inputs["C_A"]["fixed_fin_direction2"] = 2

        # ファンの比消費電力（入力しない or 入力する）
        inputs["C_A"]["input_f_SFP"] = 1  # ★ 入力しない
        # ▼ 入力する場合のみ
        inputs["C_A"]["f_SFP"] = 1  # ファンの比消費電力W [(m3/h)/W]

        result = calc(inputs, test_mode=True)

        assert result['TInput'].q_rtd_C == pytest.approx(expected_inputs.q_rtd_C)
        assert result['TInput'].q_rtd_H == pytest.approx(expected_inputs.q_rtd_H)
        assert result['TInput'].q_max_C == pytest.approx(expected_inputs.q_max_C)
        assert result['TInput'].q_max_H == pytest.approx(expected_inputs.q_max_H)
        assert result['TInput'].e_rtd_C == pytest.approx(expected_inputs.e_rtd_C)
        assert result['TInput'].e_rtd_H == pytest.approx(expected_inputs.e_rtd_H)
        
        assert result['TValue'].E_H == pytest.approx(expected_result_type2.E_H)
        assert result['TValue'].E_C != expected_result_type2.E_C
        assert result['TValue'].E_C == pytest.approx(15569.488273045597)

    def test_入力値入替_R6_方式2(self):
        """ 以前のプログラムと同じ計算結果になる
            冷房の機器仕様を入力する2
        """
        inputs = copy.deepcopy(self._inputs)

        # 冷房能力の入力（面積から能力を算出 or 性能を直接入力）
        inputs["C_A"]["input_rac_performance"] = 2  # ★ 面積から能力を算出
        # ▼ 入力する場合のみ
        inputs["C_A"]["q_rac_rtd_C"] = 2800  # 冷房定格能力 [W]
        inputs["C_A"]["q_rac_max_C"] = 3400  # 冷房最大能力 [W]
        inputs["C_A"]["e_rac_rtd_C"] = 2.59  # 定格エネルギー効率 [-]

        # 小能力時高効率型コンプレッサー（評価しない or 搭載する）
        inputs["C_A"]["dualcompressor"] = 1

        # 設置方法の入力（設置方法を入力する or 補正係数を直接入力する）
        inputs["C_A"]["input_C_af_C2"] = 2  # ★ 補正係数を直接入力する
        # ▼ 入力する場合のみ
        inputs["C_A"]["C_af_C2"] = 0.980  # 室内機吹き出し風量に関する冷房出力補正係数の入力

        # 専用チャンバーに格納される方式（該当しない or 該当する）
        inputs["C_A"]["dedicated_chamber2"] = 1
        # フィン向きが中央位置に固定される方式（該当しない or 該当する）
        inputs["C_A"]["fixed_fin_direction2"] = 1

        # ファンの比消費電力（入力しない or 入力する）
        inputs["C_A"]["input_f_SFP"] = 2  # ★ 入力する
        # ▼ 入力する場合のみ
        inputs["C_A"]["f_SFP"] = 0.3  # ファンの比消費電力W [(m3/h)/W]

        result = calc(inputs, test_mode=True)

        # TODO: 暖房についても結果が変わるのは大丈夫?
        assert result['TInput'].q_rtd_C == pytest.approx(2800.0)
        assert result['TInput'].q_rtd_H == pytest.approx(3300.1)
        assert result['TInput'].q_max_C == pytest.approx(3400.0)
        assert result['TInput'].q_max_H == pytest.approx(5569.28)
        assert result['TInput'].e_rtd_C == pytest.approx(2.59)
        assert result['TInput'].e_rtd_H == pytest.approx(3.6543)

        assert result['TValue'].E_C == pytest.approx(23871.2772537999)
        assert result['TValue'].E_H == pytest.approx(46788.506664833956)

    def test_入力値入替_HEX1(self, expected_result_type2):
        """ 以前のプログラムと同じ計算結果になる
            温度交換効率（設置する場合のみ）
            etr_t
        """
        inputs = copy.deepcopy(self._inputs)
        inputs["HEX"]["install"] = 2
        inputs["HEX"]["etr_t"] = 0.3

        result = calc(inputs, test_mode=True)

        assert result['TValue'].E_C == pytest.approx(expected_result_type2.E_C)
        assert result['TValue'].E_H != expected_result_type2.E_H
        assert result['TValue'].E_H == pytest.approx(37895.580913287784)
