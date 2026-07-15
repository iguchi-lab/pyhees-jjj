import pytest
import json
import copy

from jjjexperiment.main import calc
from jjjexperiment.inputs.options import *

from test_utils.utils import INPUT_SAMPLE_TYPE3_PATH, deep_update

class Test潜熱評価結合:
    """潜熱評価 (latent_load) の結合テスト"""

    def _change_testmode_VAV(self, inputs: dict) -> dict:
        """VAVを有効"""
        fixtures = {
            "H_A": {"VAV": 2},
            "C_A": {"VAV": 2},
        }
        inputs_copied = copy.deepcopy(inputs)
        return deep_update(inputs_copied, fixtures)

    def _change_testmode_input_V_hs_min_H(self, inputs: dict) -> dict:
        """最低風量直接入力 - 暖房"""
        fixtures = {
            "H_A": {
                "input_V_hs_min_H": 最低風量直接入力.入力する.value,
                "V_hs_min_H": 1200,
            }
        }
        inputs_copied = copy.deepcopy(inputs)
        return deep_update(inputs_copied, fixtures)

    def _change_testmode_input_V_hs_min_C(self, inputs: dict) -> dict:
        """最低風量直接入力 - 冷房"""
        fixtures = {
            "C_A": {
                "input_V_hs_min_C": 最低風量直接入力.入力する.value,
                "V_hs_min_C": 1200,
            }
        }
        inputs_copied = copy.deepcopy(inputs)
        return deep_update(inputs_copied, fixtures)

    def _change_testmode_fan_coeff_H(self, inputs: dict) -> dict:
        """暖房機の風量設定変更"""
        fixtures = {"H_A": {"fan_coeff": [0, 0, 27, -8, 13]}}
        inputs_copied = copy.deepcopy(inputs)
        return deep_update(inputs_copied, fixtures)

    def _change_testmode_fan_coeff_C(self, inputs: dict) -> dict:
        """冷房機の風量設定変更"""
        fixtures = {"C_A": {"fan_coeff": [0, 0, 13, 20, -6]}}
        inputs_copied = copy.deepcopy(inputs)
        return deep_update(inputs_copied, fixtures)

    def _change_testmode_compressor_coeff_H(self, inputs: dict) -> dict:
        """暖房機の圧縮比設定変更"""
        fixtures = {"H_A": {"compressor_coeff": [0, 0.153, -0.53, 0.53, 0]}}
        inputs_copied = copy.deepcopy(inputs)
        return deep_update(inputs_copied, fixtures)

    def _change_testmode_compressor_coeff_C(self, inputs: dict) -> dict:
        """冷房機の圧縮比設定変更"""
        fixtures = {"C_A": {"compressor_coeff": [0, 0, -0.444, 0.444, 0]}}
        inputs_copied = copy.deepcopy(inputs)
        return deep_update(inputs_copied, fixtures)

    with open(INPUT_SAMPLE_TYPE3_PATH, 'r') as f:
        _inputs_type3: dict = json.load(f)
    _default_E_H = 42533.87
    _default_E_C = 19321.055317072976

    def test_基本計算_正常値(self):
        """Type3基本計算の統合テスト"""
        # Arrange
        inputs = self._inputs_type3.copy()
        # Act
        result = calc(inputs, test_mode=True)
        # Assert
        t_input = result['TInput']
        t_value = result['TValue']
        assert t_value.E_H == pytest.approx(self._default_E_H, abs=1e-1)
        assert t_value.E_C == pytest.approx(self._default_E_C, abs=1e-1)

    def test_最低風量直接入力_影響なし(self):
        """最低風量直接入力がType3に影響しないことの確認"""
        # Arrange
        inputs = self._inputs_type3.copy()
        # タイプ3への影響はないこと
        inputs = self._change_testmode_input_V_hs_min_C(inputs)
        inputs = self._change_testmode_input_V_hs_min_H(inputs)
        # Act
        result = calc(inputs, test_mode=True)
        # Assert
        t_input = result['TInput']
        t_value = result['TValue']
        assert t_value.E_H == pytest.approx(self._default_E_H, abs=1e-1)
        assert t_value.E_C == pytest.approx(self._default_E_C, abs=1e-1)

    def test_VAV設定_エネルギー変化(self):
        """VAV設定によるエネルギー計算変化の確認"""
        # Arrange
        inputs = self._inputs_type3.copy()
        inputs = self._change_testmode_VAV(inputs)
        # Act
        result = calc(inputs, test_mode=True)
        # Assert
        t_input = result['TInput']
        t_value = result['TValue']
        assert t_value.E_H == pytest.approx(42924.37, abs=1e-1)
        assert t_value.E_C == pytest.approx(19403.08321671903, abs=1e-1)

    def test_冷房ファン係数_変更効果(self):
        """冷房ファン係数変更による計算結果への影響"""
        # Arrange
        inputs = self._inputs_type3.copy()
        inputs = self._change_testmode_fan_coeff_C(inputs)
        # Act
        result = calc(inputs, test_mode=True)
        # Assert
        t_input = result['TInput']
        t_value = result['TValue']
        assert t_value.E_H == pytest.approx(self._default_E_H, abs=1e-1)
        assert t_value.E_C == pytest.approx(18381.92071356255, abs=1e-1)

    def test_暖房ファン係数_変更効果(self):
        """暖房ファン係数変更による計算結果への影響"""
        # Arrange
        inputs = self._inputs_type3.copy()
        inputs = self._change_testmode_fan_coeff_H(inputs)
        # Act
        result = calc(inputs, test_mode=True)
        # Assert
        t_input = result['TInput']
        t_value = result['TValue']
        assert t_value.E_H == pytest.approx(42385.77, abs=1e-1)
        assert t_value.E_C == pytest.approx(self._default_E_C, abs=1e-1)

    def test_冷房圧縮機係数_変更効果(self):
        """冷房圧縮機係数変更による計算結果への影響"""
        # Arrange
        inputs = self._inputs_type3.copy()
        inputs = self._change_testmode_compressor_coeff_C(inputs)
        # Act
        result = calc(inputs, test_mode=True)
        # Assert
        t_input = result['TInput']
        t_value = result['TValue']
        assert t_value.E_H == pytest.approx(self._default_E_H, abs=1e-1)
        assert t_value.E_C == pytest.approx(13946.124774900112, abs=1e-1)

    def test_暖房圧縮機係数_変更効果(self):
        """暖房圧縮機係数変更による計算結果への影響"""
        # Arrange
        inputs = self._inputs_type3.copy()
        inputs = self._change_testmode_compressor_coeff_H(inputs)
        # Act
        result = calc(inputs, test_mode=True)
        # Assert
        t_input = result['TInput']
        t_value = result['TValue']
        assert t_value.E_H == pytest.approx(71069.76, abs=1e-1)
        assert t_value.E_C == pytest.approx(self._default_E_C, abs=1e-1)
