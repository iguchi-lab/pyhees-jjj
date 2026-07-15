import os
import pytest
import numpy as np
# JJJ
from jjjexperiment.inputs.di_container import create_injector_from_json
from jjjexperiment.inputs.common import HouseInfo
from jjjexperiment.underfloor_ac.section3_1_e import get_Theta_uf_d_t_runup
from jjjexperiment.underfloor_ac.section4_2 import get_A_s_ufac_i, get_r_A_NR_uf_1F
from test_utils.utils import load_input_yaml

class Test_床下空調時_共通:

    def test_床下空調利用時の有効面積と比率(self):
        """
        床下空調利用時の有効面積と比率
        """
        # Arrange
        yaml_fullpath = os.path.join(os.path.dirname(__file__), 'test_input.yaml')
        injector = create_injector_from_json(load_input_yaml(yaml_fullpath))

        house = injector.get(HouseInfo)

        # Act
        A_s_ufac_i, r_A_s_ufac = get_A_s_ufac_i(house.A_A, house.A_MR, house.A_OR)

        # Assert
        assert np.shape(A_s_ufac_i) == (12, 1)
        assert r_A_s_ufac == pytest.approx(65.4/house.A_A, rel=1e-2)

    def test_非居室1F床下接触面積比(self):
        """
        get_r_A_NR_uf_1F が標準住戸における
        1F床下接触非居室面積 / 非居室合計面積 ≈ 0.489 を返すこと
        """
        # Act
        r = get_r_A_NR_uf_1F()

        # Assert
        # ゾーン6,7,8,9の有効面積 (3.31+1.66+3.31+10.76) / 非居室合計 (38.93)
        expected = (3.31 + 1.66 + 3.31 + 10.76) / 38.93
        assert r == pytest.approx(expected, rel=1e-4)

    def test_新床下空調助走計算用床下温度(self):
        """
        get_Theta_uf_d_t_runup_new_ufac が 8760 要素の定数配列を返し、
        季節区分ごとの温度が正しいこと
        """
        # Act
        arr = get_Theta_uf_d_t_runup()

        # Assert
        assert len(arr) == 24 * 365
        # 第1期間 [0, 2663]: 27.69 ℃
        assert arr[0]    == pytest.approx(27.69)
        assert arr[2663] == pytest.approx(27.69)
        # 第2期間 [2664, 6384]: 25.62 ℃
        assert arr[2664] == pytest.approx(25.62)
        assert arr[6384] == pytest.approx(25.62)
        # 第3期間 [6385, 8759]: 27.69 ℃
        assert arr[6385] == pytest.approx(27.69)
        assert arr[8759] == pytest.approx(27.69)


class Test_床下地盤熱損失助走計算:
    """calc_sum_Theta_dash_g_surf_A_m_runup のテスト

    暫定値（マジックナンバー）は 6地域の気候データを元に手計算されたとみられる。
    新関数は同一地域で計算した Theta_g_avg を使用するため、
    数値は近いが完全一致はしない。許容誤差（rel=0.03）を設けて検証する。
    """

    MAGIC_WARM = 11.2224  # IGUCHI 暖房期 27.69℃の暫定値
    MAGIC_COOL = 9.15940  # IGUCHI 冷房期 25.62℃の暫定値

    def test_暖房期_旧暫定値と近似一致(self):
        """THETA_UF_WARM で助走した結果が旧暫定値 11.2224 と 3% 以内に収まること"""
        from jjjexperiment.underfloor_ac.section3_1_e import (
            calc_sum_Theta_dash_g_surf_A_m_runup,
            THETA_UF_WARM,
        )
        from pyhees.section3_1_e import get_Theta_g_avg
        from pyhees.section11_1 import load_climate

        climate = load_climate(6)  # 旧暫定値の導出に使われたとみられる地域
        Theta_g_avg = get_Theta_g_avg(climate['外気温[℃]'].values)

        result = calc_sum_Theta_dash_g_surf_A_m_runup(THETA_UF_WARM, Theta_g_avg)

        assert result == pytest.approx(self.MAGIC_WARM, rel=0.03)

    def test_冷房期_旧暫定値と近似一致(self):
        """THETA_UF_COOL で助走した結果が旧暫定値 9.15940 と 3% 以内に収まること"""
        from jjjexperiment.underfloor_ac.section3_1_e import (
            calc_sum_Theta_dash_g_surf_A_m_runup,
            THETA_UF_COOL,
        )
        from pyhees.section3_1_e import get_Theta_g_avg
        from pyhees.section11_1 import load_climate

        climate = load_climate(6)
        Theta_g_avg = get_Theta_g_avg(climate['外気温[℃]'].values)

        result = calc_sum_Theta_dash_g_surf_A_m_runup(THETA_UF_COOL, Theta_g_avg)

        assert result == pytest.approx(self.MAGIC_COOL, rel=0.03)

    def test_暖房期は冷房期より大きい(self):
        """暖房期（高温）の助走結果は冷房期（低温）より大きくなること"""
        from jjjexperiment.underfloor_ac.section3_1_e import (
            calc_sum_Theta_dash_g_surf_A_m_runup,
            THETA_UF_WARM,
            THETA_UF_COOL,
        )

        result_warm = calc_sum_Theta_dash_g_surf_A_m_runup(THETA_UF_WARM, Theta_g_avg=15.0)
        result_cool = calc_sum_Theta_dash_g_surf_A_m_runup(THETA_UF_COOL, Theta_g_avg=15.0)

        assert result_warm > result_cool

    def test_高いTheta_g_avgは結果を小さくする(self):
        """Theta_g_avg が高いほど sum_Theta_dash_g_surf_A_m が小さくなること"""
        from jjjexperiment.underfloor_ac.section3_1_e import (
            calc_sum_Theta_dash_g_surf_A_m_runup,
            THETA_UF_WARM,
        )

        result_cold_region = calc_sum_Theta_dash_g_surf_A_m_runup(THETA_UF_WARM, Theta_g_avg=10.0)
        result_warm_region = calc_sum_Theta_dash_g_surf_A_m_runup(THETA_UF_WARM, Theta_g_avg=20.0)

        assert result_cold_region > result_warm_region
