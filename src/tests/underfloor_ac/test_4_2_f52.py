import pytest

from jjjexperiment.common import JJJ_HCM
from jjjexperiment.underfloor_ac.section4_2 import get_r_A_NR_uf_1F
from jjjexperiment.underfloor_ac.section4_2_f52 import get_Theta_star_NR

@pytest.mark.xfail(reason="260323_井口先生よりロジック修正中のため")
class Test式52負荷バランス時非居室室温:
    """式52 負荷バランス時の非居室の室温計算のテストクラス"""

    def test_暖房期_基本計算1(self):
        """基本計算のテスト - 正常値での式52計算"""
        # NOTE: ダクトセントラル・床下定数 変更なし(index=0)
        # Arrange & Act
        result = get_Theta_star_NR(
            Theta_star_HBR=20.0,    # 居室設定温度
            Q=2.65,                 # 熱損失係数
            A_NR=38.9,              # 非居室面積
            V_vent_l_NR=0.0,        # 非居室換気風量
            V_dash_supply_A=1306.0, # 全館供給風量
            U_prt=2.17,             # 部位熱貫流率
            A_prt_A=107.0,          # 部位面積合計
            L_H_NR_A=5.17,          # 非居室暖房負荷
            L_CS_NR_A=0.0,          # 非居室冷房負荷
            Theta_NR=20.0,          # 非居室温度
            Theta_uf=31.2,          # 床下空調温度
            HCM=JJJ_HCM.H,          # 暖冷房区画の熱容量
            r_A_NR_1F=get_r_A_NR_uf_1F()
        )
        # Assert
        assert result == pytest.approx(25.4, abs=1e-1)  # 25.4 -> 18.6

    def test_暖房期_基本計算2(self):
        """基本計算のテスト - 正常値での式52計算"""
        # NOTE: ダクトセントラル・床下定数 変更あり(index=0)
        # Arrange & Act
        result = get_Theta_star_NR(
            Theta_star_HBR=20.0,    # 居室設定温度
            Q=2.65,                 # 熱損失係数
            A_NR=38.9,              # 非居室面積
            V_vent_l_NR=0.0,        # 非居室換気風量
            V_dash_supply_A=933.0,  # 全館供給風量
            U_prt=2.17,             # 部位熱貫流率
            A_prt_A=107.0,          # 部位面積合計
            L_H_NR_A=5.17,          # 非居室暖房負荷
            L_CS_NR_A=0.0,          # 非居室冷房負荷
            Theta_NR=20.0,          # 非居室温度
            Theta_uf=23.2,          # 床下空調温度
            HCM=JJJ_HCM.H,          # 暖冷房区画の熱容量
            r_A_NR_1F=get_r_A_NR_uf_1F()
        )
        # Assert
        assert result == pytest.approx(19.4, abs=1e-1)  # 19.4 -> 17.8

    def test_冷房期_基本計算1(self):
        """冷房期のテスト - 夏期条件での式52計算"""
        # NOTE: ダクトセントラル・床下定数 変更なし(index=4848)
        # Arrange & Act
        result = get_Theta_star_NR(
            Theta_star_HBR=27.0,    # 冷房設定温度
            Q=2.65,                 # 熱損失係数
            A_NR=38.9,              # 非居室面積
            V_vent_l_NR=0.0,        # 非居室換気風量
            V_dash_supply_A=454.0,  # 全館供給風量
            U_prt=2.17,             # 部位熱貫流率
            A_prt_A=107.0,          # 部位面積合計
            L_H_NR_A=0.0,           # 暖房負荷なし
            L_CS_NR_A=0.495,        # 冷房負荷
            Theta_NR=27.0,          # 非居室温度
            Theta_uf=24.3,          # 床下空調温度
            HCM=JJJ_HCM.C,          # 暖冷房区画の熱容量
            r_A_NR_1F=get_r_A_NR_uf_1F()
        )
        # Assert
        assert result == pytest.approx(26.3, abs=1e-1)  # 26.3 -> 27.1
        # FIXME: 冷房設定温度より低くて問題ないのか
        assert 24.3 < result, "床下空調温度より高い"
