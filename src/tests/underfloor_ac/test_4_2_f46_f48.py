import os
import pytest
import numpy as np

# JJJ
from jjjexperiment.common import *

from jjjexperiment.inputs.options import *
from jjjexperiment.inputs.di_container import create_injector_from_json
from jjjexperiment.inputs.environment_service import EnvironmentService
from jjjexperiment.inputs.common import HouseInfo, OuterSkin
from jjjexperiment.underfloor_ac.section4_2 import get_A_s_ufac_i, get_r_A_NR_uf_1F
from jjjexperiment.underfloor_ac.section4_2_f46_f48 import get_Theta_HBR_i, get_Theta_NR

from test_utils.utils import load_input_yaml

@pytest.mark.xfail(reason="260323_井口先生よりロジック修正中のため")
class Test_床下空調時_式46_式48:

    def test_式46_時点計算例(self):
        """
        (46) 暖冷房区画iの実際の居室の室温
        """
        # Arrange
        yaml_fullpath = os.path.join(os.path.dirname(__file__), 'test_input.yaml')
        injector = create_injector_from_json(load_input_yaml(yaml_fullpath))

        skin = injector.get(OuterSkin)
        house = injector.get(HouseInfo)
        environment = EnvironmentService(house, skin)

        A_s_ufac_i, _ = get_A_s_ufac_i(house.A_A, house.A_MR, house.A_OR)
        A_HCZ_i = environment.get_A_HCZ_i().reshape(-1, 1)

        # Act
        Theta_star_HBR = 20.0
        Theta_HBR_i = get_Theta_HBR_i(
            Theta_star_HBR = Theta_star_HBR,
            V_supply_i = np.array([342.3, 190.2, 152.0, 123.6, 123.7]).reshape(-1, 1),
            Theta_supply_i = np.array([24.89, 24.89, 36.78, 35.39, 36.85]).reshape(-1, 1),
            U_prt = 2.17,
            A_prt_i = np.array([32.92, 24.02, 19.22, 15.61, 15.61]).reshape(-1, 1),
            Q = 2.6472,
            A_HCZ_i = A_HCZ_i[:5, :],
            L_star_H_i = np.array([3.639, 1.308, 1.881, 1.752, 2.178]).reshape(-1, 1),
            L_star_CS_i = np.array([0, 0, 0, 0, 0]).reshape(-1, 1),
            HCM = JJJ_HCM.H,
            A_s_ufac_i = A_s_ufac_i[:5, :],
            Theta_uf = 24.89
        )

        # Assert
        # NOTE: 従来式同様のキャップロジックも今回も有効にする(確認済:25'04/11)
        assert Theta_HBR_i[0, 0] == pytest.approx(max(19.62, Theta_star_HBR), abs=1e-2)
        assert Theta_HBR_i[1, 0] == pytest.approx(max(20.65, Theta_star_HBR), abs=1e-2)
        assert Theta_HBR_i[2, 0] == pytest.approx(max(22.60, Theta_star_HBR), abs=1e-2)
        assert Theta_HBR_i[3, 0] == pytest.approx(max(21.45, Theta_star_HBR), abs=1e-2)
        assert Theta_HBR_i[4, 0] == pytest.approx(max(20.90, Theta_star_HBR), abs=1e-2)


    def test_式48_時点計算例(self):
        """
        (48) 実際の非居室の室温
        """
        # Arrange

        # Act
        Theta_NR = get_Theta_NR(
            Theta_star_NR = 19.40,
            Theta_star_HBR = 20.0,
            # 上テストのアサートと同一値
            Theta_HBR_i = np.array([19.62, 20.65, 22.60, 21.45, 20.90]).reshape(-1, 1),
            A_NR = 38.93,
            V_vent_l_NR = 0,
            V_dash_supply_i = np.array([342.3, 190.2, 152.0, 123.6, 123.7]).reshape(-1, 1),
            V_supply_i = np.array([342.3, 190.2, 152.0, 123.6, 123.7]).reshape(-1, 1),
            U_prt = 2.17,
            A_prt_i = np.array([32.92, 24.02, 19.22, 15.61, 15.61]).reshape(-1, 1),
            Q = 2.6472,
            Theta_uf = 24.89,
            r_A_NR_1F = get_r_A_NR_uf_1F()
        )
        # Assert
        assert Theta_NR == pytest.approx(20.65, abs=1e-2)  # 20.64 -> 20.40
