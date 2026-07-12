import numpy as np
import pytest

import pyhees.section4_2 as dc
from jjjexperiment.common import JJJ_HCM
from jjjexperiment.underfloor_ac.section3_1_e import (
    calc_sum_Theta_dash_g_surf_A_m_d_t,
)
from jjjexperiment.underfloor_ac.section4_2_f46_f48 import get_Theta_NR
from jjjexperiment.underfloor_ac.section4_2_f52 import get_Theta_star_NR
from jjjexperiment.underfloor_ac.section4_2 import (
    calc_L_flr1st_d_t,
    calc_delta_L_uf2outdoor,
    get_r_A_NR_uf_1F,
)


def test_first_floor_non_room_area_includes_bathroom():
    expected_area = 3.31 + 1.66 + 3.31 + 10.76
    assert 38.93 * get_r_A_NR_uf_1F() == pytest.approx(expected_area)


def test_first_floor_load_uses_zone_contact_ratios():
    loads = np.zeros((12, 24 * 365))
    loads[:, 0] = 1.0
    expected = sum([1.0, 1.0, 1.0, 1.0, 1.0, 0.8120754716981132])
    assert calc_L_flr1st_d_t(loads, cooling=False)[0] == pytest.approx(expected)
    assert calc_L_flr1st_d_t(loads, cooling=True)[0] == pytest.approx(-expected)


def test_outdoor_heat_transfer_preserves_direction():
    positive = calc_delta_L_uf2outdoor(0.8, 30.0, 5.0)
    negative = calc_delta_L_uf2outdoor(0.8, 30.0, -5.0)
    assert negative == pytest.approx(-positive)


def test_formula_52_uses_first_floor_contact_area():
    area_nr = 40.0
    contact_ratio = 0.4
    q_value = 2.65
    theta_hbr = 20.0
    theta_uf = 30.0

    actual = get_Theta_star_NR(
        Theta_star_HBR=theta_hbr,
        Q=q_value,
        A_NR=area_nr,
        V_vent_l_NR=0.0,
        V_dash_supply_A=0.0,
        U_prt=0.0,
        A_prt_A=0.0,
        L_H_NR_A=0.0,
        L_CS_NR_A=0.0,
        Theta_NR=theta_hbr,
        Theta_uf=theta_uf,
        HCM=JJJ_HCM.H,
        r_A_NR_1F=contact_ratio,
    )

    k1 = (q_value - 0.35 * 0.5 * 2.4) * area_nr
    k2 = dc.get_U_s() * area_nr * contact_ratio
    expected = (k1 * theta_hbr + k2 * theta_uf) / (k1 + k2)
    assert actual == pytest.approx(expected)


def test_formula_48_uses_first_floor_contact_area():
    area_nr = 40.0
    contact_ratio = 0.4
    q_value = 2.65
    theta_star_nr = 20.0
    theta_uf = 30.0
    zeros = np.zeros((5, 1))

    actual = get_Theta_NR(
        Theta_star_NR=theta_star_nr,
        Theta_star_HBR=theta_star_nr,
        Theta_HBR_i=np.full((5, 1), theta_star_nr),
        A_NR=area_nr,
        V_vent_l_NR=0.0,
        V_dash_supply_i=zeros,
        V_supply_i=zeros,
        U_prt=0.0,
        A_prt_i=zeros,
        Q=q_value,
        Theta_uf=theta_uf,
        r_A_NR_1F=contact_ratio,
    )

    k_evp = (q_value - 0.35 * 0.5 * 2.4) * area_nr
    k_floor = dc.get_U_s() * area_nr * contact_ratio
    expected = (
        k_evp * theta_star_nr + k_floor * theta_uf
    ) / (k_evp + k_floor)
    assert actual == pytest.approx(expected)


def test_appendix_e_ground_response_is_hourly():
    hours = 24 * 365
    phase = np.arange(hours) * 2.0 * np.pi / hours
    theta_ex = 15.0 + 10.0 * np.sin(phase)
    theta_uf = 20.0 + 5.0 * np.sin(phase + 0.25)

    response = calc_sum_Theta_dash_g_surf_A_m_d_t(
        theta_uf, theta_ex, underfloor_insulation=False
    )

    assert response.shape == (hours,)
    assert np.all(np.isfinite(response))
    assert np.ptp(response) > 0.0


def test_appendix_e_ground_response_uses_configured_ground_temperature():
    hours = 24 * 365
    theta_ex = np.full(hours, 10.0)
    theta_uf = np.full(hours, 20.0)
    low = calc_sum_Theta_dash_g_surf_A_m_d_t(
        theta_uf, theta_ex, False, Theta_g_avg=10.0
    )
    high = calc_sum_Theta_dash_g_surf_A_m_d_t(
        theta_uf, theta_ex, False, Theta_g_avg=18.0
    )
    assert not np.allclose(low, high)
