import numpy as np
import pytest

import pyhees.section3_1_e as appendix_e
import pyhees.section4_2 as dc
import jjjexperiment.section4_2 as experiment_section4_2
from jjjexperiment.common import JJJ_HCM
from jjjexperiment.inputs.options import 床下空調ロジック
from jjjexperiment.section4_2 import (
    _solve_shared_ground_feedback,
    combine_corrected_cooling_output,
    get_appendix_e_ground_parameters,
    merge_annual_floor_temperature,
)
from jjjexperiment.underfloor_ac.section3_1_e import (
    calc_Theta_uf_d_t_2023,
    calc_sum_Theta_dash_g_surf_A_m_d_t,
)
from jjjexperiment.underfloor_ac.section4_2_f46_f48 import (
    get_Theta_HBR_i,
    get_Theta_NR,
)
from jjjexperiment.underfloor_ac.section4_2_f52 import get_Theta_star_NR
from jjjexperiment.underfloor_ac.section4_2 import (
    calc_L_flr1st_area_apportioned_d_t,
    calc_Theta_uf,
    calc_delta_L_uf2outdoor,
    get_r_A_NR_uf_1F,
)
from jjjexperiment.underfloor_ac.inputs.common import UnderfloorAc, UfVarsDataFrame


def test_first_floor_non_room_area_includes_bathroom():
    expected_area = 3.31 + 1.66 + 3.31 + 10.76
    assert 38.93 * get_r_A_NR_uf_1F() == pytest.approx(expected_area)


def test_first_floor_load_uses_conditioned_area_ratio():
    q_hat_hs = np.full(24 * 365, 18.526515768)
    area_conditioned_first_floor = 46.37
    area_conditioned_total = 81.15
    expected = q_hat_hs[0] * area_conditioned_first_floor / area_conditioned_total

    heating = calc_L_flr1st_area_apportioned_d_t(
        q_hat_hs, area_conditioned_first_floor, area_conditioned_total, cooling=False
    )
    cooling = calc_L_flr1st_area_apportioned_d_t(
        q_hat_hs, area_conditioned_first_floor, area_conditioned_total, cooling=True
    )

    assert heating[0] == pytest.approx(expected)
    assert heating[0] == pytest.approx(10.5862542965)
    assert cooling[0] == pytest.approx(-expected)


def test_cooling_output_adds_latent_after_sensible_underfloor_correction():
    total_base = np.array([5.1308])
    sensible_base = np.array([2.4613])
    adjusted_total_before_component_clip = np.array([0.8917])

    actual = combine_corrected_cooling_output(
        total_base,
        sensible_base,
        adjusted_total_before_component_clip,
    )

    # 顕熱補正後は負値になるため0とし、補正前潜熱だけが残る。
    expected_latent = total_base - sensible_base
    np.testing.assert_allclose(actual, expected_latent)


def test_underfloor_temperature_uses_distinct_load_and_supply_u_values():
    area_conditioned_first_floor = 46.37
    u_load = 0.5422264459
    u_supply = 2.223
    theta_in = 20.0
    theta_ex = 0.0
    airflow = 500.0
    load = 10.0

    actual = calc_Theta_uf(
        1.0, None, load, area_conditioned_first_floor,
        u_supply, u_load, theta_in, theta_ex, airflow,
    )

    b = dc.get_ro_air() * 1.006 * airflow \
        + u_supply * area_conditioned_first_floor * 3.6
    original_floor_loss = (
        u_load * area_conditioned_first_floor * (theta_in - theta_ex) * 0.7 * 3.6
    )
    expected = (load * 1e3 - original_floor_loss + theta_in * b) / b
    assert actual == pytest.approx(expected)


def test_floor_foundation_and_ground_values_follow_appendix_e():
    theta_ex = np.array([10.0, 12.0, 14.0])
    q_value = 2.6479621918720855

    u_floor, psi_foundation, theta_ground = get_appendix_e_ground_parameters(
        region=6,
        Q=q_value,
        Theta_ex_d_t=theta_ex,
    )

    assert u_floor == pytest.approx(0.5422264459)
    assert psi_foundation == pytest.approx(0.8459576178)
    assert theta_ground == pytest.approx(np.average(theta_ex))


def test_outdoor_heat_transfer_preserves_direction():
    positive = calc_delta_L_uf2outdoor(0.8, 30.0, 5.0)
    negative = calc_delta_L_uf2outdoor(0.8, 30.0, -5.0)
    assert negative == pytest.approx(-positive)


def test_formula_9_subtracts_partition_heat_transfer_for_cooling():
    _, cooling, _ = dc.get_season_array_d_t(6)
    t = np.flatnonzero(cooling)[0]
    sensible_load = np.zeros((5, 24 * 365))
    partition_heat_transfer = np.zeros((5, 24 * 365))
    sensible_load[0, t] = 1.0
    partition_heat_transfer[0, t] = -0.2

    actual = dc.get_L_star_CS_d_t_i(
        sensible_load, partition_heat_transfer, region=6
    )

    # Q* is negative when heat enters the conditioned room from the non-room.
    assert actual[0, t] == pytest.approx(1.0 - (-0.2))


def test_target_floor_temperature_uses_official_season_masks():
    hours = 24 * 365
    heating = np.zeros(hours, dtype=bool)
    cooling = np.zeros(hours, dtype=bool)
    middle = np.ones(hours, dtype=bool)
    # 外気温25℃でも、4章2節の冷房期間に該当する時刻は冷房式を使う。
    cooling[0] = True
    middle[0] = False

    heating_load = np.zeros((5, hours))
    sensible_cooling_load = np.zeros((5, hours))
    sensible_cooling_load[0, 0] = 1.0
    airflow = np.full((5, hours), 100.0)
    outdoor_temperature = np.full(hours, 25.0)

    actual = calc_Theta_uf_d_t_2023(
        heating_load,
        sensible_cooling_load,
        A_A=120.08,
        A_MR=29.81,
        A_OR=51.34,
        r_A_ufvnt=65.41 / 120.08,
        V_dash_supply_d_t_i=airflow,
        Theta_ex_d_t=outdoor_temperature,
        H=heating,
        C=cooling,
        M=middle,
    )

    assert actual[0] < 27.0
    assert actual[0] != pytest.approx(outdoor_temperature[0])
    np.testing.assert_allclose(actual[1:], outdoor_temperature[1:])


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


@pytest.mark.parametrize(
    "hcm,theta_star,theta_uf",
    [
        (JJJ_HCM.H, 20.0, 30.0),
        (JJJ_HCM.C, 27.0, 20.0),
    ],
)
def test_formula_46_uses_uninsulated_floor_heat_transfer(
        hcm, theta_star, theta_uf):
    zeros = np.zeros((5, 1))
    floor_area = np.array([10.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    room_area = np.ones((5, 1))

    actual = get_Theta_HBR_i(
        Theta_star_HBR=theta_star,
        V_supply_i=zeros,
        Theta_supply_i=np.full((5, 1), theta_star),
        U_prt=0.0,
        A_prt_i=zeros,
        Q=1.0,
        A_HCZ_i=room_area,
        L_star_H_i=zeros,
        L_star_CS_i=zeros,
        HCM=hcm,
        A_s_ufac_i=floor_area,
        Theta_uf=theta_uf,
    )

    k_floor = dc.get_U_s() * floor_area[0, 0]
    expected_zone1 = theta_star + (
        k_floor * (theta_uf - theta_star) / (1.0 + k_floor)
    )
    assert dc.get_U_s() == pytest.approx(2.223)
    assert actual[0, 0] == pytest.approx(expected_zone1)
    np.testing.assert_allclose(actual[1:, 0], theta_star)


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


def test_appendix_e_response_at_t_uses_heat_flux_at_t_minus_1():
    hours = 24 * 365
    theta_ex = np.full(hours, 10.0)
    theta_uf_base = np.full(hours, 20.0)
    theta_uf_changed = theta_uf_base.copy()
    theta_uf_changed[0] = 25.0

    base = calc_sum_Theta_dash_g_surf_A_m_d_t(
        theta_uf_base, theta_ex, True, Theta_g_avg=10.0
    )
    changed = calc_sum_Theta_dash_g_surf_A_m_d_t(
        theta_uf_changed, theta_ex, True, Theta_g_avg=10.0
    )

    # 当時刻の応答成分は前時刻の熱流から決まり、当時刻の床下温度には依存しない。
    assert changed[0] == pytest.approx(base[0])
    # 当時刻の床下温度は、次時刻の熱流・応答成分には反映される。
    assert changed[1] != pytest.approx(base[1])


def test_required_supply_temperature_uses_actual_floor_temperature_for_ground_response():
    hours = 24 * 365
    target_floor_temperature = np.full(hours, 25.0)
    outdoor_temperature = np.full(hours, 10.0)
    airflow = np.full(hours, 200.0)
    loads = np.zeros((12, hours))
    new_ufac = UnderfloorAc(
        new_ufac_flg=床下空調ロジック.変更する
    )
    args = dict(
        region=6,
        A_A=120.08,
        A_MR=29.81,
        A_OR=51.34,
        Q=0.87,
        r_A_ufvnt=65.41 / 120.08,
        underfloor_insulation=True,
        Theta_sa_d_t=target_floor_temperature,
        Theta_ex_d_t=outdoor_temperature,
        V_sa_d_t_A=airflow,
        H_OR_C="",
        L_dash_H_R_d_t_i=loads,
        L_dash_CS_R_d_t_i=loads,
        calc_backwards=True,
        new_ufac=new_ufac,
    )

    _, _, target_based_supply = appendix_e.calc_Theta(
        **args,
        new_ufac_df=UfVarsDataFrame(),
    )
    actual_floor_temperature = target_floor_temperature.copy()
    actual_floor_temperature[0] = 20.0
    _, _, actual_based_supply = appendix_e.calc_Theta(
        **args,
        new_ufac_df=UfVarsDataFrame(),
        Theta_uf_ground_feedback_d_t=actual_floor_temperature,
    )

    # 当時刻の応答は前時刻の熱流から決まるため、最初の時刻は同じ。
    assert actual_based_supply[0] == pytest.approx(target_based_supply[0])
    # 1時間目の実床下温度の変更は、2時間目の要求温度へ反映される。
    assert actual_based_supply[1] != pytest.approx(target_based_supply[1])


def test_ground_response_feedback_uses_actual_floor_temperature_and_exports_once(
        monkeypatch):
    calls = []
    output_holder = UfVarsDataFrame()

    def fake_calc_once(**kwargs):
        feedback = kwargs.get("Theta_uf_ground_feedback_d_t")
        export_outputs = kwargs.get("export_outputs", True)
        calls.append((
            None if feedback is None else feedback.copy(),
            export_outputs,
            kwargs["new_ufac_df"],
        ))
        actual = (
            np.zeros(1)
            if feedback is None
            else 0.25 * feedback + 1.5
        )
        marker = "final" if export_outputs else "iteration"
        return marker, 1, 2, 3, 4, 5, 6, actual

    monkeypatch.setattr(
        experiment_section4_2, "_calc_Q_UT_A_once", fake_calc_once
    )

    result = experiment_section4_2.calc_Q_UT_A(
        case_name="case",
        climateFile=None,
        house=None,
        ac_setting=None,
        skin=None,
        heat_CRAC=None,
        cool_CRAC=None,
        new_ufac=UnderfloorAc(
            new_ufac_flg=床下空調ロジック.変更する
        ),
        new_ufac_df=output_holder,
        v_min_heat_input=None,
        v_min_cool_input=None,
        V_hs_dsgn_H=0.0,
        V_hs_dsgn_C=0.0,
        v_supply_cap_dto=None,
        carryover_heat_dto=None,
        load=None,
    )

    assert result[0] == "final"
    assert calls[0][0] is None
    assert calls[1][0] == pytest.approx(np.zeros(1))
    assert all(not export for _, export, _ in calls[:-1])
    assert all(holder is not output_holder for _, _, holder in calls[:-1])
    assert calls[-1][1] is True
    assert calls[-1][2] is output_holder


def test_merge_annual_floor_temperature_uses_each_season_series():
    H = np.zeros(24 * 365, dtype=bool)
    C = np.zeros(24 * 365, dtype=bool)
    H[:2664] = True
    C[2664:6385] = True

    merged = merge_annual_floor_temperature(
        np.full(24 * 365, 21.0),
        np.full(24 * 365, 26.0),
        np.full(24 * 365, 23.0),
        H,
        C,
    )

    assert np.all(merged[:2664] == 21.0)
    assert np.all(merged[2664:6385] == 26.0)
    assert np.all(merged[6385:] == 23.0)


def test_shared_ground_feedback_passes_same_annual_history_to_both_modes(
        monkeypatch):
    H = np.zeros(24 * 365, dtype=bool)
    C = np.zeros(24 * 365, dtype=bool)
    H[:3000] = True
    C[3000:7000] = True
    middle = np.full(24 * 365, 15.0)
    received = {"H": [], "C": []}

    def fake_calc_once(**kwargs):
        mode = kwargs["ac_setting"]
        feedback = kwargs.get("Theta_uf_ground_feedback_d_t")
        received[mode].append(None if feedback is None else feedback.copy())
        if feedback is None:
            actual = np.zeros(24 * 365)
        elif mode == "H":
            actual = 0.25 * feedback + 10.0
        else:
            actual = 0.25 * feedback + 20.0
        return 0, 1, 2, 3, 4, 5, 6, actual, middle

    monkeypatch.setattr(
        experiment_section4_2, "_calc_Q_UT_A_once", fake_calc_once
    )
    feedback, iteration_count, max_delta = _solve_shared_ground_feedback(
        {"ac_setting": "H", "new_ufac_df": UfVarsDataFrame()},
        {"ac_setting": "C", "new_ufac_df": UfVarsDataFrame()},
        H,
        C,
    )

    assert iteration_count <= experiment_section4_2.GROUND_FEEDBACK_MAX_ITERATIONS
    assert max_delta <= experiment_section4_2.GROUND_FEEDBACK_TOLERANCE
    np.testing.assert_allclose(feedback[:3000], 10.0 / 0.75, atol=1.0e-4)
    np.testing.assert_allclose(feedback[3000:7000], 20.0 / 0.75, atol=1.0e-4)
    assert np.all(feedback[7000:] == 15.0)
    for heating_feedback, cooling_feedback in zip(
            received["H"][1:], received["C"][1:]):
        np.testing.assert_allclose(heating_feedback, cooling_feedback)
