"""Exact 32-event oracle: anchors, boundaries, symmetries, normalization."""

import numpy as np
import pytest

from bitflip.enumeration import clean_logical_failure, event_table, exact_rates


def test_table_has_32_normalized_events():
    for p, q in [(0.1, 0.02), (0.3, 0.0), (0.5, 0.5), (0.0, 0.9), (1.0, 0.2)]:
        table = event_table(p, q)
        assert len(table) == 32
        assert table["prob"].sum() == pytest.approx(1.0, abs=1e-12)


def test_anchors_at_p01_q002():
    # Independent audit anchors.
    rates = exact_rates(0.1, 0.02)
    assert rates["logical_failure"] == pytest.approx(0.0337024, abs=1e-9)
    assert rates["any_residual_x"] == pytest.approx(0.0664912, abs=1e-9)
    assert rates["outside_code_space"] == pytest.approx(0.0396, abs=1e-9)
    assert rates["unencoded_failure"] == 0.1


def test_clean_law_matches_enumeration():
    for p in np.linspace(0.0, 1.0, 11):
        assert exact_rates(p, 0.0)["logical_failure"] == pytest.approx(
            clean_logical_failure(p), abs=1e-12
        )
        # With a truthful readout nothing leaves the code space.
        assert exact_rates(p, 0.0)["outside_code_space"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("q", [0.0, 0.02, 0.5, 1.0])
def test_symmetry_l_1mp_equals_1_minus_l(q):
    for p in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        left = exact_rates(1.0 - p, q)["logical_failure"]
        right = 1.0 - exact_rates(p, q)["logical_failure"]
        assert left == pytest.approx(right, abs=1e-12)


def test_boundary_p_zero():
    # No physical errors: a bad report can still inject a residual, and the
    # block leaves the code space with probability 1 - (1-q)^2, while the
    # ideal majority always reads the correct logical bit.
    for q in [0.0, 0.02, 0.5, 1.0]:
        rates = exact_rates(0.0, q)
        assert rates["logical_failure"] == pytest.approx(0.0, abs=1e-12)
        assert rates["outside_code_space"] == pytest.approx(1.0 - (1.0 - q) ** 2, abs=1e-12)
        assert rates["any_residual_x"] == pytest.approx(1.0 - (1.0 - q) ** 2, abs=1e-12)


def test_boundary_p_one_and_p_half():
    # p = 1: all-ones error is a logical X inside the code space when the
    # report is truthful, and stays weight 3 after any correction.
    assert exact_rates(1.0, 0.0)["logical_failure"] == pytest.approx(1.0, abs=1e-12)
    assert exact_rates(1.0, 0.0)["outside_code_space"] == pytest.approx(0.0, abs=1e-12)
    # p = 0.5: code and unencoded both sit at 50% for any q.
    for q in [0.0, 0.02, 0.7]:
        assert exact_rates(0.5, q)["logical_failure"] == pytest.approx(0.5, abs=1e-12)


def test_noisy_analytic_formula_matches_oracle():
    # L = L0 + (4q - 2q^2) p(1-p)(1-2p), derived independently of the oracle.
    for p in np.linspace(0.0, 1.0, 21):
        for q in [0.0, 0.02, 0.15, 0.5, 1.0]:
            expected = clean_logical_failure(p) + (4 * q - 2 * q * q) * p * (1 - p) * (1 - 2 * p)
            assert exact_rates(p, q)["logical_failure"] == pytest.approx(expected, abs=1e-12)


def test_residual_formula_matches_oracle():
    for p in [0.0, 0.1, 0.4, 0.9]:
        for q in [0.0, 0.02, 0.4, 1.0]:
            expected = 1.0 - (1.0 - q) ** 2 * (1.0 - clean_logical_failure(p))
            assert exact_rates(p, q)["any_residual_x"] == pytest.approx(expected, abs=1e-12)


def test_rates_within_unit_interval():
    for p in np.linspace(0.0, 1.0, 11):
        for q in np.linspace(0.0, 1.0, 6):
            for value in exact_rates(p, q).values():
                assert 0.0 <= value <= 1.0


def test_per_site_residual_marginals_and_residual_while_correct():
    # The block-level "any residual X" observable (6.64912% at p=0.1,
    # q=0.02) is a different physical question from per-qubit damage.
    # Per-site marginals are asymmetric because the decoder breaks the
    # symmetry: 4.656% / 3.39648% / 4.656% for qubits 1, 2 and 3.
    table = event_table(0.1, 0.02)
    for i, expected in enumerate([0.04656, 0.0339648, 0.04656]):
        site = table.loc[table["final_block"].str[i] == "1", "prob"].sum()
        assert site == pytest.approx(expected, abs=1e-9)
    # Residual X while the ideal majority still gives the correct bit
    # (weight exactly 1): 3.27888%. This equals 6.64912% - 3.37024%.
    weight1 = table.loc[table["weight"] == 1, "prob"].sum()
    assert weight1 == pytest.approx(0.0327888, abs=1e-9)


def test_q_one_reports_are_still_decodable_deterministically():
    # q = 1 flips both reports deterministically: reported = NOT true.
    # Check the oracle against a hand-derived case: p=0 -> true 00,
    # reported 11, correction on q2, final 010, majority 0 -> no failure.
    assert exact_rates(0.0, 1.0)["logical_failure"] == pytest.approx(0.0, abs=1e-12)
