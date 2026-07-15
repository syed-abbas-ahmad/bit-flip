"""Anchor bindings: each expected value must match its own (p, q, observable).

Regression coverage for the delivered notebook's once-failing clean
benchmark: the clean anchor was compared against the noisy oracle and
printed ``match False`` while the cell still exited successfully.
"""

import pytest

from bitflip.anchors import ANCHOR_TOL, check_anchors, swapped_anchors


def test_all_anchors_match_their_own_settings():
    rows = check_anchors()
    assert len(rows) == 4
    for row in rows:
        assert row["match"], f"anchor failed: {row}"


def test_clean_anchor_uses_q_zero_and_noisy_uses_q_point_zero_two():
    rows = check_anchors()
    clean = next(r for r in rows if "clean" in r["label"])
    noisy = next(r for r in rows if r["label"] == "logical_failure (q=0.02)")
    assert clean["q"] == 0.0 and clean["observed"] == pytest.approx(0.028, abs=ANCHOR_TOL)
    assert noisy["q"] == 0.02 and noisy["observed"] == pytest.approx(
        0.0337024, abs=ANCHOR_TOL)


def test_swapping_settings_makes_the_comparison_fail():
    # Proves the comparison is discriminative: binding the clean anchor to
    # the noisy oracle must produce a visible mismatch (the notebook bug).
    rows = check_anchors(swapped_anchors())
    clean = next(r for r in rows if "clean" in r["label"])
    assert not clean["match"]
    assert abs(clean["observed"] - clean["expected"]) > 1e-3
    # The untouched noisy anchors still match.
    for row in rows:
        if "clean" not in row["label"]:
            assert row["match"]
