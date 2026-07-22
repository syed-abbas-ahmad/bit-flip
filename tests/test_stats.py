"""Release statistical gate: all rows contribute, negative controls trip it.

Regression coverage for the notebook's once-insensitive gate: the delivered
cell filtered to the 62 sweep rows, so mutating a p = 0.5 boundary row by
316 sigma still printed "Global statistical criterion passed". The gate now
lives in ``bitflip.stats`` and reads every row.
"""

import numpy as np
import pytest

from bitflip.stats import gate_report, population_report, z_scores


@pytest.fixture()
def sweep_frame():
    import pandas as pd

    return pd.read_csv("results/sweep.csv")


def test_gate_passes_on_the_delivered_sweep(sweep_frame):
    report = gate_report(sweep_frame)
    assert report["n_rows"] == 64
    assert report["n_deterministic"] == 2
    assert report["deterministic_ok"]
    assert report["passed"]
    # The published numbers, recomputed from the named population.
    assert report["max_abs_z"] == pytest.approx(2.92734, abs=1e-4)
    assert report["chi2_red"] == pytest.approx(1.11688916, abs=1e-6)


def test_gate_uses_all_rows_including_the_boundary(sweep_frame):
    # Regression control: mutate the p = 0.5, q = 0.02 boundary row to a
    # 316-sigma outlier. The gate must fail even though the row is outside
    # the plotted sweep grid.
    mutated = sweep_frame.copy()
    mask = np.isclose(mutated["p"], 0.5) & np.isclose(mutated["q"], 0.02)
    assert mask.sum() == 1
    mutated.loc[mask, "logical_failure_count"] = mutated.loc[mask, "shots"]
    mutated.loc[mask, "logical_failure"] = 1.0
    report = gate_report(mutated)
    assert not report["passed"]
    assert report["max_abs_z"] > 100


def test_gate_trips_on_an_in_sweep_row_too(sweep_frame):
    mutated = sweep_frame.copy()
    mask = np.isclose(mutated["p"], 0.1) & np.isclose(mutated["q"], 0.0)
    mutated.loc[mask, "logical_failure_count"] = mutated.loc[mask, "shots"]
    mutated.loc[mask, "logical_failure"] = 1.0
    assert not gate_report(mutated)["passed"]


def test_deterministic_rows_are_checked_explicitly(sweep_frame):
    # A p = 0 row with observed failures must fail the gate even though its
    # exact rate (and therefore z) is zero.
    mutated = sweep_frame.copy()
    mask = np.isclose(mutated["p"], 0.0)
    assert mask.sum() == 2
    mutated.loc[mask, "logical_failure_count"] = 5
    mutated.loc[mask, "logical_failure"] = 5e-5
    report = gate_report(mutated)
    assert not report["deterministic_ok"]
    assert not report["passed"]


def test_population_report_computes_both_subsets(sweep_frame):
    populations = population_report(sweep_frame)
    assert populations["all_rows"]["n"] == 64
    assert populations["excluding_deterministic_rows"]["n"] == 62
    assert populations["all_rows"]["chi2_red"] == pytest.approx(1.11688916, abs=1e-6)
    assert populations["excluding_deterministic_rows"]["chi2_red"] == pytest.approx(
        1.15291785, abs=1e-6)
    # The z-scores themselves must be identical between the full population
    # and the subset for the shared rows.
    full = z_scores(sweep_frame)
    sub = z_scores(sweep_frame[~np.isclose(sweep_frame["p"], 0.0)])
    assert np.allclose(full.loc[sub.index], sub)


def test_empty_frame_rejected():
    import pandas as pd

    with pytest.raises(ValueError):
        gate_report(pd.DataFrame(columns=["p", "q", "shots", "logical_failure",
                                          "logical_failure_exact", "logical_failure_count"]))
