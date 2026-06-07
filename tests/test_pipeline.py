"""Pipeline, scenario records and CSV cross-checks."""

import numpy as np
import pandas as pd
import pytest

from bitflip.enumeration import event_table, exact_rates
from bitflip.scenarios import scenarios
from bitflip.simulation import RATE_COLUMNS, run_sweep, sweep_frame


def test_scenario_records_match_oracle():
    data = scenarios()

    # Scenario 1: clean correction of mask 010.
    s = data["clean"]
    assert (s["error_mask"], s["reported_syndrome"]) == ([0, 1, 0], (1, 1))
    assert s["final_block"] == [0, 0, 0]
    assert s["outcome"] == "clean_success"

    # Scenario 2: misleading report causes a logical failure (audit example).
    s = data["misleading"]
    assert (s["error_mask"], s["reported_syndrome"]) == ([1, 0, 0], (1, 1))
    assert s["true_syndrome"] == (1, 0)
    assert s["final_block"] == [1, 1, 0]
    assert s["logical_bit"] == 1
    assert s["outcome"] == "logical_failure"

    # Scenario 3: false report injects a residual, logical bit survives.
    s = data["false_report"]
    assert (s["error_mask"], s["reported_syndrome"]) == ([0, 0, 0], (0, 1))
    assert s["final_block"] == [0, 0, 1]
    assert s["logical_bit"] == 0
    assert s["outcome"] == "residual_only"
    assert s["outside_code_space"] is True

    # Every reported syndrome in the scenarios is reachable in the oracle
    # table with the stated noise bit (i.e. consistent with the model).
    table = event_table(0.1, 0.02)
    for name, s in data.items():
        if s["noise_bit"] is None:
            noise = "00"
        elif s["noise_bit"] == 1:
            noise = "01"  # flips s2 only
        else:
            noise = "10"  # flips s1 only
        match = table[
            (table["error_mask"] == "".join(map(str, s["error_mask"])))
            & (table["report_noise"] == noise)
        ]
        assert len(match) == 1, name
        row = match.iloc[0]
        assert row["reported_syndrome"] == "".join(map(str, s["reported_syndrome"]))
        assert row["final_block"] == "".join(map(str, s["final_block"]))


def test_small_sweep_frame_consistency():
    results = run_sweep([0.0, 0.1, 0.3, 0.5], (0.0, 0.02), shots=50_000, seed=42)
    frame = sweep_frame(results)
    assert len(frame) == 8
    for name in RATE_COLUMNS:
        assert {name, f"{name}_exact", f"{name}_ci_low", f"{name}_ci_high", f"{name}_count"} <= set(frame.columns)

    # Rates inside their Wilson intervals and inside [0, 1].
    for _, row in frame.iterrows():
        for name in RATE_COLUMNS:
            assert 0.0 <= row[name] <= 1.0
            assert row[f"{name}_ci_low"] <= row[name] + 1e-12
            assert row[name] <= row[f"{name}_ci_high"] + 1e-12

    # MC must agree with the oracle within the documented global z-criterion.
    zs = []
    for _, row in frame.iterrows():
        exact = row["logical_failure_exact"]
        sigma = np.sqrt(max(exact * (1 - exact), 1e-15) / row["shots"])
        if sigma > 0:
            zs.append((row["logical_failure"] - exact) / sigma)
    assert max(abs(z) for z in zs) < 4.0
    assert float(np.mean(np.square(zs))) < 2.5


def test_exactly_zero_failures_handled():
    results = run_sweep([0.0], (0.02,), shots=20_000, seed=7)
    frame = sweep_frame(results)
    row = frame.iloc[0]
    assert row["logical_failure_count"] == 0
    assert 0.0 < row["logical_failure_ci_high"] < 0.001
    assert row["logical_failure_exact"] == 0.0
