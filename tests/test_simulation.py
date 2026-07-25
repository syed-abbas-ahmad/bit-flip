"""Monte Carlo simulation: statistical agreement, boundaries, reproducibility."""

import numpy as np
import pytest

from bitflip.cli import P_CHECK, P_SWEEP, Q_VALUES
from bitflip.enumeration import clean_logical_failure, exact_rates
from bitflip.simulation import (
    RATE_COLUMNS,
    run_sweep,
    shots_for_point,
    simulate_point,
    wilson_interval,
)

# Documented tolerance policy (see README "Statistical tolerance"):
# each point is judged by its z-score (mc - exact) / Wilson-or-binomial sigma;
# a *global* criterion is applied across many points, never a per-point 95%
# gate, which would fail by chance on ~1 point in 20.
Z_LIMIT = 4.0


def z_scores(results):
    zs = []
    for r in results:
        exact = exact_rates(r.p, r.q)["logical_failure"]
        sigma = np.sqrt(max(exact * (1 - exact), 1e-15) / r.shots)
        if sigma <= 0:
            continue  # unreachable for 0 < p < 1; p = 0 handled by boundary tests
        zs.append((r.rates["logical_failure"] - exact) / sigma)
    return zs


def test_reproducible_with_same_seed():
    a = simulate_point(0.1, 0.02, 50_000, seed=12345)
    b = simulate_point(0.1, 0.02, 50_000, seed=12345)
    assert a.counts == b.counts
    c = simulate_point(0.1, 0.02, 50_000, seed=12346)
    assert a.counts != c.counts  # different stream, essentially certain a priori


def test_mc_matches_exact_across_points():
    # 12 points across the sweep; global z-criterion, no per-point gate.
    results = [
        simulate_point(p, q, 200_000, seed=1000 + i)
        for i, (p, q) in enumerate(
            [(0.02, 0.0), (0.05, 0.0), (0.1, 0.0), (0.2, 0.0),
             (0.02, 0.02), (0.05, 0.02), (0.1, 0.02), (0.2, 0.02),
             (0.3, 0.0), (0.3, 0.02), (0.5, 0.0), (0.5, 0.02)]
        )
    ]
    zs = z_scores(results)
    assert len(zs) == 12
    assert max(abs(z) for z in zs) < Z_LIMIT
    # Reduced chi-square should be O(1); allow generous upper tail.
    chi2_red = float(np.mean(np.square(zs)))
    assert chi2_red < 2.5


def test_correction_actually_used_before_readout():
    # A simulation that ignored the correction would reproduce the clean
    # curve at positive q. The noisy rate must instead sit above L0 by the
    # predicted (4q - 2q^2) p(1-p)(1-2p), which here is ~0.0057 - about 8
    # binomial sigmas at 1e6 shots, far beyond noise.
    p, q, shots = 0.1, 0.02, 1_000_000
    r = simulate_point(p, q, shots, seed=777)
    l0 = clean_logical_failure(p)
    delta_pred = (4 * q - 2 * q * q) * p * (1 - p) * (1 - 2 * p)
    sigma = np.sqrt(l0 * (1 - l0) / shots)
    assert r.rates["logical_failure"] > l0 + 4 * sigma  # clearly above clean
    assert abs(r.rates["logical_failure"] - (l0 + delta_pred)) < 4.5 * sigma


def test_boundary_p_zero():
    r = simulate_point(0.0, 0.02, 200_000, seed=1)
    assert r.counts["logical_failure"] == 0
    assert r.counts["any_residual_x"] > 0  # corrections inject physical flips
    exact = exact_rates(0.0, 0.02)
    assert r.rates["any_residual_x"] == pytest.approx(exact["any_residual_x"], abs=0.01)
    assert r.rates["outside_code_space"] == pytest.approx(exact["outside_code_space"], abs=0.01)
    # Zero failures must still yield a finite Wilson upper bound.
    assert 0.0 < r.ci_high["logical_failure"] < 0.001


def test_boundary_p_half_and_p_one():
    r = simulate_point(0.5, 0.0, 100_000, seed=2)
    assert r.rates["logical_failure"] == pytest.approx(0.5, abs=0.02)
    r1 = simulate_point(1.0, 0.0, 50_000, seed=3)
    assert r1.rates["logical_failure"] == pytest.approx(1.0, abs=1e-12)


def test_unencoded_baseline_tracked():
    r = simulate_point(0.1, 0.02, 300_000, seed=4)
    assert r.rates["unencoded_failure"] == pytest.approx(0.1, abs=0.01)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        simulate_point(-0.1, 0.0, 100, seed=0)
    with pytest.raises(ValueError):
        simulate_point(0.1, 1.5, 100, seed=0)
    with pytest.raises(ValueError):
        simulate_point(0.1, 0.0, 0, seed=0)


def test_wilson_interval_properties():
    lo, hi = wilson_interval(0, 1000)
    assert lo == 0.0 and 0.0 < hi < 0.005  # rule-of-three scale for zero events
    lo, hi = wilson_interval(1000, 1000)
    assert 0.995 < lo < 1.0 and hi == 1.0
    lo, hi = wilson_interval(280, 10_000)  # 2.8%
    assert lo < 0.028 < hi
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_adaptive_budget_increases_for_rare_failures():
    assert shots_for_point(0.01, 0.0) > shots_for_point(0.25, 0.0)
    assert shots_for_point(0.0, 0.02) == 100_000  # no failures possible: minimum
    assert 100_000 <= shots_for_point(0.1, 0.02) <= 1_000_000


def test_run_sweep_materializes_iterables():
    # A one-shot q iterator must be materialized: the old code exhausted it
    # after the first p, silently producing no results for later p values.
    results = run_sweep([0.0, 0.1], iter([0.0, 0.1]), shots=100, seed=1)
    assert len(results) == 4  # 2 p x 2 q


def test_run_sweep_rejects_off_grid_values():
    # Seed derivation quantizes p and q to 4 decimals; distinct values that
    # agree to 4 decimals would silently share a random stream.
    with pytest.raises(ValueError, match="4-decimal"):
        run_sweep([0.100001], (0.02,), shots=100, seed=1)


def test_published_grid_has_distinct_random_streams():
    # The 64 published (p, q) settings must map to 64 distinct derived
    # seeds (one 32-bit word each), so no two published points share a
    # random stream.
    seeds = []
    for p in P_SWEEP + P_CHECK:
        for q in Q_VALUES:
            seq = np.random.SeedSequence(
                entropy=20261122, spawn_key=(round(p * 10_000), round(q * 10_000)))
            seeds.append(int(seq.generate_state(1, dtype=np.uint32)[0]))
    assert len(seeds) == len(set(seeds)) == 64
