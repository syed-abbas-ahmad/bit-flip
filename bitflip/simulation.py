"""Seeded Monte Carlo simulation of one correction round.

Each shot draws independent physical X errors (probability p per qubit),
computes the true adjacent syndromes, independently flips each of the two
reported parity bits with probability q, applies the correction selected by
the *reported* syndrome, and then performs an ideal majority readout of the
corrected block. Logical failure and any residual physical X are logged
separately, matching the definitions in ``enumeration.exact_rates``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .enumeration import exact_rates

#: Columns reported per (p, q) point.
RATE_COLUMNS = (
    "logical_failure",
    "any_residual_x",
    "outside_code_space",
    "unencoded_failure",
)


@dataclass(frozen=True)
class PointResult:
    """Counts and rates for one (p, q) simulation point."""

    p: float
    q: float
    shots: int
    counts: dict[str, int]
    rates: dict[str, float]
    ci_low: dict[str, float]
    ci_high: dict[str, float]

    def as_row(self) -> dict[str, float | int]:
        row: dict[str, float | int] = {"p": self.p, "q": self.q, "shots": self.shots}
        for name in RATE_COLUMNS:
            row[name] = self.rates[name]
            row[f"{name}_ci_low"] = self.ci_low[name]
            row[f"{name}_ci_high"] = self.ci_high[name]
            row[f"{name}_count"] = self.counts[name]
        return row


def wilson_interval(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default).

    Unlike the normal approximation, the Wilson interval stays in [0, 1]
    and remains finite when ``successes`` is 0, so zero observed failures
    are reported as "< upper bound" rather than a meaningless 0 ± 0.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if successes < 0 or successes > n:
        raise ValueError("successes must be within [0, n]")
    if z <= 0:
        raise ValueError(f"z must be positive, got {z}")
    if successes == 0:
        return (0.0, float(min(1.0, centre_for(successes, n, z)[1])))
    if successes == n:
        return (float(max(0.0, centre_for(successes, n, z)[0])), 1.0)
    lo, hi = centre_for(successes, n, z)
    return (max(0.0, lo), min(1.0, hi))


def centre_for(successes: int, n: int, z: float) -> tuple[float, float]:
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (phat + z2 / (2.0 * n)) / denom
    half = z * np.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / denom
    return (centre - half, centre + half)


def simulate_point(p: float, q: float, shots: int, seed: int) -> PointResult:
    """Simulate ``shots`` independent rounds at physical error rate p and
    parity-report flip rate q."""
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")

    rng = np.random.default_rng(seed)
    errors = rng.random((shots, 3)) < p
    # True adjacent syndromes: s1 = q1 XOR q2, s2 = q2 XOR q3.
    s1 = errors[:, 0] ^ errors[:, 1]
    s2 = errors[:, 1] ^ errors[:, 2]
    noise = rng.random((shots, 2)) < q
    r1 = s1 ^ noise[:, 0]
    r2 = s2 ^ noise[:, 1]

    correction = np.zeros((shots, 3), dtype=bool)
    correction[r1 & ~r2, 0] = True
    correction[r1 & r2, 1] = True
    correction[~r1 & r2, 2] = True

    final = errors ^ correction
    weight = final.sum(axis=1)

    # Unencoded baseline: a separate stream of single bits, same seed family.
    unencoded = rng.random(shots) < p

    counts = {
        "logical_failure": int((weight >= 2).sum()),
        "any_residual_x": int((weight >= 1).sum()),
        "outside_code_space": int(
            (~((~final).all(axis=1) | final.all(axis=1))).sum()
        ),
        "unencoded_failure": int(unencoded.sum()),
    }
    rates = {name: count / shots for name, count in counts.items()}
    ci_low, ci_high = {}, {}
    for name, count in counts.items():
        low, high = wilson_interval(count, shots)
        ci_low[name] = float(low)
        ci_high[name] = float(high)
    return PointResult(
        p=p, q=q, shots=shots, counts=counts, rates=rates,
        ci_low=ci_low, ci_high=ci_high,
    )


def shots_for_point(p: float, q: float, *, min_shots: int = 100_000,
                    max_shots: int = 1_000_000,
                    target_failures: float = 200.0) -> int:
    """Adaptive shot budget: raise shots where failures are rare.

    The budget aims for ``target_failures`` expected logical failures using
    the *exact* rate from the 32-event oracle, clipped to
    ``[min_shots, max_shots]``. At p = 0 the failure probability is 0, so the
    minimum budget applies and zero observed failures are reported through
    the Wilson upper bound instead of a point estimate.
    """
    rate = exact_rates(p, q)["logical_failure"]
    if rate <= 0.0:
        return min_shots
    needed = int(np.ceil(target_failures / rate))
    return int(min(max_shots, max(min_shots, needed)))


def run_sweep(
    p_values,
    q_values=(0.0, 0.02),
    shots: int | None = None,
    seed: int = 20261122,
) -> "list[PointResult]":
    """Simulate every (p, q) combination with a deterministic per-point seed.

    The seed for each point is derived from the master seed and the point
    itself (via ``numpy.random.SeedSequence`` spawn keys), so every point is
    an independent stream and results do not depend on sweep order.

    Supported grid: p and q are quantized to 4 decimal places for seed
    derivation, so distinct values that agree to 4 decimals would share a
    random stream; values that are not multiples of 1e-4 (within float
    tolerance) are rejected. The published sweep uses the 0.01 grid, which
    is safely inside this constraint. Iterables are materialized, so
    one-shot iterators (e.g. generators) are accepted.

    ``shots=None`` selects the adaptive budget from ``shots_for_point``;
    an integer fixes the budget for every point (used by tests).
    """
    p_values = tuple(p_values)
    q_values = tuple(q_values)
    results = []
    for p in p_values:
        _check_grid_value(p, "p")
        for q in q_values:
            _check_grid_value(q, "q")
            seq = np.random.SeedSequence(entropy=seed, spawn_key=(round(p * 10_000), round(q * 10_000)))
            budget = shots_for_point(p, q) if shots is None else shots
            results.append(
                simulate_point(p, q, budget, seed=int(seq.generate_state(1, dtype=np.uint32)[0]))
            )
    return results


def _check_grid_value(value: float, name: str) -> None:
    """Reject values outside the supported 4-decimal seed grid."""
    if abs(value - round(value, 4)) > 1e-9:
        raise ValueError(
            f"{name}={value!r} is outside the supported 4-decimal grid; seed "
            "derivation quantizes p and q to 4 decimals (round(value, 4))"
        )


def sweep_frame(results) -> "pd.DataFrame":
    import pandas as pd

    rows = [r.as_row() for r in results]
    frame = pd.DataFrame(rows)
    # Attach the exact oracle rates for direct comparison in the same table.
    exact = [exact_rates(r.p, r.q) for r in results]
    for name in RATE_COLUMNS:
        frame[f"{name}_exact"] = [e[name] for e in exact]
    return frame
