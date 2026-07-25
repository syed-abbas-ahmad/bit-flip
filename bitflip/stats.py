"""Release statistical gate: one documented definition over all rows.

The gate described in the README and applied in the notebook is implemented
here so that both use exactly the same population and convention:

- every row of the CSV contributes a z-score ``(MC - exact) / sigma``;
- ``sigma`` is the exact binomial standard deviation floored at 1e-15, so
  deterministic rows (p = 0, which cannot fail) contribute ``z = 0``;
- deterministic rows are additionally *verified explicitly* (zero observed
  failures), never silently dropped;
- the gate requires ``max|z| < z_limit`` and reduced chi-square
  ``mean(z^2) < chi2_limit`` over the whole population.

Because the gate reads every row, mutating any row - including the p = 0.5
boundary rows - trips it. The test suite proves this with negative
controls on both a boundary row and an in-sweep row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Numerical floor for the exact binomial sigma.
SIGMA_FLOOR = 1e-15

#: Default gate thresholds (see README "Statistical tolerance").
Z_LIMIT = 4.0
CHI2_LIMIT = 2.5


def z_scores(frame: pd.DataFrame) -> pd.Series:
    """z-scores for every row, indexed like ``frame``.

    ``sigma`` uses the exact binomial rate floored at ``SIGMA_FLOOR``, so
    deterministic rows (exact rate 0) contribute ``z = 0`` instead of NaN.
    """
    exact = frame["logical_failure_exact"].to_numpy(dtype=float)
    sigma = np.sqrt(
        np.maximum(exact * (1.0 - exact), SIGMA_FLOOR)
        / frame["shots"].to_numpy(dtype=float)
    )
    z = (frame["logical_failure"].to_numpy(dtype=float) - exact) / sigma
    return pd.Series(z, index=frame.index)


def gate_report(frame: pd.DataFrame, z_limit: float = Z_LIMIT,
                chi2_limit: float = CHI2_LIMIT) -> dict:
    """Evaluate the release gate over **all** rows of ``frame``.

    Returns a report with the named populations and both required
    statistics; ``passed`` is False if any deterministic row shows failures,
    any z-score breaches the limit, or the reduced chi-square breaches its
    limit.
    """
    if frame.empty:
        raise ValueError("empty sweep frame")

    deterministic = frame[np.isclose(frame["p"], 0.0)]
    deterministic_ok = bool(
        (deterministic["logical_failure_count"] == 0).all()
    ) if len(deterministic) else True

    z = z_scores(frame)
    max_abs_z = float(z.abs().max())
    chi2_red = float((z**2).mean())

    return {
        "population": "all rows",
        "n_rows": int(len(frame)),
        "n_deterministic": int(len(deterministic)),
        "deterministic_ok": deterministic_ok,
        "max_abs_z": max_abs_z,
        "z_limit": z_limit,
        "chi2_red": chi2_red,
        "chi2_limit": chi2_limit,
        "passed": bool(
            deterministic_ok
            and max_abs_z < z_limit
            and chi2_red < chi2_limit
        ),
    }


def population_report(frame: pd.DataFrame) -> dict:
    """Chi-square for the documented subset populations, computed (never
    hard-coded), for transparency next to the release gate."""
    non_deterministic = frame[~np.isclose(frame["p"], 0.0)]

    def chi2(rows: pd.DataFrame) -> float:
        if rows.empty:
            return float("nan")
        z = z_scores(rows)
        return float((z**2).mean())

    return {
        "all_rows": {"n": int(len(frame)), "chi2_red": chi2(frame)},
        "excluding_deterministic_rows": {
            "n": int(len(non_deterministic)),
            "chi2_red": chi2(non_deterministic),
        },
    }
