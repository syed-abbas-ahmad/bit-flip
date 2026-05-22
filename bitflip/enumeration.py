"""Exact weighted enumeration of the 32 possible events in one round.

Eight physical error masks (probabilities from independent X errors with
probability p per qubit) times four reported-parity noise masks
(independent flips of the two reports with probability q each) give 32
mutually exclusive events. Each event is weighted by its exact probability
and carried through the correction table to the final three-bit block.

This module is a self-contained oracle: it imports nothing from the
simulator, so Monte Carlo results can be checked against it independently.
"""

from __future__ import annotations

import itertools

import pandas as pd

from .model import apply_correction, correction_for, majority, true_syndrome


def event_table(p: float, q: float) -> pd.DataFrame:
    """Exact probability of every (error mask, report-noise mask) event.

    Columns: ``error_mask`` (3 bits as a string, qubit 1 first),
    ``report_noise`` (2 bits as a string), ``true_syndrome``,
    ``reported_syndrome``, ``correction``, ``final_block``,
    ``logical_bit`` (majority of the final block), ``weight``
    (Hamming weight of the final block), ``prob``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")

    rows = []
    for err in itertools.product((0, 1), repeat=3):
        p_err = 1.0
        for bit in err:
            p_err *= p if bit else (1.0 - p)
        syn = true_syndrome(err)
        for noise in itertools.product((0, 1), repeat=2):
            p_noise = 1.0
            for bit in noise:
                p_noise *= q if bit else (1.0 - q)
            reported = (syn[0] ^ noise[0], syn[1] ^ noise[1])
            corr = correction_for(reported)
            final = apply_correction(err, reported)
            weight = sum(final)
            rows.append(
                {
                    "error_mask": "".join(map(str, err)),
                    "report_noise": "".join(map(str, noise)),
                    "true_syndrome": "".join(map(str, syn)),
                    "reported_syndrome": "".join(map(str, reported)),
                    "correction": "".join(map(str, corr)),
                    "final_block": "".join(map(str, final)),
                    "weight": weight,
                    "logical_bit": majority(final),
                    "outside_code_space": int(
                        tuple(final) not in ((0, 0, 0), (1, 1, 1))
                    ),
                    "prob": p_err * p_noise,
                }
            )
    return pd.DataFrame(rows)


def exact_rates(p: float, q: float) -> dict[str, float]:
    """Aggregate the 32-event table into the reported observables.

    ``logical_failure``: ideal majority of the corrected block returns the
    wrong logical bit (weight >= 2; an all-ones block is a logical X inside
    the code space).

    ``any_residual_x``: any physical X remains relative to the input
    (weight >= 1).

    ``outside_code_space``: the corrected block is neither 000 nor 111.

    ``unencoded_failure``: a single unencoded bit flips with probability p.
    """
    table = event_table(p, q)

    def clipped(value: float) -> float:
        # Guard against floating-point drift slightly outside [0, 1].
        return float(min(1.0, max(0.0, value)))

    return {
        "logical_failure": clipped(table.loc[table["weight"] >= 2, "prob"].sum()),
        "any_residual_x": clipped(table.loc[table["weight"] >= 1, "prob"].sum()),
        "outside_code_space": clipped(table.loc[table["outside_code_space"] == 1, "prob"].sum()),
        "unencoded_failure": p,
    }


def clean_logical_failure(p: float) -> float:
    """Analytic clean law L0 = 3p^2 - 2p^3 (two or three physical flips)."""
    return 3.0 * p * p - 2.0 * p**3
