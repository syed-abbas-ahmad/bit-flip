"""Verified reference anchors, each bound to its own (p, q, observable).

Every anchor states which simulation setting it refers to, so a binding
mistake (for example comparing the clean prediction against the noisy
oracle) fails loudly instead of printing a mismatch. The delivered notebook
and the test suite both run ``check_anchors``; ``swapped_anchors`` exists so
tests can prove the comparison is actually discriminative.
"""

from __future__ import annotations

from .enumeration import exact_rates

#: Tolerance for exact floating-point anchor comparisons.
ANCHOR_TOL = 1e-9

ANCHORS: list[dict] = [
    {
        "label": "logical_failure (clean q=0)",
        "p": 0.1, "q": 0.0,
        "observable": "logical_failure",
        "expected": 0.028,
    },
    {
        "label": "logical_failure (q=0.02)",
        "p": 0.1, "q": 0.02,
        "observable": "logical_failure",
        "expected": 0.0337024,
    },
    {
        "label": "any_residual_x (q=0.02)",
        "p": 0.1, "q": 0.02,
        "observable": "any_residual_x",
        "expected": 0.0664912,
    },
    {
        "label": "outside_code_space (q=0.02)",
        "p": 0.1, "q": 0.02,
        "observable": "outside_code_space",
        "expected": 0.0396,
    },
]


def check_anchors(anchors: list[dict] | None = None,
                  tol: float = ANCHOR_TOL) -> list[dict]:
    """Compare every anchor against the oracle at its own (p, q, observable).

    Returns one row per anchor with ``observed``, ``expected`` and ``match``;
    the caller decides whether to assert or display them.
    """
    rows = []
    for anchor in (anchors if anchors is not None else ANCHORS):
        observed = exact_rates(anchor["p"], anchor["q"])[anchor["observable"]]
        rows.append({
            "label": anchor["label"],
            "p": anchor["p"],
            "q": anchor["q"],
            "observable": anchor["observable"],
            "observed": observed,
            "expected": anchor["expected"],
            "match": abs(observed - anchor["expected"]) <= tol,
        })
    return rows


def swapped_anchors() -> list[dict]:
    """Deliberately wrong bindings: the clean row is compared against the
    noisy oracle (the mistake the delivered notebook once printed). Used by
    the test suite to prove the comparison fails programmatically."""
    swapped = [dict(a) for a in ANCHORS]
    for row in swapped:
        if row["q"] == 0.0:
            row["q"] = 0.02
    return swapped
