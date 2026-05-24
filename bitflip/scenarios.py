"""Verified scenario records that drive the walkthrough animation.

Each scenario is *derived* by running the actual correction-table code, not
hardcoded: the residuals and outcomes below are computed through
``model.apply_correction`` and checked against the 32-event oracle by the
test suite (``tests/test_pipeline.py``).
"""

from __future__ import annotations

from .model import apply_correction, correction_for, majority, true_syndrome


def _build(error_mask, reported, noise_bit, readout_q):
    """Derive every downstream quantity from (error mask, reported syndrome)."""
    err = list(error_mask)
    rep = tuple(reported)
    syn = true_syndrome(err)
    corr = correction_for(rep)
    final = apply_correction(err, rep)
    weight = sum(final)
    logical = majority(final)
    inside = tuple(final) in ((0, 0, 0), (1, 1, 1))
    if logical == 1:
        outcome = "logical_failure"
    elif weight >= 1:
        outcome = "residual_only"
    else:
        outcome = "clean_success"
    return {
        "error_mask": err,
        "true_syndrome": syn,
        "reported_syndrome": rep,
        "noise_bit": noise_bit,          # which reported parity was flipped, or None
        "readout_q": readout_q,
        "correction_qubit": next((i for i, c in enumerate(corr) if c), None),
        "final_block": final,
        "weight": weight,
        "logical_bit": logical,
        "outside_code_space": not inside,
        "outcome": outcome,
    }


def scenarios() -> dict[str, dict]:
    """The three walkthrough scenarios shown in animation A.

    ``clean``         single physical error, perfect readout: full correction.
    ``misleading``    single physical error (mask 100), true syndrome 10, the
                      second report flips so the decoder corrects qubit 2 and
                      leaves residual 110: majority reads 1, a logical failure.
    ``false_report``  no physical error, one flipped report: the correction
                      itself injects a physical X (residual 001) yet the ideal
                      majority still returns the right logical bit.
    """
    return {
        "clean": _build((0, 1, 0), (1, 1), None, 0.0),
        "misleading": _build((1, 0, 0), (1, 1), 1, 0.02),
        "false_report": _build((0, 0, 0), (0, 1), 1, 0.02),
    }
