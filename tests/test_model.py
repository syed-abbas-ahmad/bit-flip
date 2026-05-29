"""Decoder table, syndrome order and majority readout."""

import pytest

from bitflip.model import (
    apply_correction,
    correction_for,
    logical_outcome,
    majority,
    true_syndrome,
)


def test_syndrome_order():
    # s1 = q1 XOR q2, s2 = q2 XOR q3, in that fixed order.
    assert true_syndrome((0, 0, 0)) == (0, 0)
    assert true_syndrome((1, 0, 0)) == (1, 0)
    assert true_syndrome((0, 1, 0)) == (1, 1)
    assert true_syndrome((0, 0, 1)) == (0, 1)
    assert true_syndrome((1, 1, 1)) == (0, 0)


def test_correction_table():
    assert correction_for((0, 0)) == [0, 0, 0]
    assert correction_for((1, 0)) == [1, 0, 0]
    assert correction_for((1, 1)) == [0, 1, 0]
    assert correction_for((0, 1)) == [0, 0, 1]
    with pytest.raises(ValueError):
        correction_for((2, 0))


def test_all_single_errors_corrected_exactly():
    # Every single-qubit error with a truthful report must return to 000.
    for i in range(3):
        mask = [0, 0, 0]
        mask[i] = 1
        syn = true_syndrome(mask)
        assert apply_correction(mask, syn) == [0, 0, 0]


def test_all_ones_mask_is_logical_x_inside_code_space():
    # 111 has zero syndrome: no correction, majority reads 1 -> logical X.
    assert true_syndrome((1, 1, 1)) == (0, 0)
    assert apply_correction((1, 1, 1), (0, 0)) == [1, 1, 1]
    assert logical_outcome((1, 1, 1), (0, 0)) == 1


def test_majority_boundaries():
    assert majority((0, 0, 0)) == 0
    assert majority((1, 0, 0)) == 0
    assert majority((1, 1, 0)) == 1
    assert majority((1, 1, 1)) == 1
    with pytest.raises(ValueError):
        majority((1, 0))


def test_wrong_report_leaves_residual_and_can_fail_logically():
    # The audit-required example: mask 100, true syndrome 10, reported 11.
    # Correction lands on qubit 2, residual 110, majority reads 1: failure.
    final = apply_correction((1, 0, 0), (1, 1))
    assert final == [1, 1, 0]
    assert logical_outcome((1, 0, 0), (1, 1)) == 1


def test_bad_report_on_clean_block_injects_residual_but_majority_holds():
    # p = 0, reported (0,1): correction injects q3, residual 001,
    # ideal majority still returns the correct logical bit 0.
    final = apply_correction((0, 0, 0), (0, 1))
    assert final == [0, 0, 1]
    assert logical_outcome((0, 0, 0), (0, 1)) == 0
