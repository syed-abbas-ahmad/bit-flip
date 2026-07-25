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


def test_apply_correction_validates_mask_shape_and_bits():
    with pytest.raises(ValueError):
        apply_correction((1, 0), (0, 0))       # wrong length
    with pytest.raises(ValueError):
        apply_correction((1, 0, 2), (0, 0))    # non-binary bit
    assert apply_correction((1, 0, 0), (1, 0)) == [0, 0, 0]


def test_weight_two_error_with_truthful_reports_leaves_undetected_logical_x():
    # 110 has the same parity reports as a single error on qubit 3: the
    # decoder flips qubit 3 and produces 111. The final syndrome is zero,
    # yet a logical X remains - the block is inside the code space but the
    # physical errors were NOT removed. This is why "restores the code
    # space" and "removes residual physical errors" are different claims.
    mask = (1, 1, 0)
    syn = true_syndrome(mask)
    assert syn == (0, 1)  # same reports as a single error on qubit 3
    corrected = apply_correction(mask, syn)
    assert corrected == [1, 1, 1]
    assert true_syndrome(corrected) == (0, 0)  # zero syndrome afterwards
    assert logical_outcome(mask, syn) == 1     # undetected logical X


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


def test_error_shifted_codewords_share_the_mask_syndrome():
    # A fixed error mask m moves both codewords to m XOR codeword, and both
    # shifted words carry the syndrome of m. For mask 100 the components
    # become 100 and 011, both with syndrome 10: the stabilizer measurement
    # projects onto that sector without revealing the amplitudes.
    for mask in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                 (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]:
        shifted_a = tuple(m ^ c for m, c in zip(mask, (0, 0, 0)))
        shifted_b = tuple(m ^ c for m, c in zip(mask, (1, 1, 1)))
        assert true_syndrome(shifted_a) == true_syndrome(mask)
        assert true_syndrome(shifted_b) == true_syndrome(mask)
    assert true_syndrome((1, 0, 0)) == (1, 0)
    assert true_syndrome((0, 1, 1)) == (1, 0)


def test_truthful_correction_preserves_the_majority_bit_not_the_weight():
    # Clean correction does NOT preserve Hamming weight: 010 corrects to
    # 000 (weight 1 -> 0) and 110 corrects to 111 (weight 2 -> 3). What it
    # preserves is the majority bit. The four input-weight cases:
    #   weight 0 -> corrected 000, majority 0
    #   weight 1 -> corrected 000, majority 0 (single error removed)
    #   weight 2 -> corrected 111, majority 1 (logical failure)
    #   weight 3 -> corrected 111, majority 1 (logical X inside code space)
    expected = {0: ([0, 0, 0], 0), 1: ([0, 0, 0], 0),
                2: ([1, 1, 1], 1), 3: ([1, 1, 1], 1)}
    for mask in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                 (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]:
        corrected = apply_correction(mask, true_syndrome(mask))
        weight, outcome = sum(mask), logical_outcome(mask, true_syndrome(mask))
        assert corrected == expected[weight][0]
        assert outcome == expected[weight][1]
