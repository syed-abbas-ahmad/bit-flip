"""Three-qubit bit-flip code: classical-mask model of one correction round.

Qubits are labelled 1, 2, 3 (array indices 0, 1, 2). The logical state is a
superposition of ``000`` and ``111``; only X (bit-flip) errors are modelled.
The two adjacent parity checks are measured in the fixed order

    s1 = q1 XOR q2,   s2 = q2 XOR q3

and the correction table for the *reported* syndrome (r1, r2) is

    (0, 0) -> no correction,  (1, 0) -> qubit 1,
    (1, 1) -> qubit 2,        (0, 1) -> qubit 3.

A single round means: physical flips, parity reports (each independently
flipped with probability q), correction from the reported syndrome, then an
ideal majority readout of the corrected block.
"""

from __future__ import annotations

N_QUBITS = 3
N_PARITIES = 2

#: Reported syndrome (r1, r2) -> qubit index to correct (0-based), or None.
CORRECTION_TABLE: dict[tuple[int, int], int | None] = {
    (0, 0): None,
    (1, 0): 0,
    (1, 1): 1,
    (0, 1): 2,
}


def true_syndrome(mask) -> tuple[int, int]:
    """Adjacent parity checks (q1 XOR q2, q2 XOR q3) of a 3-bit mask."""
    m = list(mask)
    if len(m) != N_QUBITS:
        raise ValueError(f"mask must have {N_QUBITS} bits, got {len(m)}")
    return (m[0] ^ m[1], m[1] ^ m[2])


def correction_for(reported: tuple[int, int]) -> list[int]:
    """Correction mask selected by the reported syndrome."""
    if tuple(reported) not in CORRECTION_TABLE:
        raise ValueError(f"invalid reported syndrome {reported}")
    qubit = CORRECTION_TABLE[tuple(reported)]
    corr = [0, 0, 0]
    if qubit is not None:
        corr[qubit] = 1
    return corr


def apply_correction(error_mask, reported) -> list[int]:
    """XOR the table-selected correction into the error mask.

    The result is the corrected block. Its Hamming weight decides the
    outcome: 0 or 3 is inside the code space (an all-ones block is a logical
    X, i.e. a logical bit flip), weight 1 or 2 is outside the code space.
    """
    err = list(error_mask)
    if len(err) != N_QUBITS:
        raise ValueError(f"error mask must have {N_QUBITS} bits, got {len(err)}")
    if any(bit not in (0, 1) for bit in err):
        raise ValueError(f"error mask bits must be 0 or 1, got {err}")
    corr = correction_for(reported)
    return [int(e) ^ c for e, c in zip(err, corr)]


def majority(bits) -> int:
    """Ideal majority readout of a 3-bit block: 1 iff at least two bits are 1."""
    bits = list(bits)
    if len(bits) != N_QUBITS:
        raise ValueError(f"block must have {N_QUBITS} bits, got {len(bits)}")
    return int(sum(bits) >= 2)


def logical_outcome(error_mask, reported) -> int:
    """Logical bit after correction and ideal majority readout.

    The input is treated as the error mask on logical |0>, so the majority of
    the corrected block is the logical bit (a logical X failure iff it is 1).
    """
    return majority(apply_correction(error_mask, reported))
