"""Build and execute the delivered notebook with nbformat/nbclient.

The notebook is a thin wrapper over the bitflip package (single
implementation shared with the CLI). Running this script regenerates the
executed notebook ``notebooks/bitflip_one_round.ipynb`` in place.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell

cells = [
    new_markdown_cell(
        """# Three-qubit bit-flip code: one correction round

**Research question.** How often does three-qubit repetition decoding fail
when physical bits flip with probability $p$ *and* the two reported parity
checks each flip with probability $q$?

This notebook is a thin wrapper over the `bitflip` package - the **same
implementation** the CLI (`python -m bitflip simulate`) uses, with the same
default seed. Re-running it top to bottom reproduces `results/sweep.csv`
and `results/fig_sweep.png` exactly.

**Model.** Logical $|0\\rangle$ is encoded in 000 (three data bits). Each
data bit flips independently with probability $p$. The two adjacent parities
$s_1 = q_1 \\oplus q_2$, $s_2 = q_2 \\oplus q_3$ are measured ideally, but each
*reported* bit flips independently with probability $q$. The decoder applies
the correction selected by the **reported** syndrome
($10\\to q_1$, $11\\to q_2$, $01\\to q_3$, $00\\to$ nothing), then an ideal
majority readout classifies the logical bit. Gates, state preparation and
the final readout are ideal; noise is only in X errors and parity reports.

**Predictions to verify below** (derived in the README from the 32-event
enumeration):

| Observable | Analytic law |
|---|---|
| Clean logical failure | $L_0 = 3p^2 - 2p^3$ |
| Noisy final majority failure | $L_0 + (4q - 2q^2)\\,p(1-p)(1-2p)$ |
| Any residual physical X | $1 - (1-q)^2(1-L_0)$ |
| Block outside code space | $1 - (1-q)^2$ |

At $p=0.1$: unencoded 10%, clean 2.8%, noisy ($q=0.02$) 3.37024%."""
    ),
    new_code_cell(
        """import numpy as np
import pandas as pd
from pathlib import Path

from bitflip.enumeration import clean_logical_failure, event_table, exact_rates
from bitflip.simulation import RATE_COLUMNS, run_sweep, sweep_frame, shots_for_point
from bitflip.cli import DEFAULT_SEED, P_CHECK, P_SWEEP, Q_VALUES
from bitflip.plotting import make_figure

SEED = DEFAULT_SEED  # 20261122, identical to the CLI default
# Locate the repository root (folder containing pyproject.toml) so the
# notebook writes the same results/ tree whether it is executed from the
# project root or from notebooks/.
_here = Path.cwd()
while not (_here / "pyproject.toml").exists() and _here != _here.parent:
    _here = _here.parent
RESULTS = _here / "results"
RESULTS.mkdir(exist_ok=True)
pd.set_option("display.precision", 5)"""
    ),
    new_markdown_cell(
        """## 1. The 32-event exact oracle

Eight physical error masks × four report-noise masks = 32 mutually exclusive
events, each weighted by its exact probability. The table below shows the
events at $p=0.1$, $q=0.02$; the correction column is the decoder table
applied to the *reported* syndrome."""
    ),
    new_code_cell(
        """table = event_table(0.1, 0.02)
table[["error_mask", "report_noise", "true_syndrome", "reported_syndrome",
       "correction", "final_block", "weight", "logical_bit", "prob"]]"""
    ),
    new_markdown_cell(
        """### Why a classical mask simulation reproduces the logical X statistics

Within the code space, basis states are the pair $\\{|000\\rangle, |111\\rangle\\}$
and X errors act as classical XOR masks on the bit labels, so every
amplitude is moved by exactly the same mask. A superposition
$\\alpha|000\\rangle + \\beta|111\\rangle$ under error mask $m$ becomes
$\\alpha|m\\oplus 000\\rangle + \\beta|m \\oplus 111\\rangle$ - the *relative*
amplitudes never mix, so tracking masks per shot reproduces the logical X
statistics of this stochastic Pauli channel without simulating amplitudes.

The parity checks are measured, not computed by reading data bits: measuring
the two stabilizers $Z_1Z_2$, $Z_2Z_3$ projects onto syndrome eigenspaces and
cannot reveal $\\alpha, \\beta$ - both $|000\\rangle$ and $|111\\rangle$ sit in the
same syndrome sector for every syndrome value. (Treating individual data-bit
readings as parities would *not* be nondestructive; that is why the model
uses ideal stabilizer reports with classical report noise instead.)

### The clean failure law from the eight masks

With truthful reports ($q=0$) the decoder removes any single flip exactly.
The corrected block's Hamming weight equals the input weight, so the logical
bit is wrong iff **two or three** data qubits flipped:

$$L_0 = \\binom{3}{2}p^2(1-p) + p^3 = 3p^2 - 2p^3.$$

At $p=0.1$: $L_0 = 0.028$ vs 0.1 unencoded; at $p=0.5$ both equal 0.5 - no
advantage, the code cannot help when errors are as likely as not."""
    ),
    new_code_cell(
        """anchor = exact_rates(0.1, 0.02)
predicted = {
    "logical_failure (clean q=0)": clean_logical_failure(0.1),
    "logical_failure (q=0.02)": 0.028 + (4*0.02 - 2*0.02**2)*0.1*0.9*0.8,
    "any_residual_x (q=0.02)": 1 - (1-0.02)**2 * (1 - clean_logical_failure(0.1)),
    "outside_code_space (q=0.02)": 1 - (1-0.02)**2,
}
print("Oracle at p=0.1, q=0.02 vs analytic predictions:")
for k, v in predicted.items():
    key = k.split(" (")[0]
    print(f"  {k:32s} oracle {anchor[key]:.7f}  predicted {v:.7f}  match {abs(anchor[key]-v) < 1e-12}")"""
    ),
    new_markdown_cell(
        """## 2. Seeded Monte Carlo sweep

Adaptive shot budget: at least 100,000 shots per point, raised toward a
target of 200 expected logical failures where failures are rare (max 1M).
Every point gets its own deterministic seed derived from the master seed, so
results do not depend on sweep order. Zero observed failures are reported
through the finite Wilson 95% upper bound rather than "0 ± 0"."""
    ),
    new_code_cell(
        """results = run_sweep(P_SWEEP + P_CHECK, Q_VALUES, shots=None, seed=SEED)
frame = sweep_frame(results)
frame.to_csv(RESULTS / "sweep.csv", index=False)
print(f"{len(frame)} points, seed {SEED}")
frame[["p", "q", "shots", "logical_failure", "logical_failure_exact",
       "any_residual_x", "any_residual_x_exact"]].query("p in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]")"""
    ),
    new_markdown_cell(
        """### Monte Carlo vs exact: global statistical criterion

We do **not** gate each point on its own 95% interval - with 31 points that
fails by chance on ~1 point in 20. Instead each point contributes a
z-score $(\\hat{L} - L)/\\sigma$ and we require $\\max|z| < 4$ and reduced
$\\chi^2 = \\overline{z^2} < 2.5$ across all points, documented in the README
and enforced in the test suite."""
    ),
    new_code_cell(
        """sweep = frame[frame.p <= 0.31]
z = (sweep["logical_failure"] - sweep["logical_failure_exact"]) / np.sqrt(
    sweep["logical_failure_exact"] * (1 - sweep["logical_failure_exact"]) / sweep["shots"]
)
print(f"max |z| = {z.abs().max():.2f}   reduced chi^2 = {(z**2).mean():.2f}   "
      f"points = {len(z)}")
assert z.abs().max() < 4.0 and (z**2).mean() < 2.5
print("Global statistical criterion passed")"""
    ),
    new_code_cell(
        """make_figure(RESULTS / "sweep.csv", RESULTS / "fig_sweep.png")
print("Figure written to results/fig_sweep.png")"""
    ),
    new_markdown_cell(
        """## 3. Zero-failure handling

At $p = 0$ (and small $p$ with unlucky draws) zero failures may be observed.
The point estimate is 0 but the Wilson interval keeps a finite upper bound,
so the CSV never reads "0 ± 0" - the interval is the honest statement."""
    ),
    new_code_cell(
        """zero = frame[frame.p == 0.0]
zero[["p", "q", "logical_failure", "logical_failure_count",
      "logical_failure_ci_low", "logical_failure_ci_high"]]"""
    ),
    new_markdown_cell(
        """## 4. What one round of this model establishes - and what it leaves out

**Established.** For independent X errors with independent parity-report
flips, majority decoding with truthful reports reduces failure from
$\\sim p$ to $\\sim 3p^2$; readout noise $q$ adds the term
$(4q-2q^2)p(1-p)(1-2p)$, which vanishes at $p=0$, $p=1$ and $p=0.5$, and a
wrong report can leave a residual physical error *while* the logical bit
still reads correctly - the two observables are genuinely different.

**Left out.** Phase (Z) errors, gate and state-preparation noise (the model
is phenomenological, not circuit-level), repeated syndrome extraction across
rounds, and decoder statefulness. An arbitrary-state fidelity or a
fault-tolerance threshold claim would need its own definition and
experiment; the measured observable here is the failure probability of an
encoded **computational-basis bit** under ideal final majority readout."""
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
client = NotebookClient(nb, timeout=1200, kernel_name="python3")
client.execute()
out = "notebooks/bitflip_one_round.ipynb"
Path("notebooks").mkdir(exist_ok=True)
nbformat.write(nb, out)
print(f"Executed notebook written to {out}")
