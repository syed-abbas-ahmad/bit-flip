"""Publication-quality comparison figure from the sweep records.

Colors are Okabe-Ito colorblind-safe and shared with the animations:
unencoded = gray, clean (q=0) = blue, noisy (q=0.02) = vermillion.
Curves carry text labels in addition to color.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .simulation import RATE_COLUMNS

#: Shared, colorblind-safe (Okabe-Ito) palette.
COLOR_UNENCODED = "#999999"
COLOR_CLEAN = "#0072B2"
COLOR_NOISY = "#D55E00"

LOGICAL = "logical_failure"

#: Columns every sweep CSV must contain for the figure and animations.
REQUIRED_CSV_COLUMNS = (
    "p", "q", "shots",
    "logical_failure", "logical_failure_exact",
    "logical_failure_ci_low", "logical_failure_ci_high",
    "any_residual_x", "any_residual_x_exact",
    "unencoded_failure_exact",
)


def validate_sweep_csv(frame: pd.DataFrame) -> None:
    """Raise a clear error if the frame cannot support the figure/movies.

    The published schema is the CSV written by ``bitflip simulate``: both q
    values (0 and 0.02) across a common p grid, the ``shots`` column, the
    sampled rates with Wilson bounds and the exact oracle rates. Partial
    schemas (a single q value, missing exact columns) are rejected here
    instead of failing later with an IndexError deep inside rendering.
    """
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"sweep CSV is missing required columns: {missing}; this schema is "
            "produced by 'python -m bitflip simulate'"
        )
    for q, name in ((0.0, "q = 0"), (0.02, "q = 0.02")):
        if not np.isclose(frame["q"], q).any():
            raise ValueError(
                f"sweep CSV contains no rows for {name}; both q values are "
                "required (produced by 'python -m bitflip simulate' without "
                "filtering)"
            )


def make_figure(csv_path, out_path: str) -> plt.Figure:
    """Draw the sweep figure from the CSV written by ``bitflip simulate``.

    Main panel: logical failure probability versus p for q=0 and q=0.02,
    exact curves, Monte Carlo estimates with Wilson 95% intervals, and the
    unencoded baseline. Inset panel: the p=0.5 boundary check.
    """
    frame = pd.read_csv(csv_path)
    validate_sweep_csv(frame)
    sweep = frame[np.isclose(frame["p"], frame["p"].round(2)) & (frame["p"] <= 0.31)]
    check = frame[np.isclose(frame["p"], 0.5)]

    fig, (ax, ax_check) = plt.subplots(
        1, 2, figsize=(10.0, 5.8), gridspec_kw={"width_ratios": [3.2, 1.0]}
    )

    p = sweep["p"].to_numpy()
    ax.plot(p, sweep["unencoded_failure_exact"], linestyle="--", color=COLOR_UNENCODED,
            linewidth=1.8, label="Unencoded (fails with prob. p)")
    clean = sweep[sweep["q"] == 0.0]
    noisy = sweep[sweep["q"] == 0.02]
    ax.plot(clean["p"], clean[f"{LOGICAL}_exact"], color=COLOR_CLEAN, linewidth=1.8,
            label="Encoded, exact, q = 0")
    ax.plot(noisy["p"], noisy[f"{LOGICAL}_exact"], color=COLOR_NOISY, linewidth=1.8,
            label="Encoded, exact, q = 0.02")

    # Monte Carlo estimates with Wilson 95% intervals (subset for readability).
    for data, color, marker, label in (
        (clean, COLOR_CLEAN, "o", "Encoded, Monte Carlo, q = 0"),
        (noisy, COLOR_NOISY, "s", "Encoded, Monte Carlo, q = 0.02"),
    ):
        sub = data[data["p"] > 0]
        y = sub[f"{LOGICAL}"].to_numpy()
        lo = sub[f"{LOGICAL}_ci_low"].to_numpy()
        hi = sub[f"{LOGICAL}_ci_high"].to_numpy()
        ax.errorbar(sub["p"], y, yerr=[y - lo, hi - y], fmt=marker, color=color,
                    markersize=3.4, elinewidth=0.9, capsize=1.6, linestyle="none",
                    alpha=0.85, label=label)

    # Labels on the curves, not just color. Every arrow target comes from
    # the same data row as its x coordinate, so annotations touch the curve
    # they name.
    ax.annotate("unencoded: p", xy=(0.245, 0.245), xytext=(0.155, 0.20),
                color=COLOR_UNENCODED, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=COLOR_UNENCODED, lw=1.0))
    anchor_row = clean[np.isclose(clean["p"], 0.21)].iloc[0]
    ax.annotate("q = 0: 3p² − 2p³",
                xy=(float(anchor_row["p"]), float(anchor_row[f"{LOGICAL}_exact"])),
                xytext=(0.215, 0.10), color=COLOR_CLEAN, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=COLOR_CLEAN, lw=1.0))
    noisy_row = noisy[np.isclose(noisy["p"], 0.30)].iloc[0]
    ax.annotate("q = 0.02",
                xy=(float(noisy_row["p"]), float(noisy_row[f"{LOGICAL}_exact"])),
                xytext=(0.255, 0.032), color=COLOR_NOISY, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=COLOR_NOISY, lw=1.0))

    row = clean[np.isclose(clean["p"], 0.1)].iloc[0]
    ax.annotate(
        "p = 0.1:  unencoded 10%\n"
        f"encoded, q = 0: 2.8%\n"
        f"encoded, q = 0.02: {noisy[np.isclose(noisy['p'], 0.1)][f'{LOGICAL}_exact'].iloc[0]:.2%}",
        xy=(0.1, row[f"{LOGICAL}"]), xytext=(0.135, 0.004),
        fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#cccccc"),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
    )

    ax.set_xlim(-0.005, 0.315)
    ax.set_ylim(-0.008, 0.33)
    ax.set_xlabel("Physical bit-flip probability p")
    ax.set_ylabel("Logical failure probability (one round)")
    ax.set_title("Three-qubit repetition code vs unencoded bit")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.25)

    # Boundary check at p = 0.5: code and unencoded must agree at 50%.
    # Optional: CSVs written with --no-check have no p = 0.5 rows, and a
    # partial CSV might have only one of the two q values.
    c05 = check[check["q"] == 0.0]
    n05 = check[check["q"] == 0.02]
    has_check = len(c05) > 0 and len(n05) > 0
    if has_check:
        xs = np.arange(3)
        labels = ["unencoded", "q = 0", "q = 0.02"]
        vals = [
            float(check[check["q"] == 0.0]["unencoded_failure"].iloc[0]),
            float(check[check["q"] == 0.0][LOGICAL].iloc[0]),
            float(check[check["q"] == 0.02][LOGICAL].iloc[0]),
        ]
        errs = [
            [float(check[check["q"] == 0.0]["unencoded_failure"].iloc[0] - check[check["q"] == 0.0]["unencoded_failure_ci_low"].iloc[0]),
             float(check[check["q"] == 0.0][LOGICAL].iloc[0] - check[check["q"] == 0.0][f"{LOGICAL}_ci_low"].iloc[0]),
             float(check[check["q"] == 0.02][LOGICAL].iloc[0] - check[check["q"] == 0.02][f"{LOGICAL}_ci_low"].iloc[0])],
            [float(check[check["q"] == 0.0]["unencoded_failure_ci_high"].iloc[0] - check[check["q"] == 0.0]["unencoded_failure"].iloc[0]),
             float(check[check["q"] == 0.0][f"{LOGICAL}_ci_high"].iloc[0] - check[check["q"] == 0.0][LOGICAL].iloc[0]),
             float(check[check["q"] == 0.02][f"{LOGICAL}_ci_high"].iloc[0] - check[check["q"] == 0.02][LOGICAL].iloc[0])],
        ]
        ax_check.bar(xs, vals, yerr=errs, capsize=3,
                     color=[COLOR_UNENCODED, COLOR_CLEAN, COLOR_NOISY])
        ax_check.axhline(0.5, color="black", linewidth=0.8, linestyle=":")
        ax_check.set_xticks(xs, labels, fontsize=8, rotation=20)
        ax_check.set_ylim(0, 0.62)
        ax_check.set_title("p = 0.5 check", fontsize=10)
        ax_check.text(0.5, 0.545, "both 0.5", ha="center", fontsize=9)
        ax_check.grid(alpha=0.25, axis="y")
    else:
        ax_check.axis("off")
        ax_check.text(0.5, 0.5, "p = 0.5 boundary check\nnot present in this data\n"
                      "(run simulate without --no-check)",
                      ha="center", va="center", fontsize=9, color="#777777")

    shots_min = int(frame["shots"].min())
    shots_max = int(frame["shots"].max())
    if shots_min == shots_max:
        shots_txt = (f"Seeded Monte Carlo, fixed budget {shots_min:,} shots per "
                     "point; Wilson 95% intervals")
    else:
        shots_txt = (
            f"Seeded Monte Carlo, adaptive budget {shots_min:,}-{shots_max:,} shots "
            "per point; Wilson 95% intervals"
        )
    fig.text(0.01, 0.01, shots_txt, fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return fig
