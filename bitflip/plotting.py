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


def make_figure(csv_path, out_path: str) -> plt.Figure:
    """Draw the sweep figure from the CSV written by ``bitflip simulate``.

    Main panel: logical failure probability versus p for q=0 and q=0.02,
    exact curves, Monte Carlo estimates with Wilson 95% intervals, and the
    unencoded baseline. Inset panel: the p=0.5 boundary check.
    """
    frame = pd.read_csv(csv_path)
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

    # Labels on the curves, not just color.
    ax.annotate("unencoded: p", xy=(0.245, 0.245), xytext=(0.155, 0.20),
                color=COLOR_UNENCODED, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=COLOR_UNENCODED, lw=1.0))
    ax.annotate("q = 0: 3p² − 2p³", xy=(0.21, clean[f"{LOGICAL}_exact"].iloc[-1]),
                xytext=(0.215, 0.10), color=COLOR_CLEAN, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=COLOR_CLEAN, lw=1.0))
    ax.annotate("q = 0.02", xy=(0.30, noisy[f"{LOGICAL}_exact"].iloc[-1]),
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
    # Optional: CSVs written with --no-check have no p = 0.5 rows.
    has_check = len(check) > 0 and set(np.isclose(check["p"], 0.5))
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
    shots_txt = (
        f"Seeded Monte Carlo, adaptive budget {shots_min:,}-{shots_max:,} shots "
        "per point; Wilson 95% intervals"
    )
    fig.text(0.01, 0.01, shots_txt, fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return fig
