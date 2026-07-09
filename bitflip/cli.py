"""Command-line interface: one implementation behind notebook and CLI.

Commands
--------
``simulate``  Run the seeded sweep and write the CSV of sampled and exact rates.
``figure``    Draw the comparison figure from the CSV.
``table``     Print the exact 32-event table at one (p, q).
``animate``   Render both animations (MP4 primary, looping GIF secondary).
``all``       Run simulate, figure and animate in sequence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .enumeration import event_table, exact_rates
from .plotting import make_figure
from .simulation import RATE_COLUMNS, run_sweep, sweep_frame

DEFAULT_SEED = 20261122
DEFAULT_CSV = Path("results/sweep.csv")
DEFAULT_FIGURE = Path("results/fig_sweep.png")
DEFAULT_RESULTS_DIR = Path("results")
P_SWEEP = [round(x, 2) for x in np.arange(0.0, 0.301, 0.01)]
P_CHECK = [0.5]
Q_VALUES = (0.0, 0.02)


def cmd_simulate(args: argparse.Namespace) -> int:
    p_values = P_SWEEP + (P_CHECK if not args.no_check else [])
    results = run_sweep(p_values, Q_VALUES, shots=args.shots, seed=args.seed)
    frame = sweep_frame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame)} points to {args.out} (seed {args.seed})")
    ref = frame[np.isclose(frame["p"], 0.1) & np.isclose(frame["q"], 0.02)].iloc[0]
    print(
        "Reference p=0.1, q=0.02: "
        f"logical {ref['logical_failure']:.5f} (exact {ref['logical_failure_exact']:.5f}), "
        f"residual {ref['any_residual_x']:.5f} (exact {ref['any_residual_x_exact']:.5f})"
    )
    return 0


def cmd_figure(args: argparse.Namespace) -> int:
    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv} (run 'bitflip simulate' first)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_figure(args.csv, args.out)
    print(f"Wrote figure to {args.out}")
    return 0


def cmd_table(args: argparse.Namespace) -> int:
    rates = exact_rates(args.p, args.q)
    table = event_table(args.p, args.q)
    with pd.option_context("display.width", 160, "display.max_rows", 40):
        print(table.drop(columns=["outside_code_space"]))
    print("\nAggregated exact rates:")
    for name, value in rates.items():
        print(f"  {name:22s} {value:.7f}")
    return 0


def cmd_animate(args: argparse.Namespace) -> int:
    from .animate import render_animation_a, render_animation_b

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv = args.csv if args.csv.exists() else None
    paths_a = render_animation_a(args.outdir, fps=args.fps)
    paths_b = render_animation_b(args.outdir, fps=args.fps, csv_path=csv)
    for path in paths_a + paths_b:
        print(f"Wrote {path}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Run every step; the CSV output doubles as the figure/animate input.

    The figure is written to ``--figure-out`` (never to the CSV path), and
    the animations read the freshly written CSV.
    """
    cmd_simulate(args)
    fig_args = argparse.Namespace(csv=args.out, out=args.figure_out)
    cmd_figure(fig_args)
    anim_args = argparse.Namespace(outdir=args.outdir, csv=args.out, fps=args.fps)
    cmd_animate(anim_args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bitflip", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run the seeded sweep and write the CSV")
    sim.add_argument("--out", type=Path, default=DEFAULT_CSV)
    sim.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sim.add_argument("--shots", type=int, default=None,
                     help="fixed shots per point (default: adaptive budget)")
    sim.add_argument("--no-check", action="store_true",
                     help="skip the p=0.5 boundary check point")
    sim.set_defaults(func=cmd_simulate)

    fig = sub.add_parser("figure", help="draw the comparison figure from the CSV")
    fig.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    fig.add_argument("--out", type=Path, default=DEFAULT_FIGURE)
    fig.set_defaults(func=cmd_figure)

    tab = sub.add_parser("table", help="print the exact 32-event table at one (p, q)")
    tab.add_argument("--p", type=float, default=0.1)
    tab.add_argument("--q", type=float, default=0.02)
    tab.set_defaults(func=cmd_table)

    anim = sub.add_parser("animate", help="render both animations (MP4 + looping GIF)")
    anim.add_argument("--outdir", type=Path, default=DEFAULT_RESULTS_DIR)
    anim.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                      help="CSV of sweep records driving the sweep animation")
    anim.add_argument("--fps", type=int, default=12)
    anim.set_defaults(func=cmd_animate)

    everything = sub.add_parser("all", help="simulate, figure and animate")
    everything.add_argument("--out", type=Path, default=DEFAULT_CSV)
    everything.add_argument("--figure-out", type=Path, default=DEFAULT_FIGURE,
                            dest="figure_out")
    everything.add_argument("--seed", type=int, default=DEFAULT_SEED)
    everything.add_argument("--shots", type=int, default=None)
    everything.add_argument("--no-check", action="store_true")
    everything.add_argument("--outdir", type=Path, default=DEFAULT_RESULTS_DIR)
    everything.add_argument("--fps", type=int, default=12)
    everything.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
