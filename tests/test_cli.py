"""CLI tests: output-path routing and an end-to-end simulate + figure run.

The `all` command once wrote the figure over the CSV because the parser had
no separate figure-output argument; these tests pin that routing down.
"""

from pathlib import Path

from bitflip.cli import DEFAULT_CSV, DEFAULT_FIGURE, build_parser, main


def test_all_parser_routes_csv_and_figure_outputs_separately():
    args = build_parser().parse_args(["all"])
    assert args.out == DEFAULT_CSV
    assert args.figure_out == DEFAULT_FIGURE
    assert args.out != args.figure_out


def test_simulate_then_figure_end_to_end(tmp_path):
    csv = tmp_path / "sweep.csv"
    png = tmp_path / "fig.png"
    assert main(["simulate", "--out", str(csv), "--shots", "20_000",
                 "--no-check", "--seed", "1"]) == 0
    assert main(["figure", "--csv", str(csv), "--out", str(png)]) == 0
    assert csv.exists() and png.exists()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
