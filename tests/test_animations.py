"""Animation regression tests: exercise the real render + write path on
single timeline frames instead of the full 312/336-frame renders.

Regression coverage: animation B used to crash with an IndexError when the
CSV had no p = 0.5 rows (``simulate --no-check``) exactly in its final
frame, and the sweep captions used to clip at the right frame edge.
"""

import numpy as np
import pytest

from bitflip.animate import render_animation_a, render_animation_b
from bitflip.cli import main


@pytest.fixture(scope="module")
def no_check_csv(tmp_path_factory):
    """Sweep CSV without the p = 0.5 boundary rows, as --no-check produces."""
    csv = tmp_path_factory.mktemp("sweep") / "sweep_no_check.csv"
    assert main(["simulate", "--out", str(csv), "--shots", "20_000",
                 "--no-check", "--seed", "1"]) == 0
    return csv


def test_sweep_animation_final_frame_without_p05_rows(tmp_path, no_check_csv):
    # Regression for F5: the final frame must render (with a graceful note)
    # when the CSV has no p = 0.5 rows.
    paths = render_animation_b(tmp_path, fps=12, csv_path=no_check_csv,
                               frame_indices=[0, 110, 335])
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_sweep_animation_final_frame_with_p05_rows(tmp_path):
    # The default CSV keeps the boundary check; the final frame shows it.
    paths = render_animation_b(tmp_path, fps=12,
                               csv_path=None,  # computes and saves default sweep
                               frame_indices=[335])
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_round_animation_final_frame(tmp_path):
    paths = render_animation_a(tmp_path, fps=12, frame_indices=[0, 311])
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_captions_stay_inside_the_frame(tmp_path, no_check_csv):
    # Regression for F6: no caption ink may touch the right frame edge.
    paths = render_animation_b(tmp_path, fps=12, csv_path=no_check_csv,
                               frame_indices=[60, 110, 335])
    import imageio.v2 as iio

    reader = iio.get_reader(paths[0])
    for frame in reader:
        # Column 959 is the last pixel column; allow only pure background.
        assert (frame[:, -1, :] == 255).all(), "non-white pixels at the right edge"
    reader.close()
