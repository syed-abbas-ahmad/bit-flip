"""Animation regression tests: exercise the real render + write path on
single timeline frames instead of the full 312/336-frame renders.

Regression coverage: animation B used to crash with an IndexError when the
CSV had no p = 0.5 rows (``simulate --no-check``) exactly in its final
frame, and the sweep captions used to clip at the right frame edge.
"""

from pathlib import Path

import numpy as np
import pytest

from bitflip.animate import render_animation_a, render_animation_b
from bitflip.cli import main

REPO = Path(__file__).resolve().parents[1]


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
        # Column 959 is the last pixel column; allow only near-white
        # background (>= 250) so antialiasing tolerance does not make the
        # test brittle while real ink still fails.
        assert (frame[:, -1, :] >= 250).all(), "ink detected at the right edge"
    reader.close()


def test_manifest_verification_passes_on_delivered_results():
    from bitflip.animate import verify_manifest

    ok, report = verify_manifest(Path("results"))
    assert ok, report
    assert len(report["entries"]) == 2
    for name, entry in report["entries"].items():
        assert entry["ok"]
        # Decoded metadata: 960x540 at 12 fps.
        assert entry["mp4_size_px"]["actual"] == [960, 540]
        assert float(entry["mp4_fps"]["actual"]) == 12.0
        assert entry["mp4_size_px"]["match"] and entry["mp4_fps"]["match"]


def test_manifest_verification_fails_on_single_fps_change(tmp_path):
    # Negative control from the release review: changing just one fps field
    # in a disposable manifest must fail verification with a nonzero exit.
    import json
    import shutil

    for name in ("anim_round.mp4", "anim_round.gif", "anim_sweep.mp4",
                 "anim_sweep.gif", "animation_verification.json"):
        shutil.copy2(Path("results") / name, tmp_path / name)
    manifest_path = tmp_path / "animation_verification.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["animations"][0]["fps"] = 24  # only one field changed
    manifest_path.write_text(json.dumps(manifest, indent=2))

    from bitflip.animate import verify_manifest

    ok, report = verify_manifest(tmp_path)
    assert not ok
    round_entry = next(entry for name, entry in report["entries"].items()
                       if "walkthrough" in name)
    assert not round_entry["fps"]["match"]          # declared vs encoded
    assert round_entry["mp4_fps"]["match"] is True  # decoded container rate


def test_verify_command_exit_codes(tmp_path):
    # The CLI verify command must exit 0 on the delivered results and
    # nonzero when one fps field is tampered with.
    import json
    import shutil

    from bitflip.cli import main

    assert main(["verify", "--results-dir", "results"]) == 0

    for name in ("anim_round.mp4", "anim_round.gif", "anim_sweep.mp4",
                 "anim_sweep.gif", "animation_verification.json"):
        shutil.copy2(Path("results") / name, tmp_path / name)
    manifest_path = tmp_path / "animation_verification.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["animations"][1]["fps"] = 6
    manifest_path.write_text(json.dumps(manifest, indent=2))
    assert main(["verify", "--results-dir", str(tmp_path)]) == 1
    # Regression for the major audit: the animation used to run to p = 0.31
    # on endpoint-clamped interpolation, drawing a horizontal tail and
    # breaking the y = x baseline. The timeline must stop at the last
    # sampled p, and every partial segment must stay inside the records.
    import pandas as pd

    from bitflip.animate import _partial_segment, _sweep_frame_state

    frame = pd.read_csv(REPO / "results" / "sweep.csv")
    grid_max = float(frame["p"].max())
    assert grid_max == pytest.approx(0.5)  # includes the boundary check rows
    sampled = frame[frame["p"] <= 0.31]
    p_final = 0.30  # last sampled sweep p used by the timeline
    for idx in (0, 24, 90, 200, 264, 335):
        state = _sweep_frame_state(idx)
        assert state["p_hi"] <= p_final + 1e-12
        for column in ("logical_failure_exact", "unencoded_failure_exact"):
            clean = frame[np.isclose(frame["q"], 0.0) & (frame["p"] <= 0.31)]
            xs, ys = _partial_segment(clean, state["p_hi"], column)
            assert xs[-1] <= p_final + 1e-12
            # Every plotted point is a record or an interpolation between
            # adjacent records - never an endpoint-clamped constant.
            if xs[-1] < state["p_hi"] - 1e-12:  # leading edge is interpolated
                lo = frame[np.isclose(frame["q"], 0.0)].sort_values("p")
                i = int(lo["p"].searchsorted(xs[-1]))
                lo_y, hi_y = lo[column].iloc[i - 1], lo[column].iloc[i]
                assert min(lo_y, hi_y) - 1e-12 <= ys[-1] <= max(lo_y, hi_y) + 1e-12
    # Baseline equality y = x on the final frame's unencoded segment.
    clean = frame[np.isclose(frame["q"], 0.0) & (frame["p"] <= 0.31)]
    xs, ys = _partial_segment(clean, p_final, "unencoded_failure_exact")
    assert np.allclose(xs, ys)

def test_sweep_never_plots_beyond_the_sampled_grid():
    # Regression for the major audit: the animation used to run to p = 0.31
    # on endpoint-clamped interpolation, drawing a horizontal tail and
    # breaking the y = x baseline. The timeline must stop at the last
    # sampled p, and every partial segment must stay inside the records.
    import pandas as pd

    from bitflip.animate import _partial_segment, _sweep_frame_state

    frame = pd.read_csv(REPO / "results" / "sweep.csv")
    grid_max = float(frame["p"].max())
    assert grid_max == pytest.approx(0.5)  # includes the boundary check rows
    p_final = 0.30  # last sampled sweep p used by the timeline
    for idx in (0, 24, 90, 200, 264, 335):
        state = _sweep_frame_state(idx)
        assert state["p_hi"] <= p_final + 1e-12
        for column in ("logical_failure_exact", "unencoded_failure_exact"):
            clean = frame[np.isclose(frame["q"], 0.0) & (frame["p"] <= 0.31)]
            xs, ys = _partial_segment(clean, state["p_hi"], column)
            assert xs[-1] <= p_final + 1e-12
            # Every plotted point is a record or an interpolation between
            # adjacent records - never an endpoint-clamped constant.
            if xs[-1] < state["p_hi"] - 1e-12:  # leading edge is interpolated
                lo = frame[np.isclose(frame["q"], 0.0)].sort_values("p")
                i = int(lo["p"].searchsorted(xs[-1]))
                lo_y, hi_y = lo[column].iloc[i - 1], lo[column].iloc[i]
                assert min(lo_y, hi_y) - 1e-12 <= ys[-1] <= max(lo_y, hi_y) + 1e-12
    # Baseline equality y = x on the final frame's unencoded segment.
    clean = frame[np.isclose(frame["q"], 0.0) & (frame["p"] <= 0.31)]
    xs, ys = _partial_segment(clean, p_final, "unencoded_failure_exact")
    assert np.allclose(xs, ys)
