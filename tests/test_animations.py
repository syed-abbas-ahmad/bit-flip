"""Animation regression tests: exercise the real render + write path on
single timeline frames instead of the full 312/336-frame renders.

Regression coverage: animation B used to crash with an IndexError when the
CSV had no p = 0.5 rows (``simulate --no-check``) exactly in its final
frame, and the sweep captions used to clip at the right frame edge.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bitflip.animate import render_animation_a, render_animation_b, verify_manifest
from bitflip.cli import main

REPO = Path(__file__).resolve().parents[1]
MANIFEST = "animation_verification.json"


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
    assert paths == [tmp_path / "anim_sweep_partial.mp4"]
    assert {path.name for path in tmp_path.iterdir()} == {
        "anim_sweep_partial.mp4", MANIFEST, "preview_sweep_partial_opening.png",
        "preview_sweep_partial_pause_p01.png", "preview_sweep_partial_final.png",
    }
    assert verify_manifest(tmp_path)[0]


def test_sweep_animation_final_frame_with_p05_rows(tmp_path):
    # The default CSV keeps the boundary check; the final frame shows it.
    paths = render_animation_b(tmp_path, fps=12,
                               csv_path=None,  # computes and saves default sweep
                               frame_indices=[335])
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_round_animation_final_frame(tmp_path):
    paths = render_animation_a(tmp_path, fps=12, frame_indices=iter([0, 311]))
    assert paths == [tmp_path / "anim_round_partial.mp4"]
    entry, = json.loads((tmp_path / MANIFEST).read_text())["animations"]
    assert not any(key.startswith("gif") for key in entry)
    assert entry["mp4"] == paths[0].name
    assert entry["previews"] == ["preview_round_partial_opening.png", "preview_round_partial_final.png"]
    assert {path.name for path in tmp_path.iterdir()} == {MANIFEST, entry["mp4"], *entry["previews"]}
    assert verify_manifest(tmp_path)[0]


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
    ok, report = verify_manifest(REPO / "results")
    assert ok, report
    assert len(report["entries"]) == 2
    for name, entry in report["entries"].items():
        assert entry["ok"]
        assert not any(key.startswith("gif") for key in entry)
        # Decoded metadata: 960x540 at 12 fps.
        assert entry["mp4_size_px"]["actual"] == [960, 540]
        assert float(entry["mp4_fps"]["actual"]) == 12.0
        assert entry["mp4_size_px"]["match"] and entry["mp4_fps"]["match"]


@pytest.fixture
def copied_results(tmp_path):
    # Give the copy a different directory name from the one recorded in the
    # rendered media paths. The originals in the checkout remain available.
    dst = tmp_path / "copied_media"
    dst.mkdir()
    for name in (MANIFEST, "anim_round.mp4", "anim_sweep.mp4"):
        shutil.copy2(REPO / "results" / name, dst / name)
    return dst


def test_manifest_verification_fails_on_single_fps_change(copied_results):
    manifest_path = copied_results / MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["animations"][0]["fps"] = 24  # only one field changed
    manifest_path.write_text(json.dumps(manifest, indent=2))

    ok, report = verify_manifest(copied_results)
    assert not ok
    round_entry = next(entry for name, entry in report["entries"].items()
                       if "walkthrough" in name)
    assert not round_entry["fps"]["match"]          # declared vs encoded
    assert round_entry["mp4_fps"]["match"] is True  # decoded container rate


def test_mp4_integrity_checks_each_declared_property(copied_results):
    manifest = copied_results / MANIFEST
    entry = json.loads(manifest.read_text())["animations"][0]
    changes = dict(mp4_frames=1, mp4_duration_s=1.0, mp4_size_px=[1, 1],
                   mp4_fps=1.0, mp4_sha256="0" * 64)
    entry.update(changes)
    manifest.write_text(json.dumps({"animations": [entry]}))
    ok, report = verify_manifest(copied_results)
    assert not ok
    checks = report["entries"][entry["name"]]
    assert all(checks[key]["match"] is False for key in changes)


@pytest.fixture
def legacy_results(copied_results):
    # Old distributions shipped both formats. A tiny GIF exercises that
    # manifest schema without shipping an extra movie in the test suite.
    manifest = copied_results / MANIFEST
    record = json.loads(manifest.read_text())
    entry = record["animations"][0]
    gif = copied_results / "anim_round.gif"
    Image.new("RGB", (8, 8), "white").save(
        gif, save_all=True, append_images=[Image.new("RGB", (8, 8), "black")],
        duration=[100, 200], loop=0,
    )
    entry.update(gif=gif.name, gif_frames=2, gif_duration_s=0.3,
                 gif_sha256=hashlib.sha256(gif.read_bytes()).hexdigest())
    record["animations"] = [entry]
    manifest.write_text(json.dumps(record))
    return copied_results


def test_relocated_legacy_manifest_uses_shared_rate_and_size(legacy_results):
    manifest = legacy_results / MANIFEST
    record = json.loads(manifest.read_text())
    entry, = record["animations"]
    for key in ("fps", "mp4_fps", "mp4_size_px"):
        del entry[key]
    entry["mp4"] = r"X:\previous-render\anim_round.mp4"
    entry["gif"] = "/previous-render/anim_round.gif"
    manifest.write_text(json.dumps({
        "animations": [entry], "fps": record["fps"], "size_px": record["size_px"],
    }))
    ok, report = verify_manifest(legacy_results)
    assert ok, report
    assert report["entries"][entry["name"]]["mp4_size_px"]["match"]
    assert report["entries"][entry["name"]]["gif_duration_s"]["actual"] == 0.3


def test_verify_command_exit_codes_for_copied_media(copied_results, monkeypatch):
    monkeypatch.chdir(REPO)
    command = ["verify", "--results-dir", str(copied_results)]
    assert main(command) == 0
    with (copied_results / "anim_round.mp4").open("ab") as movie:
        movie.write(b"changed after copying")
    assert main(command) == 1


def test_missing_copied_media_cannot_use_original(copied_results, monkeypatch):
    monkeypatch.chdir(REPO)
    walkthrough, sweep = json.loads((copied_results / MANIFEST).read_text())["animations"]
    (copied_results / "anim_round.mp4").unlink()
    ok, report = verify_manifest(copied_results)
    assert not ok and report["ok"] is False
    assert report["entries"][walkthrough["name"]]["ok"] is False
    assert report["entries"][sweep["name"]]["ok"] is True


@pytest.mark.parametrize("animations", [[], [None], [{}]], ids=[
    "empty-list", "non-object-entry", "missing-fields",
])
def test_invalid_animation_list_fails_cleanly(tmp_path, animations):
    (tmp_path / MANIFEST).write_text(json.dumps({"animations": animations}))
    ok, report = verify_manifest(tmp_path)
    assert not ok and report["ok"] is False


@pytest.mark.parametrize("field,bad_value", [
    ("mp4", None), ("fps", "unknown"), ("mp4_size_px", [None, 540]),
    ("mp4", "../anim_round.mp4"), ("mp4", "anim_round.gif"),
    ("mp4_sha256", "0"), ("mp4_duration_s", 0), ("mp4_frames", True),
])
def test_invalid_animation_field_fails_cleanly(copied_results, field, bad_value):
    manifest = copied_results / MANIFEST
    entry = json.loads(manifest.read_text())["animations"][0]
    entry[field] = bad_value
    manifest.write_text(json.dumps({"animations": [entry]}))
    ok, report = verify_manifest(copied_results)
    assert not ok and report["ok"] is False


def test_missing_mp4_key_fails_cleanly(copied_results):
    manifest = copied_results / MANIFEST
    entry = json.loads(manifest.read_text())["animations"][0]
    del entry["mp4"]
    manifest.write_text(json.dumps({"animations": [entry]}))
    ok, report = verify_manifest(copied_results)
    assert not ok and report["ok"] is False


@pytest.mark.parametrize("field", ["gif", "gif_sha256", "gif_frames", "gif_duration_s"])
def test_incomplete_gif_declaration_fails(legacy_results, field):
    manifest = legacy_results / MANIFEST
    record = json.loads(manifest.read_text())
    del record["animations"][0][field]
    manifest.write_text(json.dumps(record))
    assert not verify_manifest(legacy_results)[0]


def test_missing_declared_gif_fails(legacy_results):
    (legacy_results / "anim_round.gif").unlink()
    assert not verify_manifest(legacy_results)[0]


@pytest.mark.parametrize("field,bad_value", [
    ("gif", None), ("gif", "anim_round.mp4"), ("gif_sha256", "0" * 64),
    ("gif_frames", 3), ("gif_duration_s", 0.6),
])
def test_invalid_gif_declaration_fails(legacy_results, field, bad_value):
    manifest = legacy_results / MANIFEST
    record = json.loads(manifest.read_text())
    record["animations"][0][field] = bad_value
    manifest.write_text(json.dumps(record))
    assert not verify_manifest(legacy_results)[0]


@pytest.mark.parametrize("kind", ["mp4", "gif"])
def test_unreadable_media_with_matching_digest_fails_cleanly(legacy_results, kind):
    manifest = legacy_results / MANIFEST
    entry = json.loads(manifest.read_text())["animations"][0]
    damaged = b"unreadable media"
    (legacy_results / f"anim_round.{kind}").write_bytes(damaged)
    entry[f"{kind}_sha256"] = hashlib.sha256(damaged).hexdigest()
    manifest.write_text(json.dumps({"animations": [entry]}))
    ok, report = verify_manifest(legacy_results)
    assert not ok and report["ok"] is False
    assert report["entries"][entry["name"]]["ok"] is False


def test_absent_decoder_fps_fails_cleanly(copied_results, monkeypatch):
    manifest = copied_results / MANIFEST
    entry = json.loads(manifest.read_text())["animations"][0]
    manifest.write_text(json.dumps({"animations": [entry]}))
    monkeypatch.setattr("bitflip.animate._probe", lambda *args: dict(
        entry, mp4_fps=None, mp4_duration_s=None,
    ))
    ok, report = verify_manifest(copied_results)
    assert not ok and report["ok"] is False


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
