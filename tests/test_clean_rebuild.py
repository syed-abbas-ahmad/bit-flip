"""Clean-rebuild safety: path validation must never touch existing content,
and comparison failures must surface as a nonzero exit status."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rebuild_module():
    spec = importlib.util.spec_from_file_location(
        "clean_rebuild", REPO / "scripts" / "clean_rebuild.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refuses_destination_equal_to_source(tmp_path, rebuild_module):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("precious")
    with pytest.raises(rebuild_module.UnsafePathError):
        rebuild_module.validate_paths(tmp_path, tmp_path)
    assert sentinel.read_text() == "precious"  # untouched


def test_refuses_destination_inside_source(tmp_path, rebuild_module):
    (tmp_path / "data").mkdir()
    sentinel = tmp_path / "data" / "keep.txt"
    sentinel.write_text("precious")
    with pytest.raises(rebuild_module.UnsafePathError):
        rebuild_module.validate_paths(tmp_path, tmp_path / "data" / "sub")
    assert sentinel.read_text() == "precious"  # untouched


def test_refuses_source_inside_destination(tmp_path, rebuild_module):
    outer = tmp_path / "outer"
    inner = outer / "project"
    inner.mkdir(parents=True)
    with pytest.raises(rebuild_module.UnsafePathError):
        rebuild_module.validate_paths(inner, outer)


def test_refuses_missing_source(tmp_path, rebuild_module):
    with pytest.raises(rebuild_module.UnsafePathError):
        rebuild_module.validate_paths(tmp_path / "nope", tmp_path / "dest")


def test_refuses_nonempty_destination(tmp_path, rebuild_module):
    src = tmp_path / "src"
    src.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    sentinel = dest / "keep.txt"
    sentinel.write_text("precious")
    with pytest.raises(rebuild_module.UnsafePathError):
        rebuild_module.validate_paths(src, dest)
    assert sentinel.read_text() == "precious"  # existing work never deleted


def test_accepts_fresh_or_empty_destination(tmp_path, rebuild_module):
    src = tmp_path / "src"
    src.mkdir()
    empty_dest = tmp_path / "empty_dest"
    empty_dest.mkdir()
    got_src, got_dest = rebuild_module.validate_paths(src, empty_dest)
    assert got_dest == empty_dest.resolve()
    fresh = tmp_path / "fresh"  # does not exist yet: fine, created later
    rebuild_module.validate_paths(src, fresh)


def test_comparison_failure_is_detected(tmp_path, rebuild_module):
    candidate = tmp_path / "candidate"
    rebuild = tmp_path / "rebuild"
    for root in (candidate, rebuild):
        (root / "results").mkdir(parents=True)
    (candidate / "results" / "sweep.csv").write_text("p,q\n0.1,0.0\n")
    (rebuild / "results" / "sweep.csv").write_text("p,q\n0.1,0.02\n")  # altered!
    ok, report = rebuild_module.compare_artifacts(candidate, rebuild)
    assert not ok
    assert report["results/sweep.csv"]["status"] == "MISMATCH"


def test_lenient_media_downgrades_media_but_not_data(tmp_path, rebuild_module):
    candidate = tmp_path / "candidate"
    rebuild = tmp_path / "rebuild"
    for root in (candidate, rebuild):
        (root / "results").mkdir(parents=True)
    (candidate / "results" / "sweep.csv").write_text("data\n")
    (rebuild / "results" / "sweep.csv").write_text("data\n")  # identical data
    (candidate / "results" / "anim_sweep.mp4").write_bytes(b"old-bytes")
    (rebuild / "results" / "anim_sweep.mp4").write_bytes(b"new-bytes")  # re-encoded
    comparisons = [("results/sweep.csv", True), ("results/anim_sweep.mp4", True)]
    ok_strict, _ = rebuild_module.compare_artifacts(candidate, rebuild, comparisons)
    assert not ok_strict  # default policy: media must match too
    ok_lenient, report = rebuild_module.compare_artifacts(
        candidate, rebuild, comparisons, lenient_media=True)
    assert ok_lenient
    assert "WARNING" in report["results/anim_sweep.mp4"]["status"]


def test_comparison_passes_when_identical(tmp_path, rebuild_module):
    candidate = tmp_path / "candidate"
    rebuild = tmp_path / "rebuild"
    for root in (candidate, rebuild):
        (root / "results").mkdir(parents=True)
        (root / "results" / "sweep.csv").write_text("p,q\n0.1,0.0\n")
    ok, report = rebuild_module.compare_artifacts(
        candidate, rebuild, [("results/sweep.csv", True)])
    assert ok
    assert report["results/sweep.csv"]["identical"] is True


def test_exit_status_wiring(tmp_path, rebuild_module):
    # main() returns 2 for refused paths (usage/safety) without any build.
    rc = rebuild_module.main([str(tmp_path / "nope"), str(tmp_path / "dest")])
    assert rc == 2
