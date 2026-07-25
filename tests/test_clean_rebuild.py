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


def test_main_pipeline_copies_sources_and_reports(tmp_path, monkeypatch, rebuild_module):
    # Regression for the MYSELF re-audit: main() once skipped copy_sources()
    # entirely, so pip failed on a missing requirements.txt. The wiring test
    # mocks the slow steps (venv, pip, pytest, notebook) and lets the fake
    # copy step populate a fresh destination with simulated pipeline outputs;
    # the comparison must then pass.
    candidate = tmp_path / "candidate"
    rebuild = tmp_path / "rebuild"
    (candidate / "results").mkdir(parents=True)
    (candidate / "requirements.txt").write_text("numpy\n")
    for rel, _ in rebuild_module.COMPARISONS:
        (candidate / rel).write_text("same-bytes\n")

    ran = []

    def fake_copy(src, dst):
        ran.append("copy")
        (dst / "results").mkdir(parents=True, exist_ok=True)
        (dst / "requirements.txt").write_text("numpy\n")
        for rel, _ in rebuild_module.COMPARISONS:
            (dst / rel).write_text("same-bytes\n")
        (dst / "results" / "sweep_cli.csv").write_text("same-bytes\n")

    class _FakeEnv:
        def create(self, path):
            ran.append("venv")

    monkeypatch.setattr(rebuild_module, "copy_sources", fake_copy)
    monkeypatch.setattr(rebuild_module.venv, "EnvBuilder", lambda **kw: _FakeEnv())
    monkeypatch.setattr(
        rebuild_module, "run",
        lambda cmd, cwd, env=None:
            ran.append(" ".join(map(str, cmd[1:4]))) or "")

    rc = rebuild_module.main([str(candidate), str(rebuild)])
    assert rc == 0
    assert ran[0] == "copy"                       # sources copied FIRST
    assert any("pytest" in step for step in ran)
    assert sum("simulate" in step for step in ran) == 2  # sweep + CLI-vs-notebook check

    report = (rebuild / "rebuild_report.json")
    assert report.exists()                        # completed report written


def test_main_pipeline_returns_nonzero_on_mismatch(tmp_path, monkeypatch, rebuild_module):
    # A deliberately altered expected-identical artifact must produce a
    # nonzero process status at the comparison step.
    candidate = tmp_path / "candidate"
    rebuild = tmp_path / "rebuild"
    (candidate / "results").mkdir(parents=True)
    (candidate / "requirements.txt").write_text("numpy\n")
    for rel, _ in rebuild_module.COMPARISONS:
        (candidate / rel).write_text("same-bytes\n")

    def fake_copy(src, dst):
        (dst / "results").mkdir(parents=True, exist_ok=True)
        (dst / "requirements.txt").write_text("numpy\n")
        for rel, _ in rebuild_module.COMPARISONS:
            # One altered data artifact in the rebuild:
            (dst / rel).write_text("same-bytes\n" if rel != "results/sweep.csv"
                                   else "altered\n")
        (dst / "results" / "sweep_cli.csv").write_text("altered\n")

    class _FakeEnv:
        def create(self, path):
            pass

    monkeypatch.setattr(rebuild_module, "copy_sources", fake_copy)
    monkeypatch.setattr(rebuild_module.venv, "EnvBuilder", lambda **kw: _FakeEnv())
    monkeypatch.setattr(rebuild_module, "run", lambda cmd, cwd, env=None: "")

    rc = rebuild_module.main([str(candidate), str(rebuild)])
    assert rc == 1
    report = (rebuild / "rebuild_report.json")
    assert report.exists()
    assert report.read_text().count("MISMATCH") == 1
