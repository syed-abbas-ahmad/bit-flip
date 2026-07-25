"""Clean rebuild: copy the candidate sources to an isolated directory,
install into a fresh virtual environment, and run the documented commands.
Compares regenerated artifacts against the delivered ones.

Safety policy:
- The destination must not exist, or must be an *empty* directory. Existing
  content is never deleted.
- Source and destination are resolved and checked for overlap before any
  write; identical or nested paths are refused without touching either.
- The process exits nonzero when any expected-identical artifact differs,
  so script-driven release checks cannot accept altered artifacts.

Encoder policy: with the pinned ``requirements.txt`` the MP4/GIF encodings
are expected byte-identical. Pass ``--lenient-media`` to downgrade media
mismatches to warnings when deliberately comparing across encoder versions;
data artifacts (CSV, figure) are always strict.

Usage:
    py scripts\\clean_rebuild.py <candidate_root> <rebuild_root>
    py scripts\\clean_rebuild.py <candidate_root> <rebuild_root> --lenient-media
    py scripts\\clean_rebuild.py <candidate_root> <rebuild_root> --compare-only

``--compare-only`` skips the build and only re-compares existing
``results/`` trees (used to re-verify a completed rebuild, including a
deliberately altered artifact for the nonzero-exit check).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ARGS = sys.argv[1:]
CANDIDATE = Path(ARGS[0]).resolve() if len(ARGS) > 0 else None
REBUILD = Path(ARGS[1]).resolve() if len(ARGS) > 1 else None
LENIENT_MEDIA = "--lenient-media" in ARGS
COMPARE_ONLY = "--compare-only" in ARGS
EXCLUDE_DIRS = {".git", "bitflip.egg-info", "__pycache__", ".pytest_cache",
                "results", ".venv"}

#: Local-only workflow files that must never travel with a rebuild copy.
LOCAL_ONLY = {"PROMPT.MD", "REVIEW_PROMPT.MD", "PROJECT_STATE.md", "AGENTS.md",
              "PROJECT_PLAN.md", ".agents", "reviews"}

#: (relative path, must be byte-identical). Media entries may be downgraded
#: to warnings by --lenient-media; data artifacts are always strict.
COMPARISONS = [
    ("results/sweep.csv", True),
    ("results/fig_sweep.png", True),
    ("results/anim_round.mp4", True),
    ("results/anim_round.gif", True),
    ("results/anim_sweep.mp4", True),
    ("results/anim_sweep.gif", True),
]


class UnsafePathError(RuntimeError):
    """Raised when source/destination paths are missing, identical or nested."""


def validate_paths(candidate: Path, rebuild: Path,
                   allow_populated: bool = False) -> tuple[Path, Path]:
    """Refuse unsafe source/destination combinations before any write.

    Both must exist as directories. The destination may exist only if empty
    (or always, in ``allow_populated`` compare-only mode, which never
    deletes and only adds a report file). Paths must be distinct after
    resolution and must not contain each other.
    """
    candidate = Path(candidate).resolve()
    rebuild = Path(rebuild).resolve()
    if not candidate.is_dir():
        raise UnsafePathError(f"candidate directory does not exist: {candidate}")
    if rebuild == candidate:
        raise UnsafePathError(
            "destination equals the candidate directory; refusing to rebuild "
            "in place (choose a fresh, non-overlapping destination)"
        )
    if candidate in rebuild.parents or rebuild in candidate.parents:
        raise UnsafePathError(
            f"source and destination overlap: {candidate} and {rebuild}; "
            "refusing to write inside either directory"
        )
    if rebuild.exists():
        if not rebuild.is_dir():
            raise UnsafePathError(f"destination exists and is not a directory: {rebuild}")
        if not allow_populated and any(rebuild.iterdir()):
            raise UnsafePathError(
                f"destination exists and is not empty: {rebuild}; existing "
                "content is never deleted - choose a fresh directory"
            )
    return candidate, rebuild


def copy_sources(candidate: Path, rebuild: Path):
    """Copy the tracked project sources into the rebuild directory.

    Generated outputs (``results/``) are excluded: the pipeline regenerates
    them and the comparison then checks them against the delivered bytes.
    Local-only workflow files (prompts, project state, agent notes, review
    records) never travel with a rebuild.
    """
    rebuild.mkdir(parents=True, exist_ok=True)
    for item in candidate.iterdir():
        if item.name in EXCLUDE_DIRS or item.name in LOCAL_ONLY:
            continue
        if item.name.endswith(".egg-info"):
            continue
        dest = rebuild / item.name
        if item.is_dir():
            shutil.copytree(item, dest, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(item, dest)


def run(cmd, cwd, env=None):
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          env=env, shell=False)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return proc.stdout


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_artifacts(candidate: Path, rebuild: Path,
                      comparisons=COMPARISONS, lenient_media: bool = False):
    """Compare rebuilt artifacts against the delivered ones.

    Returns ``(ok, report)``. A missing or differing expected-identical file
    fails the comparison unless it is a media file and ``lenient_media`` is
    set, in which case the mismatch is recorded as a warning.
    """
    report = {}
    ok = True
    for rel, expect_equal in comparisons:
        a, b = candidate / rel, rebuild / rel
        entry = {"exists": b.exists()}
        if entry["exists"]:
            entry["candidate_sha256"] = sha256(a)
            entry["rebuild_sha256"] = sha256(b)
            entry["identical"] = entry["candidate_sha256"] == entry["rebuild_sha256"]
            if not entry["identical"]:
                is_media = rel.endswith((".mp4", ".gif"))
                if expect_equal and not (lenient_media and is_media):
                    ok = False
                    entry["status"] = "MISMATCH"
                elif lenient_media and is_media:
                    entry["status"] = "WARNING (encoder-dependent bytes; " \
                                      "data artifacts are compared strictly)"
                else:
                    entry["status"] = "ok (not expected identical)"
        else:
            ok = False
            entry["status"] = "MISSING"
        report[rel] = entry
    return ok, report


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    lenient_media = "--lenient-media" in argv
    compare_only = "--compare-only" in argv
    argv = [a for a in argv if a not in ("--lenient-media", "--compare-only")]
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        candidate, rebuild = validate_paths(Path(argv[0]), Path(argv[1]),
                                            allow_populated=compare_only)
    except UnsafePathError as exc:
        print(f"REFUSED: {exc}")
        return 2

    print(f"candidate: {candidate}")
    print(f"rebuild:   {rebuild}")

    if compare_only:
        print("compare-only mode: skipping the build, comparing existing results/")
        ok, report = compare_artifacts(candidate, rebuild, lenient_media=lenient_media)
        same = sha256(rebuild / "results/sweep.csv") == sha256(
            rebuild / "results/sweep_cli.csv")
        report["notebook_vs_cli_csv_identical"] = same
        ok = ok and same
        (rebuild / "rebuild_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print("CLEAN REBUILD:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    print("0. copy tracked sources")
    copy_sources(candidate, rebuild)

    print("1. fresh venv + install (pinned delivered environment)")
    venv.EnvBuilder(with_pip=True, clear=True).create(rebuild / ".venv")
    py = rebuild / ".venv" / "Scripts" / "python.exe"
    run([str(py), "-m", "pip", "install", "--quiet", "-r", "requirements.txt"], rebuild)
    run([str(py), "-m", "pip", "install", "--quiet", "-e", ".[dev]"], rebuild)

    print("2. simulate (generate the CSV the tests read)")
    run([str(py), "-m", "bitflip", "simulate"], rebuild)

    print("3. tests")
    out = run([str(py), "-m", "pytest", "-q"], rebuild)
    lines = out.strip().splitlines()
    if lines:
        print(lines[-1])

    print("4. figure + animations")
    run([str(py), "-m", "bitflip", "figure"], rebuild)
    run([str(py), "-m", "bitflip", "animate"], rebuild)

    print("5. notebook execution")
    run([str(py), "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--inplace", "notebooks/bitflip_one_round.ipynb"], rebuild)

    print("6. artifact comparison")
    ok, report = compare_artifacts(candidate, rebuild, lenient_media=lenient_media)

    print("7. notebook CSV vs CLI CSV in rebuild")
    run([str(py), "-m", "bitflip", "simulate", "--out", "results/sweep_cli.csv"], rebuild)
    same = sha256(rebuild / "results/sweep.csv") == sha256(rebuild / "results/sweep_cli.csv")
    report["notebook_vs_cli_csv_identical"] = same
    ok = ok and same

    (rebuild / "rebuild_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("CLEAN REBUILD:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
