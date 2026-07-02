"""Clean rebuild: copy the candidate sources to an isolated directory,
install into a fresh virtual environment, and run the documented commands.
Compares regenerated artifacts against the delivered ones.

Usage:  py scripts\\clean_rebuild.py <candidate_root> <rebuild_root>
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path

CANDIDATE = Path(sys.argv[1]).resolve()
REBUILD = Path(sys.argv[2]).resolve()
EXCLUDE_DIRS = {".git", "bitflip.egg-info", "__pycache__", ".pytest_cache",
                "results", ".venv"}


def copy_sources():
    if REBUILD.exists():
        shutil.rmtree(REBUILD)
    REBUILD.mkdir(parents=True)
    for item in CANDIDATE.iterdir():
        if item.name in EXCLUDE_DIRS or item.name.endswith(".egg-info"):
            continue
        dest = REBUILD / item.name
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


def main():
    print(f"candidate: {CANDIDATE}")
    print(f"rebuild:   {REBUILD}")
    copy_sources()

    print("1. fresh venv + install (pinned delivered environment)")
    venv.EnvBuilder(with_pip=True, clear=True).create(REBUILD / ".venv")
    py = REBUILD / ".venv" / "Scripts" / "python.exe"
    run([str(py), "-m", "pip", "install", "--quiet", "-r", "requirements.txt"], REBUILD)
    run([str(py), "-m", "pip", "install", "--quiet", "-e", ".[dev]"], REBUILD)

    print("2. tests")
    out = run([str(py), "-m", "pytest", "-q"], REBUILD)
    print(out.strip().splitlines()[-1])

    print("3. simulate + figure")
    run([str(py), "-m", "bitflip", "simulate"], REBUILD)
    run([str(py), "-m", "bitflip", "figure"], REBUILD)

    print("4. animations")
    run([str(py), "-m", "bitflip", "animate"], REBUILD)

    print("5. notebook execution")
    run([str(py), "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--inplace", "notebooks/bitflip_one_round.ipynb"], REBUILD)

    print("6. artifact comparison")
    report = {}
    comparisons = [
        ("results/sweep.csv", True),
        ("results/fig_sweep.png", True),
        ("results/anim_round.mp4", False),
        ("results/anim_round.gif", True),
        ("results/anim_sweep.mp4", False),
        ("results/anim_sweep.gif", True),
    ]
    for rel, expect_equal in comparisons:
        a, b = CANDIDATE / rel, REBUILD / rel
        entry = {"exists": b.exists()}
        if entry["exists"]:
            entry["candidate_sha256"] = sha256(a)
            entry["rebuild_sha256"] = sha256(b)
            entry["identical"] = entry["candidate_sha256"] == entry["rebuild_sha256"]
            if expect_equal and not entry["identical"]:
                print(f"  DIFFER (expected identical): {rel}")
            elif not expect_equal and entry["identical"]:
                print(f"  identical (nice, encoder deterministic): {rel}")
        report[rel] = entry

    print("7. notebook CSV vs CLI CSV in rebuild")
    out = run([str(py), "-m", "bitflip", "simulate", "--out", "results/sweep_cli.csv"], REBUILD)
    same = sha256(REBUILD / "results/sweep.csv") == sha256(REBUILD / "results/sweep_cli.csv")
    report["notebook_vs_cli_csv_identical"] = same

    (REBUILD / "rebuild_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    ok = all(
        v.get("identical", False) for k, v in report.items()
        if isinstance(v, dict)
    ) and same
    print("CLEAN REBUILD:", "PASS" if ok else "CHECK DIFFERENCES")


if __name__ == "__main__":
    main()
