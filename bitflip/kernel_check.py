"""Notebook kernel-isolation check.

The notebook's first cell calls :func:`kernel_report` and asserts the
result when expectations are provided through environment variables:

- ``BITFLIP_EXPECT_PYTHON``: absolute path of the interpreter that must be
  running (the rebuild venv's ``python.exe``);
- ``BITFLIP_EXPECT_SOURCE``: absolute path of the source tree whose
  ``bitflip`` package must be imported (the rebuild checkout);
- ``BITFLIP_KERNEL_EVIDENCE``: optional file path for the absolute
  diagnostic paths (kept out of the public notebook output; the rebuild
  writes it next to its report for the local audit record).

Without expectations (a reader running the notebook themselves) the cell
prints portable confirmations only - whether the kernel is a virtual
environment and whether ``bitflip`` was imported from a source checkout -
never absolute paths. With expectations set, a mismatch raises AssertionError
and therefore fails the nbconvert run and the whole rebuild.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ENV_EXPECT_PYTHON = "BITFLIP_EXPECT_PYTHON"
ENV_EXPECT_SOURCE = "BITFLIP_EXPECT_SOURCE"
ENV_EVIDENCE = "BITFLIP_KERNEL_EVIDENCE"


def kernel_report(expected_python: str | None = None,
                  expected_source: str | None = None) -> dict:
    """Compare the running kernel against the expected environment.

    Returns a report with portable boolean fields (safe to print anywhere)
    plus the absolute diagnostic paths (for the local audit evidence only).
    """
    interpreter = Path(sys.executable).resolve()
    package = Path(bitflip_module_file()).resolve()
    in_venv = Path(sys.prefix).resolve() != Path(sys.base_prefix).resolve()
    source_checkout = (package.parent.parent / "pyproject.toml").exists()

    report: dict = {
        "in_virtualenv": in_venv,
        "from_source_checkout": source_checkout,
        "interpreter": str(interpreter),
        "package": str(package),
    }

    if expected_python:
        expected = Path(expected_python).resolve()
        report["expected_interpreter"] = str(expected)
        report["interpreter_matches"] = interpreter == expected
    if expected_source:
        expected_root = Path(expected_source).resolve()
        report["expected_source"] = str(expected_root)
        report["package_matches"] = (
            package == expected_root / "bitflip" / "__init__.py"
            or expected_root in package.parents
        )
    return report


def assert_kernel_isolation(expected_python: str | None = None,
                            expected_source: str | None = None) -> dict:
    """Assert the kernel matches the expected environment and return the
    portable confirmation lines. Expectations default to the environment
    variables set by the rebuild helper. Raises AssertionError on mismatch,
    which fails the nbconvert run and the rebuild."""
    expected_python = expected_python or os.environ.get(ENV_EXPECT_PYTHON)
    expected_source = expected_source or os.environ.get(ENV_EXPECT_SOURCE)
    report = kernel_report(expected_python, expected_source)

    portable = {
        "kernel is a virtual environment": report["in_virtualenv"],
        "bitflip imported from a source checkout": report["from_source_checkout"],
    }
    if expected_python:
        portable["interpreter is the expected venv Python"] = report["interpreter_matches"]
    if expected_source:
        portable["bitflip comes from the expected source tree"] = report["package_matches"]

    # Absolute diagnostics go to the local evidence file, never to the
    # public notebook output.
    evidence_path = os.environ.get(ENV_EVIDENCE)
    if evidence_path:
        Path(evidence_path).parent.mkdir(parents=True, exist_ok=True)
        Path(evidence_path).write_text(json.dumps(report, indent=2))

    failed = [name for name, ok in portable.items()
              if "expected" in name and not ok]
    if failed:
        raise AssertionError(
            "kernel isolation check failed: " + "; ".join(failed)
        )
    return portable


def bitflip_module_file() -> str:
    """Absolute path of the imported ``bitflip`` package ``__init__``."""
    import bitflip

    return bitflip.__file__
