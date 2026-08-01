"""Notebook kernel-isolation check: portable confirmations, absolute paths
only in the evidence file, and a mismatch that raises."""

import json
from pathlib import Path
import sys

import pytest

from bitflip.kernel_check import (
    assert_kernel_isolation,
    kernel_report,
    ENV_EVIDENCE,
    ENV_EXPECT_PYTHON,
    ENV_EXPECT_SOURCE,
)


def test_local_run_reports_portable_booleans(monkeypatch):
    for var in (ENV_EXPECT_PYTHON, ENV_EXPECT_SOURCE, ENV_EVIDENCE):
        monkeypatch.delenv(var, raising=False)
    portable = assert_kernel_isolation()
    # No absolute paths are returned for the public output.
    assert set(portable) == {
        "kernel is a virtual environment",
        "bitflip imported from a source checkout",
    }
    assert isinstance(portable["kernel is a virtual environment"], bool)
    # This test suite runs against the editable source checkout.
    assert portable["bitflip imported from a source checkout"] is True


def test_matching_expectations_pass():
    checkout = Path(__file__).resolve().parents[1]
    report = kernel_report(expected_python=sys.executable,
                           expected_source=str(checkout))
    assert report["interpreter_matches"] is True
    assert report["package_matches"] is True
    portable = assert_kernel_isolation(expected_python=sys.executable,
                                       expected_source=str(checkout))
    assert portable["interpreter is the expected venv Python"] is True
    assert portable["bitflip comes from the expected source tree"] is True


def test_wrong_interpreter_fails(tmp_path):
    with pytest.raises(AssertionError, match="interpreter is the expected venv Python"):
        assert_kernel_isolation(expected_python=str(tmp_path / "venv" / "python.exe"))


def test_wrong_source_tree_fails_even_with_correct_interpreter(monkeypatch, tmp_path):
    # A venv installed inside the expected checkout may contain a different
    # copy of bitflip. The package must be the checkout's own copy.
    other = tmp_path / ".venv" / "Lib" / "site-packages" / "bitflip" / "__init__.py"
    monkeypatch.setattr("bitflip.kernel_check.bitflip_module_file", lambda: str(other))
    monkeypatch.setenv(ENV_EXPECT_PYTHON, sys.executable)
    monkeypatch.setenv(ENV_EXPECT_SOURCE, str(tmp_path))
    with pytest.raises(AssertionError, match="bitflip comes from the expected source tree"):
        assert_kernel_isolation()


@pytest.mark.parametrize("variable", [ENV_EXPECT_PYTHON, ENV_EXPECT_SOURCE])
def test_blank_expectation_fails(monkeypatch, variable):
    monkeypatch.setenv(variable, "")
    with pytest.raises(ValueError, match=variable):
        assert_kernel_isolation()


def test_absolute_paths_written_only_to_evidence(monkeypatch, tmp_path):
    evidence = tmp_path / "audit" / "kernel_evidence.json"
    monkeypatch.setenv(ENV_EVIDENCE, str(evidence))
    assert_kernel_isolation(expected_python=sys.executable)
    saved = json.loads(evidence.read_text())
    assert saved["interpreter"] == saved["expected_interpreter"] == str(Path(sys.executable).resolve())
