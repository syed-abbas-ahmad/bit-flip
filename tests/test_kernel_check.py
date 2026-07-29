"""Notebook kernel-isolation check: portable confirmations, absolute paths
only in the evidence file, and a mismatch that raises."""

import json

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


def test_matching_expectations_pass(monkeypatch, tmp_path):
    import sys

    report = kernel_report(expected_python=sys.executable,
                           expected_source=str(tmp_path))
    # Fake the package location inside tmp_path for the containment check.
    monkeypatch.setattr("bitflip.kernel_check.bitflip_module_file",
                        lambda: str(tmp_path / "bitflip" / "__init__.py"))
    portable = assert_kernel_isolation(expected_python=sys.executable,
                                       expected_source=str(tmp_path))
    assert portable["interpreter is the expected venv Python"] is True
    assert portable["bitflip comes from the expected source tree"] is True
    assert "matches" in str(report)


def test_wrong_interpreter_fails(monkeypatch, tmp_path):
    portable = None
    with pytest.raises(AssertionError, match="kernel isolation check failed"):
        portable = assert_kernel_isolation(expected_python=str(tmp_path / "venv" / "python.exe"))
    assert portable is None


def test_absolute_paths_written_only_to_evidence(monkeypatch, tmp_path):
    import sys

    evidence = tmp_path / "audit" / "kernel_evidence.json"
    monkeypatch.setenv(ENV_EVIDENCE, str(evidence))
    assert_kernel_isolation(expected_python=sys.executable)
    saved = json.loads(evidence.read_text())
    assert "interpreter" in saved and saved["interpreter"] == str(
        __import__("pathlib").Path(sys.executable).resolve())
    assert "expected_interpreter" in saved
