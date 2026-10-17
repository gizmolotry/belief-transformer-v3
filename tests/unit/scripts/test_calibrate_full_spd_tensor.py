from __future__ import annotations

from dataclasses import replace
import json

import pytest

import scripts.calibrate_full_spd_tensor as cli
from core.full_spd_experiment import (
    CalibrationError,
    CalibrationResult,
    GateThresholds,
)
from core.full_spd_simulation import LearnedPotentialRecovery


def _recovery() -> LearnedPotentialRecovery:
    return LearnedPotentialRecovery(
        g_train=0.9,
        g_hold=0.8,
        g_gap=0.1,
        gradient_cosine_mean=0.95,
        gradient_cosine_min=0.85,
        accepted=True,
        thresholds={"g_hold_min": 0.5, "gradient_cosine_mean_min": 0.8},
    )


def test_dry_run_is_default_and_creates_nothing(tmp_path):
    output = tmp_path / "calibration"
    payload = cli.run(["--output-dir", str(output)])
    assert payload["mode"] == "dry-run"
    assert payload["real_artifacts_read"] is False
    assert payload["records_per_kind"] >= 40
    assert not output.exists()


def test_execute_uses_actual_reduced_geometry_and_fails_before_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "run_learned_potential_recovery", lambda **_: _recovery())
    output = tmp_path / "failed"
    with pytest.raises(CalibrationError, match="practical margin"):
        cli.run(
            [
                "--output-dir",
                str(output),
                "--records-per-kind",
                "1",
                "--execute",
            ]
        )
    assert output.exists()
    assert list(output.iterdir()) == []


def test_successful_execute_writes_two_exclusive_bound_json_files(monkeypatch, tmp_path):
    actual_records = cli.generate_simulation_records(records_per_kind=1)
    thresholds = GateThresholds(0.01, 0.2, 0.5, 0.2, 10.0, 0.01)
    calibrated = CalibrationResult(thresholds, 1.0, 1.0, 0.0, 1.0)
    monkeypatch.setattr(cli, "generate_simulation_records", lambda **_: actual_records)
    monkeypatch.setattr(cli, "run_learned_potential_recovery", lambda **_: _recovery())
    monkeypatch.setattr(cli, "calibrate_synthetic_records", lambda records: calibrated)

    output = tmp_path / "written"
    payload = cli.run(
        ["--output-dir", str(output), "--records-per-kind", "1", "--execute"]
    )
    records_path = output / cli.RECORDS_FILENAME
    calibration_path = output / cli.CALIBRATION_FILENAME
    assert records_path.is_file() and calibration_path.is_file()
    records = json.loads(records_path.read_text(encoding="utf-8"))
    manifest = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert len(records["records"]) == 4
    assert manifest == payload
    assert manifest["real_artifacts_read"] is False
    assert manifest["records_content_hash"] == records["content_hash"]

    with pytest.raises(cli.CalibrationCLIError, match="already exist"):
        cli.run(
            ["--output-dir", str(output), "--records-per-kind", "1", "--execute"]
        )


def test_nonpositive_record_count_fails_without_creating_output(tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(cli.CalibrationCLIError, match="positive"):
        cli.run(["--output-dir", str(output), "--records-per-kind", "0", "--execute"])
    assert not output.exists()
