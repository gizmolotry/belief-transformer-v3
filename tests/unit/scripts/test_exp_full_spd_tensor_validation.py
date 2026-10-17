"""Tests for the thin full-SPD validation CLI; no model inference is exercised."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import pytest

from core import full_spd_geometry as geometry


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "exp_full_spd_tensor_validation.py"
SPEC = importlib.util.spec_from_file_location("exp_full_spd_tensor_validation", SCRIPT)
assert SPEC and SPEC.loader
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _potential_provenance(*, seed: int, q_hash: str) -> dict:
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    payload = {
        "schema": "full_spd_geometry.v1",
        "tensor_family": "full_spd",
        "formula": "g_i = I + v_i v_i^T; edge full trapezoid",
        "target_source_kind": "observer_derived",
        "ordered_ids_hash": digest("ordered"),
        "node_gradient_hash": digest("gradient"),
        "support_hash": digest("support"),
        "support_weight_hash": digest("weights"),
        "model_state_hash": digest(f"model:{seed}"),
        "loss_trace_hash": digest(f"loss:{seed}"),
        "train_id_hash": digest("train"),
        "heldout_id_hash": digest("heldout"),
        "target_hash": digest("target"),
        "q_bank_hash": q_hash,
        "seed": seed,
        "git_commit": "abc",
        "training": {"prediction_hashes": {
            name: digest(f"{name}:{seed}") for name in ("train", "train_z", "heldout", "heldout_z")
        }},
    }
    payload["provenance_hash"] = geometry.canonical_json_sha256(payload)
    return payload


def _real_validation_fixture(tmp_path: Path):
    manifest = _write(tmp_path / "banks.json", {"banks": {}})
    manifest_record = cli._path_record(manifest)
    preflight = {
        "schema": "full_spd_preflight/v1",
        "git_commit": "abc",
        "git_dirty": False,
        "banks": {bank: {"prompt_hash": bank.lower() * 32} for bank in ("Q0", "Q1", "Q2")},
        "inputs": {"bank_manifest": manifest_record},
    }
    preflight_path = _write(tmp_path / "preflight.json", preflight)
    authorization = {
        "schema": "full_spd_real_authorization/v1",
        "git_commit": "abc",
        "preflight": cli._path_record(preflight_path),
        "preflight_hash": hashlib.sha256(cli._canonical_json(preflight)).hexdigest(),
        "bank_manifest": manifest_record,
    }
    authorization_path = _write(tmp_path / "authorization.json", authorization)
    binding = {
        "authorization": cli._path_record(authorization_path),
        "preflight": cli._path_record(preflight_path),
        "bank_manifest": manifest_record,
    }
    cells = []
    for bank in ("Q0", "Q1", "Q2"):
        q_hash = bank.lower() * 32
        for fold in range(5):
            for seed in cli.SEEDS:
                shuffled_seed = seed + 100_000 + fold
                eccentricities = {arm: [1.0, 1.0] for arm in cli.BUILDER_ARMS}
                tolerances = {
                    arm: geometry.near_tie_tolerance(eccentricities[arm])
                    for arm in cli.BUILDER_GATE_ARMS
                }
                digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
                provenance = {
                    "schema": "full_spd_cached_cell/v1",
                    "target_source_kind": "observer_derived",
                    "q_bank_hash": q_hash,
                    "git_commit": "abc",
                    "seed": seed,
                    "observer_shuffled_seed": shuffled_seed,
                    "ordered_ids_hash": digest("ordered"),
                    "train_id_hash": digest("train"),
                    "heldout_id_hash": digest("heldout"),
                    "target_hash": digest("target"),
                    "x_hash": digest(f"x:{bank}"),
                    "t0_hash": digest(f"t0:{bank}"),
                    "support_hash": digest(f"support:{fold}"),
                    "weight_hashes": {arm: digest(f"weight:{arm}") for arm in cli.BUILDER_ARMS},
                    "distance_hashes": {arm: digest(f"distance:{arm}") for arm in cli.BUILDER_ARMS},
                    "bank_manifest": manifest_record,
                    "near_tie_tolerance_rule": geometry.NEAR_TIE_TOLERANCE_RULE,
                    "near_tie_tolerances": tolerances,
                    "potential_training": {
                        "base": _potential_provenance(seed=seed, q_hash=q_hash),
                        "observer_shuffled": _potential_provenance(seed=shuffled_seed, q_hash=q_hash),
                    },
                    "authorization_binding": binding,
                }
                provenance["provenance_hash"] = hashlib.sha256(
                    json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                ).hexdigest()
                cells.append({
                    "mode": "real", "bank": bank, "fold": fold, "seed": seed,
                    "eccentricities": eccentricities,
                    "gate_metrics": {arm: {} for arm in cli.BUILDER_GATE_ARMS},
                    "near_tie_tolerances": tolerances,
                    "connected": {arm: True for arm in cli.BUILDER_GATE_ARMS},
                    "construction_valid": True, "provenance_valid": True,
                    "corpus_control_passed": False, "provenance": provenance,
                })
    return cells, preflight, authorization, authorization_path


def _contract(*, k: int = 10) -> dict:
    return {
        "seeds": [42, 420, 4200],
        "k": k,
        "permutations": 10_000,
        "bootstraps": 10_000,
        "arms": sorted(cli.PRIMARY_ARMS | cli.NEGATIVE_CONTROLS),
        "thresholds": {
            "delta_min": 0.01,
            "tau_disp": 0.1,
            "tau_tie": 0.5,
            "tau_gen": 0.1,
            "tau_gap": 0.5,
            "tau_amp": 3.0,
        },
    }


def _calibration() -> dict:
    return {"calibration": _contract()["thresholds"]}


def _banks(*, shared_x: bool = False, independent: bool = True) -> dict:
    def digest(char: str) -> str:
        return char * 64

    framing = {
        "Q0": {
            "relation_to_Q0": "anchor",
            "independently_authored_without_q0_exposure": False,
            "author": "anchor author",
            "source_process": "anchor process",
            "declaration_basis": cli.FRAMING_DECLARATION_BASIS,
        },
        "Q1": {
            "relation_to_Q0": "independent_operationalization" if independent else "paraphrase_of_Q0",
            "independently_authored_without_q0_exposure": independent,
            "author": "author one" if independent else "unspecified",
            "source_process": "process one" if independent else "direct paraphrase of Q0",
            "declaration_basis": cli.FRAMING_DECLARATION_BASIS,
        },
        "Q2": {
            "relation_to_Q0": "independent_operationalization" if independent else "paraphrase_of_Q0",
            "independently_authored_without_q0_exposure": independent,
            "author": "author two" if independent else "unspecified",
            "source_process": "process two" if independent else "direct paraphrase of Q0",
            "declaration_basis": cli.FRAMING_DECLARATION_BASIS,
        },
    }
    return {
        "banks": {
            bank: {
                "prompt_hash": digest(str(number + 1)),
                "framing_provenance": framing[bank],
                "artifacts": {
                    "X": {"sha256": digest("f" if shared_x else str(number + 4))},
                    "T0": {"sha256": digest(("7", "8", "9")[number])},
                    "T1": {"sha256": digest(("a", "b", "c")[number])},
                },
            }
            for number, bank in enumerate(("Q0", "Q1", "Q2"))
        }
    }


class _FakeEngine:
    def calibrate(self, *, simulation_summaries):
        assert simulation_summaries["kind"] == "synthetic"
        return {"calibration": _contract()["thresholds"], "decision": "calibrated"}

    def evaluate_cached(self, *, cells):
        return {"decision": "pass", "cell_count": len(cells), "synthetic": True}

    def run_real(self, **_kwargs):
        return {"decision": "fail"}


def test_engine_resolution_accepts_fake_pure_engine_module(monkeypatch):
    fake = SimpleNamespace(__file__=str(SCRIPT), calibrate=lambda **_kwargs: {"calibration": _calibration()["calibration"]})
    monkeypatch.setitem(sys.modules, cli.ENGINE_MODULE, fake)
    assert cli._engine() is fake


def test_preflight_rejects_shared_bank_embedding_hashes(tmp_path, capsys):
    banks = _write(tmp_path / "banks.json", _banks(shared_x=True))
    contract = _write(tmp_path / "contract.json", _contract())
    calibration = _write(tmp_path / "calibration.json", _calibration())
    assert cli.main(["preflight", "--bank-manifest", str(banks), "--contract", str(contract), "--calibration", str(calibration)]) == 2
    assert "shared X hashes" in capsys.readouterr().err


def test_preflight_rejects_nonfrozen_runtime_contract(tmp_path, capsys):
    banks = _write(tmp_path / "banks.json", _banks())
    contract = _write(tmp_path / "contract.json", _contract(k=5))
    calibration = _write(tmp_path / "calibration.json", _calibration())
    original = cli._assert_prompt_sources_tracked
    cli._assert_prompt_sources_tracked = lambda _manifest: None
    try:
        assert cli.main(["preflight", "--bank-manifest", str(banks), "--contract", str(contract), "--calibration", str(calibration)]) == 2
    finally:
        cli._assert_prompt_sources_tracked = original
    assert "k must be exactly 10" in capsys.readouterr().err


def test_real_preflight_rejects_current_paraphrase_banks(tmp_path, capsys):
    banks = _write(tmp_path / "banks.json", _banks(independent=False))
    contract = _write(tmp_path / "contract.json", _contract())
    calibration = _write(tmp_path / "calibration.json", _calibration())
    assert cli.main([
        "preflight", "--bank-manifest", str(banks), "--contract", str(contract),
        "--calibration", str(calibration),
    ]) == 2
    assert "secondary paraphrase analysis" in capsys.readouterr().err


def test_independent_framing_gate_accepts_distinct_human_declarations():
    manifest = _banks(independent=True)
    cli._validate_bank_manifest(manifest)
    cli._assert_independent_framing_for_real(manifest)


def test_independent_framing_gate_rejects_shared_author_or_process():
    shared_author = _banks(independent=True)
    shared_author["banks"]["Q2"]["framing_provenance"]["author"] = "author one"
    with pytest.raises(cli.ValidationGateError, match="distinct authors"):
        cli._assert_independent_framing_for_real(shared_author)
    shared_process = _banks(independent=True)
    shared_process["banks"]["Q2"]["framing_provenance"]["source_process"] = "process one"
    with pytest.raises(cli.ValidationGateError, match="distinct source_process"):
        cli._assert_independent_framing_for_real(shared_process)


def test_real_preflight_requires_committed_prompt_sources(tmp_path, monkeypatch):
    banks = _write(tmp_path / "banks.json", _banks(independent=True))
    contract = _write(tmp_path / "contract.json", _contract())
    calibration = _write(tmp_path / "calibration.json", _calibration())
    observed = []
    monkeypatch.setattr(cli, "_assert_prompt_sources_tracked", lambda manifest: observed.append(manifest))
    monkeypatch.setattr(cli, "_git_commit", lambda: "abc")
    monkeypatch.setattr(cli, "_git", lambda *_args, **_kwargs: "")
    preflight = cli._preflight(banks, contract, calibration)
    assert observed == [_banks(independent=True)]
    assert preflight["banks"]["Q1"]["framing_provenance"]["author"] == "author one"


def test_authorization_rechecks_and_rejects_paraphrase_manifest(tmp_path, monkeypatch):
    manifest_path = _write(tmp_path / "banks.json", _banks(independent=False))
    contract_path = _write(tmp_path / "contract.json", _contract())
    calibration_path = _write(tmp_path / "calibration.json", _calibration())
    preflight = {
        "schema": "full_spd_preflight/v1",
        "git_commit": "abc",
        "git_dirty": False,
        "inputs": {
            "bank_manifest": cli._path_record(manifest_path),
            "contract": cli._path_record(contract_path),
            "calibration": cli._path_record(calibration_path),
        },
    }
    preflight_path = _write(tmp_path / "preflight.json", preflight)
    monkeypatch.setattr(cli, "_assert_clean_git", lambda: None)
    monkeypatch.setattr(cli, "_git_commit", lambda: "abc")
    monkeypatch.setattr(cli, "_assert_tracked_and_committed", lambda _path: None)
    with pytest.raises(cli.ValidationGateError, match="secondary paraphrase analysis"):
        cli._authorize_real(preflight_path, tmp_path / "authorization.json")
    assert not (tmp_path / "authorization.json").exists()


def test_calibration_is_delegated_and_output_is_exclusive(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_engine", lambda: _FakeEngine())
    simulation = _write(tmp_path / "simulation.json", {"kind": "synthetic"})
    output = tmp_path / "calibration.json"
    assert cli.main(["calibrate", "--simulation-summaries", str(simulation), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["engine_result"]["decision"] == "calibrated"
    assert cli.main(["calibrate", "--simulation-summaries", str(simulation), "--output", str(output)]) == 2
    assert "refusing to overwrite" in capsys.readouterr().err


def test_cached_synthetic_end_to_end_is_thin_and_records_engine_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_engine", lambda: _FakeEngine())
    cells = _write(tmp_path / "cells.json", {"cells": [{"bank": "Q0", "fold": 0, "seed": 42, "synthetic": True}]})
    output = tmp_path / "cached.json"
    assert cli.main(["evaluate-cached", "--cell-artifacts", str(cells), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["engine_result"] == {"decision": "pass", "cell_count": 1, "synthetic": True}


def test_cached_evaluation_rejects_real_cells(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_engine", lambda: _FakeEngine())
    cells = _write(tmp_path / "cells.json", {"cells": [{"bank": "Q0", "fold": 0, "seed": 42}]})
    assert cli.main(["evaluate-cached", "--cell-artifacts", str(cells), "--output", str(tmp_path / "out.json")]) == 2
    assert "synthetic/development" in capsys.readouterr().err


def test_run_real_requires_explicit_one_shot_flag(tmp_path, capsys):
    authorization = _write(tmp_path / "authorization.json", {"schema": "full_spd_real_authorization/v1"})
    cells = _write(tmp_path / "cells.json", {"cells": []})
    assert cli.main(["run-real", "--authorization", str(authorization), "--cell-artifacts", str(cells), "--output", str(tmp_path / "out.json")]) == 2
    assert "literal --authorize-real-run flag" in capsys.readouterr().err


def test_real_cells_require_builder_grid_arms_booleans_and_hashed_provenance(tmp_path):
    cells, preflight, authorization, authorization_path = _real_validation_fixture(tmp_path)
    cli._validate_real_cells(cells, preflight, authorization, authorization_path)
    with pytest.raises(cli.ValidationGateError, match="exactly 45"):
        cli._validate_real_cells(cells[:-1], preflight, authorization, authorization_path)
    bad_arms = json.loads(json.dumps(cells))
    bad_arms[0]["eccentricities"].pop("full")
    with pytest.raises(cli.ValidationGateError, match="eight frozen"):
        cli._validate_real_cells(bad_arms, preflight, authorization, authorization_path)
    bad_hash = json.loads(json.dumps(cells))
    bad_hash[0]["provenance"]["target_source_kind"] = "external"
    with pytest.raises(cli.ValidationGateError, match="hashed provenance"):
        cli._validate_real_cells(bad_hash, preflight, authorization, authorization_path)
    bad_flag = json.loads(json.dumps(cells))
    bad_flag[0]["provenance_valid"] = 1
    with pytest.raises(cli.ValidationGateError, match="must be boolean"):
        cli._validate_real_cells(bad_flag, preflight, authorization, authorization_path)


def test_endpoint_marker_is_under_gitignored_output_tree():
    assert cli.REAL_ENDPOINT_MARKER.parent == cli.ROOT / "outputs" / "full_spd_tensor"
    relative = cli.REAL_ENDPOINT_MARKER.relative_to(cli.ROOT)
    checked = subprocess.run(
        ["git", "check-ignore", "--quiet", str(relative)], cwd=cli.ROOT, check=False
    )
    assert checked.returncode == 0


def test_authorization_refuses_dirty_tree_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_assert_clean_git", lambda: (_ for _ in ()).throw(cli.ValidationGateError("dirty")))
    with pytest.raises(cli.ValidationGateError, match="dirty"):
        cli._authorize_real(tmp_path / "missing.json", tmp_path / "authorization.json")
    assert not (tmp_path / "authorization.json").exists()


def test_fmm_import_is_rejected(tmp_path):
    engine = tmp_path / "engine.py"
    engine.write_text("from core import fmm_geodesic\n", encoding="utf-8")
    with pytest.raises(cli.ValidationGateError, match="FMM/Track4"):
        cli._assert_no_fmm_import(SimpleNamespace(__file__=str(engine)))
