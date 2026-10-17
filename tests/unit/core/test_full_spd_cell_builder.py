from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core import full_spd_experiment as experiment
from core.full_spd_cell_builder import CellBuilderError, build_cached_cells, load_bank_manifest, validate_cell_provenance
from core.full_spd_geometry import NEAR_TIE_TOLERANCE_RULE, near_tie_tolerance


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_manifest(tmp_path: Path, *, n_publishers: int = 10, per_publisher: int = 4, dim: int = 3) -> Path:
    rows = [{"id": f"a{index:03d}", "publisher": f"p{index // per_publisher:02d}"} for index in range(n_publishers * per_publisher)]
    banks = {}
    for number, bank in enumerate(("Q0", "Q1", "Q2")):
        rng = np.random.default_rng(100 + number)
        x = rng.normal(size=(len(rows), dim)).astype(np.float32)
        logits = rng.normal(size=(len(rows), 8, 3)).astype(np.float32)
        x_path, t0_path = tmp_path / f"{bank}_X.npy", tmp_path / f"{bank}_T0.npy"
        np.save(x_path, x)
        np.save(t0_path, logits)
        banks[bank] = {
            "prompt_hash": f"{number + 1:x}" * 64,
            "q_bank_hash": f"{number + 1:x}" * 64,
            "cohort": {"article_order": rows},
            "model": {"class_names": ["contradict", "neutral", "entail"]},
            "artifacts": {
                "X": {"path": str(x_path), "sha256": _file_hash(x_path)},
                "T0": {"path": str(t0_path), "sha256": _file_hash(t0_path)},
            },
        }
    manifest = tmp_path / "banks.json"
    manifest.write_text(json.dumps({"banks": banks}), encoding="utf-8")
    return manifest


def test_synthetic_builder_produces_complete_engine_compatible_grid(tmp_path):
    banks = load_bank_manifest(_synthetic_manifest(tmp_path), synthetic=True)
    calls = []

    def trainer(train_x, train_y, held_x, **kwargs):
        calls.append((tuple(kwargs["train_ids"]), tuple(kwargs["heldout_ids"]), kwargs["seed"]))
        direction = np.arange(1, held_x.shape[1] + 1, dtype=float)
        direction /= np.linalg.norm(direction)
        return type("Fit", (), {
            "predictions": held_x @ direction,
            "train_predictions": train_x @ direction,
            "node_gradients": np.tile(direction, (len(held_x), 1)),
            "provenance": {"seed": kwargs["seed"]},
        })()

    cells = build_cached_cells(banks, synthetic=True, trainer=trainer, k=7)
    assert len(cells) == 45
    assert {(cell["bank"], cell["fold"], cell["seed"]) for cell in cells} == {
        (bank, fold, seed) for bank in ("Q0", "Q1", "Q2") for fold in range(5) for seed in (42, 420, 4200)
    }
    assert all(set(cell["eccentricities"]) == {"identity", "oof_conformal", "scalar_trace", "diagonal", "full", "article_shuffled", "observer_shuffled", "spectrum_matched_random"} for cell in cells)
    assert all(set(cell["gate_metrics"]) == {"identity", "oof_conformal", "scalar_trace", "diagonal", "full"} for cell in cells)
    assert all(set(train).isdisjoint(held) for train, held, _ in calls)
    first = cells[0]
    strict_thresholds = {"delta_min": 0.0, "tau_disp": 0.0, "tau_tie": 1.0, "tau_gen": -100.0, "tau_gap": 100.0, "tau_amp": 1e9}
    assert set(strict_thresholds) == {"delta_min", "tau_disp", "tau_tie", "tau_gen", "tau_gap", "tau_amp"}
    evaluated = experiment.evaluate_cached(cells=cells, thresholds=strict_thresholds, n_replicates=2)
    assert evaluated["replicates"]["permutations"] == 2
    assert len(first["provenance"]["weight_hashes"]) == 8
    assert len(first["provenance"]["distance_hashes"]) == 8
    assert first["provenance"]["target_source_kind"] == "synthetic"
    assert first["provenance"]["near_tie_tolerance_rule"] == NEAR_TIE_TOLERANCE_RULE
    assert first["provenance_valid"] is True
    assert validate_cell_provenance(first) is True
    for arm, eccentricities in first["eccentricities"].items():
        if arm in first["near_tie_tolerances"]:
            expected = near_tie_tolerance(eccentricities)
            assert first["near_tie_tolerances"][arm] == expected
    tampered = json.loads(json.dumps(first))
    tampered["provenance"]["target_source_kind"] = "observer_derived"
    with pytest.raises(CellBuilderError, match="provenance hash mismatch"):
        validate_cell_provenance(tampered)

    tampered_rule = json.loads(json.dumps(first))
    tampered_rule["provenance"]["near_tie_tolerance_rule"] = "absolute"
    copied = dict(tampered_rule["provenance"])
    copied.pop("provenance_hash")
    tampered_rule["provenance"]["provenance_hash"] = hashlib.sha256(
        json.dumps(copied, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    with pytest.raises(CellBuilderError, match="rule provenance mismatch"):
        validate_cell_provenance(tampered_rule)


def test_controls_are_deterministic_and_share_one_support_per_cell(tmp_path):
    banks = load_bank_manifest(_synthetic_manifest(tmp_path), synthetic=True)
    one = build_cached_cells(banks, synthetic=True, k=7)
    two = build_cached_cells(banks, synthetic=True, k=7)
    assert one[0]["provenance"]["support_hash"] == two[0]["provenance"]["support_hash"]
    assert one[0]["eccentricities"]["article_shuffled"] == two[0]["eccentricities"]["article_shuffled"]
    assert one[0]["eccentricities"]["spectrum_matched_random"] == two[0]["eccentricities"]["spectrum_matched_random"]


def test_manifest_tampering_and_shared_bank_x_are_rejected(tmp_path):
    manifest = _synthetic_manifest(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    first_path = Path(data["banks"]["Q0"]["artifacts"]["X"]["path"])
    first_path.write_bytes(first_path.read_bytes() + b"tamper")
    with pytest.raises(CellBuilderError, match="hash mismatch"):
        load_bank_manifest(manifest, synthetic=True)

    manifest = _synthetic_manifest(tmp_path / "fresh") if False else _synthetic_manifest(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    q0 = data["banks"]["Q0"]["artifacts"]["X"]
    data["banks"]["Q1"]["artifacts"]["X"] = dict(q0)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CellBuilderError, match="share an X"):
        load_bank_manifest(manifest, synthetic=True)
