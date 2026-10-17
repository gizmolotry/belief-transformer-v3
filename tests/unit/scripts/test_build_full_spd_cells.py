from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "build_full_spd_cells.py"
SPEC = importlib.util.spec_from_file_location("build_full_spd_cells", SCRIPT)
assert SPEC and SPEC.loader
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def _hash(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> Path:
    rows = [{"id": f"a{i:03d}", "publisher": f"p{i // 4:02d}"} for i in range(40)]
    banks = {}
    for number, name in enumerate(("Q0", "Q1", "Q2")):
        rng = np.random.default_rng(number)
        x, t0 = tmp_path / f"{name}x.npy", tmp_path / f"{name}t0.npy"
        np.save(x, rng.normal(size=(40, 3)).astype(np.float32))
        np.save(t0, rng.normal(size=(40, 8, 3)).astype(np.float32))
        banks[name] = {
            "prompt_hash": f"{number + 1:x}" * 64,
            "q_bank_hash": f"{number + 1:x}" * 64,
            "cohort": {"article_order": rows},
            "model": {"class_names": ["contradict", "neutral", "entail"]},
            "artifacts": {"X": {"path": str(x), "sha256": _hash(x)}, "T0": {"path": str(t0), "sha256": _hash(t0)}},
        }
    path = tmp_path / "banks.json"
    path.write_text(json.dumps({"banks": banks}), encoding="utf-8")
    return path


def test_cli_defaults_to_dry_run_and_synthetic_build_is_exclusive(tmp_path, capsys):
    manifest = _manifest(tmp_path)
    assert cli.main(["--bank-manifest", str(manifest), "--synthetic"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    output = tmp_path / "cells.json"
    assert cli.main(["--bank-manifest", str(manifest), "--synthetic", "--build", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "synthetic" and len(payload["cells"]) == 45
    assert cli.main(["--bank-manifest", str(manifest), "--synthetic", "--build", "--output", str(output)]) == 2


def test_real_marker_is_under_gitignored_output_tree():
    assert cli.REAL_MARKER.parent == cli.ROOT / "outputs" / "full_spd_tensor"
    relative = cli.REAL_MARKER.relative_to(cli.ROOT)
    checked = subprocess.run(
        ["git", "check-ignore", "--quiet", str(relative)], cwd=cli.ROOT, check=False
    )
    assert checked.returncode == 0


def test_authorization_binds_exact_manifest_and_verifies_preflight_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_git_sha", lambda: "abc")
    manifest = _manifest(tmp_path)
    manifest_record = cli._path_record(manifest)
    preflight = {
        "schema": "full_spd_preflight/v1",
        "git_commit": "abc",
        "git_dirty": False,
        "inputs": {"bank_manifest": manifest_record},
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    authorization = {
        "schema": "full_spd_real_authorization/v1",
        "git_commit": "abc",
        "preflight": cli._path_record(preflight_path),
        "preflight_hash": cli._canonical_hash(preflight),
        "bank_manifest": manifest_record,
    }
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")

    loaded, binding = cli._authorization(authorization_path, manifest)
    assert loaded == authorization
    assert binding == {
        "authorization": cli._path_record(authorization_path),
        "preflight": cli._path_record(preflight_path),
        "bank_manifest": manifest_record,
    }

    other_manifest = tmp_path / "other.json"
    other_manifest.write_bytes(manifest.read_bytes())
    with pytest.raises(cli.CellBuilderError, match="exact preflight path/hash"):
        cli._authorization(authorization_path, other_manifest)

    preflight_path.write_text(json.dumps({**preflight, "git_dirty": True}), encoding="utf-8")
    with pytest.raises(cli.CellBuilderError, match="preflight hash mismatch"):
        cli._authorization(authorization_path, manifest)
