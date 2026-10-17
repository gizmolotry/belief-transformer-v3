"""Fail-closed CLI for the full-SPD tensor validation protocol.

This module deliberately owns *only* authorization, artifact provenance, and
write-once manifests.  ``core.full_spd_experiment`` owns calibration and every
statistical/geometry calculation.  It never performs model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from core.full_spd_cell_builder import (
    ARMS as BUILDER_ARMS,
    GATE_ARMS as BUILDER_GATE_ARMS,
    CellBuilderError,
    validate_cell_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
ENGINE_MODULE = "core.full_spd_experiment"
SEEDS = (42, 420, 4200)
K = 10
PERMUTATIONS = 10_000
BOOTSTRAPS = 10_000
PRIMARY_ARMS = frozenset({"identity", "oof_conformal", "scalar_trace", "diagonal", "full"})
NEGATIVE_CONTROLS = frozenset(
    {"article_shuffled", "observer_shuffled", "spectrum_matched_random"}
)
REQUIRED_CALIBRATION_KEYS = frozenset(
    {"delta_min", "tau_disp", "tau_tie", "tau_gen", "tau_gap", "tau_amp"}
)
REAL_ENDPOINT_MARKER = ROOT / "outputs" / "full_spd_tensor" / "full_spd_real_endpoint_marker.json"
FRAMING_DECLARATION_BASIS = "human_audited_declaration_not_computational_proof"


class ValidationGateError(RuntimeError):
    """Raised when a mandatory tensor-validation gate is not satisfied."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise ValidationGateError(f"required JSON artifact does not exist: {artifact}")
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationGateError(f"invalid JSON artifact {artifact}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationGateError(f"JSON artifact must be an object: {artifact}")
    return payload


def _path_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _git(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if check and completed.returncode:
        raise ValidationGateError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _git_commit() -> str:
    return _git("rev-parse", "HEAD")


def _assert_clean_git() -> None:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValidationGateError("real-run authorization requires a clean git worktree")


def _repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValidationGateError(f"artifact must be inside repository: {resolved}") from exc


def _assert_tracked_and_committed(path: str | Path) -> None:
    relative = _repo_relative(path)
    _git("ls-files", "--error-unmatch", "--", relative)
    _git("cat-file", "-e", f"HEAD:{relative}")


def _runtime_metadata() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "torch", "scipy"):
        spec = importlib.util.find_spec(name)
        if spec is not None:
            module = importlib.import_module(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def _exclusive_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise ValidationGateError(f"refusing to overwrite existing output: {output}") from exc
    return output


def _artifact_hash(value: Any, label: str) -> str:
    if isinstance(value, Mapping):
        digest = value.get("sha256")
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
            return digest
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        return value
    raise ValidationGateError(f"{label} must declare a lowercase SHA-256")


def _bank_entries(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    banks = manifest.get("banks", manifest)
    if not isinstance(banks, Mapping):
        raise ValidationGateError("bank manifest must contain an object named banks")
    expected = {"Q0", "Q1", "Q2"}
    if set(banks) != expected:
        raise ValidationGateError("bank manifest must contain exactly Q0, Q1, and Q2")
    if not all(isinstance(value, Mapping) for value in banks.values()):
        raise ValidationGateError("each bank manifest entry must be an object")
    return {str(key): value for key, value in banks.items()}


def _framing_declaration(bank: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    declaration = entry.get("framing_provenance")
    if not isinstance(declaration, Mapping):
        raise ValidationGateError(f"{bank} must declare framing_provenance")
    relation = declaration.get("relation_to_Q0")
    independent = declaration.get("independently_authored_without_q0_exposure")
    author = declaration.get("author")
    process = declaration.get("source_process")
    basis = declaration.get("declaration_basis")
    if not isinstance(relation, str) or not relation.strip():
        raise ValidationGateError(f"{bank} framing_provenance requires relation_to_Q0")
    if type(independent) is not bool:
        raise ValidationGateError(
            f"{bank} framing_provenance requires a boolean independently_authored_without_q0_exposure"
        )
    if not isinstance(author, str) or not isinstance(process, str):
        raise ValidationGateError(f"{bank} framing author/source_process must be strings")
    if basis != FRAMING_DECLARATION_BASIS:
        raise ValidationGateError(
            f"{bank} framing provenance must state that independence is a human-audited declaration, not computational proof"
        )
    return {
        "relation_to_Q0": relation.strip(),
        "independently_authored_without_q0_exposure": independent,
        "author": author.strip(),
        "source_process": process.strip(),
        "declaration_basis": basis,
    }


def _validate_bank_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    banks = _bank_entries(manifest)
    prompt_hashes: set[str] = set()
    x_hashes: set[str] = set()
    normalized: dict[str, Any] = {}
    for bank, entry in banks.items():
        prompt_hash = _artifact_hash(entry.get("prompt_hash"), f"{bank}.prompt_hash")
        artifacts = entry.get("artifacts", entry)
        if not isinstance(artifacts, Mapping):
            raise ValidationGateError(f"{bank}.artifacts must be an object")
        x_hash = _artifact_hash(artifacts.get("X", artifacts.get("x_hash")), f"{bank}.X")
        _artifact_hash(artifacts.get("T0", artifacts.get("t0_hash")), f"{bank}.T0")
        _artifact_hash(artifacts.get("T1", artifacts.get("t1_hash")), f"{bank}.T1")
        if entry.get("q_bank_hash") not in (None, prompt_hash):
            raise ValidationGateError(f"{bank}.q_bank_hash must equal prompt_hash")
        prompt_hashes.add(prompt_hash)
        x_hashes.add(x_hash)
        framing = _framing_declaration(bank, entry)
        normalized[bank] = {"prompt_hash": prompt_hash, "x_hash": x_hash, "framing_provenance": framing}
    if len(prompt_hashes) != 3:
        raise ValidationGateError("Q0/Q1/Q2 must have distinct prompt hashes")
    if len(x_hashes) != 3:
        raise ValidationGateError("refusing shared X hashes across prompt banks")
    return normalized


def _assert_independent_framing_for_real(bank_manifest: Mapping[str, Any]) -> None:
    """Reject paraphrase banks for the primary endpoint.

    These fields are human-audited declarations.  The gate makes them explicit
    and immutable; it does not claim to computationally prove independence.
    """

    banks = _bank_entries(bank_manifest)
    q0 = _framing_declaration("Q0", banks["Q0"])
    if q0["relation_to_Q0"] != "anchor":
        raise ValidationGateError("Q0 framing provenance must identify it as the anchor")
    declarations = [_framing_declaration(bank, banks[bank]) for bank in ("Q1", "Q2")]
    for bank, declaration in zip(("Q1", "Q2"), declarations):
        if declaration["relation_to_Q0"] != "independent_operationalization":
            raise ValidationGateError(
                f"{bank} is valid only for secondary paraphrase analysis; real endpoint requires relation_to_Q0=independent_operationalization"
            )
        if declaration["independently_authored_without_q0_exposure"] is not True:
            raise ValidationGateError(
                f"{bank} real endpoint requires independently_authored_without_q0_exposure=true"
            )
        if not declaration["author"] or not declaration["source_process"]:
            raise ValidationGateError(f"{bank} real endpoint requires nonempty author and source_process")
    if declarations[0]["author"] == declarations[1]["author"]:
        raise ValidationGateError("Q1 and Q2 real endpoint declarations require distinct authors")
    if declarations[0]["source_process"] == declarations[1]["source_process"]:
        raise ValidationGateError("Q1 and Q2 real endpoint declarations require distinct source_process values")


def _assert_prompt_sources_tracked(bank_manifest: Mapping[str, Any]) -> None:
    """Authorization needs the actual committed Q-bank files, not just digests."""

    for bank, entry in _bank_entries(bank_manifest).items():
        prompt_path = entry.get("prompt_path")
        if not isinstance(prompt_path, str):
            raise ValidationGateError(f"{bank} must declare prompt_path for real authorization")
        source = Path(prompt_path).resolve()
        if not source.is_file():
            raise ValidationGateError(f"{bank} prompt source is unavailable: {source}")
        if _sha256_file(source) != _artifact_hash(entry.get("prompt_hash"), f"{bank}.prompt_hash"):
            raise ValidationGateError(f"{bank} prompt source hash does not match bank manifest")
        _assert_tracked_and_committed(source)


def _validate_contract(contract: Mapping[str, Any], *, require_thresholds: bool = False) -> dict[str, Any]:
    config = contract.get("experiment", contract)
    if not isinstance(config, Mapping):
        raise ValidationGateError("contract must contain an experiment object")
    if tuple(config.get("seeds", ())) != SEEDS:
        raise ValidationGateError(f"contract seeds must be exactly {list(SEEDS)}")
    if config.get("k") != K:
        raise ValidationGateError("contract k must be exactly 10")
    if int(config.get("permutations", -1)) != PERMUTATIONS:
        raise ValidationGateError("contract must require exactly 10000 permutations")
    if int(config.get("bootstraps", -1)) != BOOTSTRAPS:
        raise ValidationGateError("contract must require exactly 10000 bootstraps")
    arms = set(config.get("arms", ()))
    if not (PRIMARY_ARMS | NEGATIVE_CONTROLS).issubset(arms):
        raise ValidationGateError("contract is missing a required metric arm or negative control")
    thresholds = config.get("thresholds", contract.get("thresholds"))
    if require_thresholds or thresholds is not None:
        if not isinstance(thresholds, Mapping) or not REQUIRED_CALIBRATION_KEYS.issubset(thresholds):
            raise ValidationGateError("contract lacks calibrated practical/collapse/inflation thresholds")
    return dict(config)


def _validate_calibration(calibration: Mapping[str, Any]) -> dict[str, Any]:
    values = calibration.get("calibration", calibration)
    if not isinstance(values, Mapping) or not REQUIRED_CALIBRATION_KEYS.issubset(values):
        raise ValidationGateError("calibration lacks required practical and validity thresholds")
    for key in REQUIRED_CALIBRATION_KEYS:
        value = values[key]
        if not isinstance(value, (int, float)) or not float(value) == float(value):
            raise ValidationGateError(f"calibration threshold {key} must be finite numeric")
    return dict(values)


def _preflight(bank_manifest: Path, contract_path: Path, calibration_path: Path) -> dict[str, Any]:
    manifest = _read_json(bank_manifest)
    banks = _validate_bank_manifest(manifest)
    _assert_independent_framing_for_real(manifest)
    _assert_prompt_sources_tracked(manifest)
    contract = _validate_contract(_read_json(contract_path))
    calibration = _validate_calibration(_read_json(calibration_path))
    return {
        "schema": "full_spd_preflight/v1",
        "created_utc": _utc_now(),
        "git_commit": _git_commit(),
        "git_dirty": bool(_git("status", "--porcelain=v1", "--untracked-files=all")),
        "runtime": _runtime_metadata(),
        "inputs": {
            "bank_manifest": _path_record(bank_manifest),
            "contract": _path_record(contract_path),
            "calibration": _path_record(calibration_path),
        },
        "banks": banks,
        "experiment": contract,
        "calibration": calibration,
    }


def _assert_no_fmm_import(module: Any) -> None:
    source_path = getattr(module, "__file__", None)
    if not source_path:
        raise ValidationGateError("pure engine has no inspectable source file")
    source = Path(source_path).read_text(encoding="utf-8")
    if re.search(r"^\s*(?:from|import)\s+.*(?:fmm_geodesic|track4)", source, re.MULTILINE):
        raise ValidationGateError("full-SPD validation refuses FMM/Track4 imports")


def _engine() -> Any:
    try:
        module = importlib.import_module(ENGINE_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == ENGINE_MODULE:
            raise ValidationGateError("pure engine core.full_spd_experiment is unavailable") from exc
        raise
    _assert_no_fmm_import(module)
    return module


def _engine_call(engine: Any, operation: str, **kwargs: Any) -> dict[str, Any]:
    func = getattr(engine, operation, None)
    if not callable(func):
        raise ValidationGateError(f"pure engine must expose callable {operation}(...)" )
    result = func(**kwargs)
    if not isinstance(result, Mapping):
        raise ValidationGateError(f"pure engine {operation}(...) must return an object manifest")
    return dict(result)


def _calibrate_engine(engine: Any, simulations: Mapping[str, Any]) -> dict[str, Any]:
    """Call the engine's public wrapper, or its current pure calibration API."""

    if callable(getattr(engine, "calibrate", None)):
        return _engine_call(engine, "calibrate", simulation_summaries=simulations)
    records = simulations.get("records")
    calibrate_thresholds = getattr(engine, "calibrate_thresholds", None)
    summary_type = getattr(engine, "SimulationSummary", None)
    if not isinstance(records, list) or not callable(calibrate_thresholds) or summary_type is None:
        raise ValidationGateError("pure engine lacks calibrate(...) and its calibration adapter inputs")
    try:
        calibrated = calibrate_thresholds([summary_type(**record) for record in records])
        thresholds = calibrated.thresholds
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValidationGateError(f"pure-engine calibration rejected simulation summaries: {exc}") from exc
    return {
        "calibration": {
            "delta_min": thresholds.delta_min,
            "tau_disp": thresholds.tau_disp,
            "tau_tie": thresholds.tau_tie,
            "tau_gen": thresholds.tau_gen,
            "tau_gap": thresholds.tau_gap,
            "tau_amp": thresholds.tau_amp,
        },
        "engine_summary": {
            "catch_rate": calibrated.catch_rate,
            "planted_retention": calibrated.planted_retention,
            "type_i_error": calibrated.type_i_error,
            "power": calibrated.power,
        },
    }


def _evaluate_engine(engine: Any, *, cells: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Use a high-level pure-engine endpoint when the concurrent engine provides it."""

    if callable(getattr(engine, "evaluate_cached", None)):
        return _engine_call(engine, "evaluate_cached", cells=list(cells), **kwargs)
    raise ValidationGateError("pure engine does not yet expose evaluate_cached(...); refusing CLI-side statistics")


def _run_real_engine(engine: Any, **kwargs: Any) -> dict[str, Any]:
    if callable(getattr(engine, "run_real", None)):
        return _engine_call(engine, "run_real", **kwargs)
    raise ValidationGateError("pure engine does not yet expose run_real(...); refusing CLI-side statistics")


def _load_cells(paths: Sequence[str], *, development_only: bool = False) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for raw_path in paths:
        payload = _read_json(raw_path)
        candidate = payload.get("cells", [payload])
        if not isinstance(candidate, list) or not all(isinstance(cell, Mapping) for cell in candidate):
            raise ValidationGateError("cell artifacts must be objects or a manifest containing cells")
        for cell in candidate:
            normalized = dict(cell)
            if development_only and not (
                normalized.get("synthetic") is True
                or normalized.get("mode") in {"synthetic", "development"}
            ):
                raise ValidationGateError("evaluate-cached accepts synthetic/development cells only")
            cells.append(normalized)
    if not cells:
        raise ValidationGateError("at least one cached cell artifact is required")
    return cells


def _validate_real_cells(
    cells: Sequence[Mapping[str, Any]], preflight: Mapping[str, Any],
    authorization: Mapping[str, Any], authorization_path: Path,
) -> None:
    expected = {(bank, fold, seed) for bank in ("Q0", "Q1", "Q2") for fold in range(5) for seed in SEEDS}
    observed: set[tuple[str, int, int]] = set()
    bank_hashes = preflight["banks"]
    inputs = preflight.get("inputs")
    manifest_record = inputs.get("bank_manifest") if isinstance(inputs, Mapping) else None
    if not isinstance(manifest_record, Mapping):
        raise ValidationGateError("preflight lacks bank-manifest input binding")
    normalized_manifest = {
        "path": str(Path(str(manifest_record.get("path"))).resolve()),
        "sha256": manifest_record.get("sha256"),
    }
    if authorization.get("bank_manifest") != normalized_manifest:
        raise ValidationGateError("authorization bank-manifest binding does not match preflight inputs")
    preflight_record = authorization.get("preflight")
    expected_binding = {
        "authorization": _path_record(authorization_path),
        "preflight": _path_record(Path(str(preflight_record.get("path")))),
        "bank_manifest": normalized_manifest,
    }
    for cell in cells:
        try:
            bank, fold, seed = str(cell["bank"]), int(cell["fold"]), int(cell["seed"])
            provenance = cell["provenance"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationGateError("every real cell needs bank, fold, seed, and provenance") from exc
        if not isinstance(provenance, Mapping):
            raise ValidationGateError("real-cell provenance must be an object")
        if cell.get("mode") != "real":
            raise ValidationGateError("real endpoint accepts only builder real-mode cells")
        eccentricities = cell.get("eccentricities")
        gate_metrics = cell.get("gate_metrics")
        connected = cell.get("connected")
        tolerances = cell.get("near_tie_tolerances")
        if not isinstance(eccentricities, Mapping) or set(eccentricities) != set(BUILDER_ARMS):
            raise ValidationGateError("real cells need exactly eight frozen eccentricity arms")
        if not isinstance(gate_metrics, Mapping) or set(gate_metrics) != set(BUILDER_GATE_ARMS):
            raise ValidationGateError("real cells need exactly five gate arms")
        if not isinstance(connected, Mapping) or set(connected) != set(BUILDER_GATE_ARMS) or not all(isinstance(value, bool) for value in connected.values()):
            raise ValidationGateError("real-cell connectivity must contain five booleans")
        if not isinstance(tolerances, Mapping) or set(tolerances) != set(BUILDER_GATE_ARMS):
            raise ValidationGateError("real cells need five frozen near-tie tolerances")
        for flag in ("construction_valid", "provenance_valid", "corpus_control_passed"):
            if not isinstance(cell.get(flag), bool):
                raise ValidationGateError(f"real-cell {flag} must be boolean")
        if cell.get("provenance_valid") is not True:
            raise ValidationGateError("real-cell provenance_valid must be derived and true")
        try:
            if not validate_cell_provenance(cell):
                raise CellBuilderError("cell provenance validation returned false")
        except CellBuilderError as exc:
            raise ValidationGateError(f"invalid real-cell hashed provenance: {exc}") from exc
        if provenance.get("authorization_binding") != expected_binding:
            raise ValidationGateError("real-cell authorization/preflight/manifest binding mismatch")
        if provenance.get("q_bank_hash") != bank_hashes.get(bank, {}).get("prompt_hash"):
            raise ValidationGateError("real-cell prompt-bank provenance hash mismatch")
        if provenance.get("git_commit") != preflight["git_commit"]:
            raise ValidationGateError("real-cell git commit does not match authorization preflight")
        if provenance.get("target_source_kind") != "observer_derived":
            raise ValidationGateError("real cells must use observer-derived targets")
        observed.add((bank, fold, seed))
    if observed != expected or len(cells) != len(expected):
        raise ValidationGateError("real run requires exactly 45 unique Q-bank/fold/seed cells")


def _authorize_real(preflight_path: Path, output: Path) -> Path:
    _assert_clean_git()
    preflight = _read_json(preflight_path)
    if preflight.get("schema") != "full_spd_preflight/v1":
        raise ValidationGateError("not a full-SPD preflight manifest")
    if preflight.get("git_dirty"):
        raise ValidationGateError("cannot authorize a preflight produced from a dirty tree")
    if preflight.get("git_commit") != _git_commit():
        raise ValidationGateError("preflight git commit does not match HEAD")
    inputs = preflight.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValidationGateError("preflight lacks input artifact records")
    for name in ("bank_manifest", "contract", "calibration"):
        record = inputs.get(name)
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ValidationGateError(f"preflight lacks {name} record")
        path = Path(record["path"])
        if not path.is_file() or _sha256_file(path) != record.get("sha256"):
            raise ValidationGateError(f"preflight {name} hash no longer matches")
        _assert_tracked_and_committed(path)
    bank_manifest = _read_json(inputs["bank_manifest"]["path"])
    _validate_bank_manifest(bank_manifest)
    _assert_independent_framing_for_real(bank_manifest)
    _assert_prompt_sources_tracked(bank_manifest)
    _validate_contract(_read_json(inputs["contract"]["path"]), require_thresholds=True)
    _validate_calibration(_read_json(inputs["calibration"]["path"]))
    authorization = {
        "schema": "full_spd_real_authorization/v1",
        "created_utc": _utc_now(),
        "git_commit": _git_commit(),
        "preflight": _path_record(preflight_path),
        "preflight_hash": _sha256_bytes(_canonical_json(preflight)),
        "bank_manifest": {
            "path": str(Path(inputs["bank_manifest"]["path"]).resolve()),
            "sha256": inputs["bank_manifest"]["sha256"],
        },
        "runtime": _runtime_metadata(),
    }
    return _exclusive_json(output, authorization)


def _run_real(authorization_path: Path, cell_paths: Sequence[str], output: Path) -> Path:
    if not REAL_ENDPOINT_MARKER.parent.exists():
        raise ValidationGateError("repository root unavailable")
    if REAL_ENDPOINT_MARKER.exists():
        raise ValidationGateError("a prior real endpoint marker exists; refusing rerun")
    _assert_clean_git()
    authorization = _read_json(authorization_path)
    if authorization.get("schema") != "full_spd_real_authorization/v1":
        raise ValidationGateError("not a full-SPD real authorization")
    if authorization.get("git_commit") != _git_commit():
        raise ValidationGateError("authorization git commit does not match HEAD")
    preflight_record = authorization.get("preflight")
    if not isinstance(preflight_record, Mapping) or not isinstance(preflight_record.get("path"), str):
        raise ValidationGateError("authorization lacks a preflight record")
    preflight_path = Path(preflight_record["path"])
    if not preflight_path.is_file() or _sha256_file(preflight_path) != preflight_record.get("sha256"):
        raise ValidationGateError("authorization preflight hash no longer matches")
    preflight = _read_json(preflight_path)
    if _sha256_bytes(_canonical_json(preflight)) != authorization.get("preflight_hash"):
        raise ValidationGateError("authorization preflight canonical hash mismatch")
    inputs = preflight["inputs"]
    normalized_manifest = {
        "path": str(Path(inputs["bank_manifest"]["path"]).resolve()),
        "sha256": inputs["bank_manifest"]["sha256"],
    }
    if authorization.get("bank_manifest") != normalized_manifest:
        raise ValidationGateError("authorization bank-manifest binding no longer matches preflight")
    current_preflight = _preflight(
        Path(inputs["bank_manifest"]["path"]), Path(inputs["contract"]["path"]), Path(inputs["calibration"]["path"])
    )
    for name in ("bank_manifest", "contract", "calibration"):
        if current_preflight["inputs"][name]["sha256"] != inputs[name]["sha256"]:
            raise ValidationGateError(f"real-run {name} hash does not match authorized preflight")
    cells = _load_cells(cell_paths)
    _validate_real_cells(cells, preflight, authorization, authorization_path.resolve())
    engine = _engine()
    marker = {
        "schema": "full_spd_real_endpoint_marker/v1",
        "created_utc": _utc_now(),
        "authorization": _path_record(authorization_path),
        "requested_output": str(Path(output).resolve()),
    }
    _exclusive_json(REAL_ENDPOINT_MARKER, marker)
    result = _run_real_engine(
        engine,
        cells=cells,
        contract=_read_json(inputs["contract"]["path"]),
        calibration=_read_json(inputs["calibration"]["path"]),
        preflight=preflight,
    )
    payload = {
        "schema": "full_spd_real_result/v1",
        "created_utc": _utc_now(),
        "git_commit": _git_commit(),
        "authorization": _path_record(authorization_path),
        "preflight": _path_record(preflight_path),
        "cell_artifacts": [_path_record(path) for path in cell_paths],
        "engine_result": result,
        "runtime": _runtime_metadata(),
    }
    return _exclusive_json(output, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--bank-manifest", required=True)
    preflight.add_argument("--contract", required=True)
    preflight.add_argument("--calibration", required=True)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--simulation-summaries", required=True)
    calibrate.add_argument("--output", required=True)
    cached = commands.add_parser("evaluate-cached")
    cached.add_argument("--cell-artifacts", required=True, nargs="+")
    cached.add_argument("--output", required=True)
    authorize = commands.add_parser("authorize-real")
    authorize.add_argument("--preflight-manifest", required=True)
    authorize.add_argument("--output", required=True)
    real = commands.add_parser("run-real")
    real.add_argument("--authorization", required=True)
    real.add_argument("--cell-artifacts", required=True, nargs="+")
    real.add_argument("--output", required=True)
    real.add_argument("--authorize-real-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            print(json.dumps(_preflight(Path(args.bank_manifest), Path(args.contract), Path(args.calibration)), indent=2, sort_keys=True))
            return 0
        if args.command == "calibrate":
            engine = _engine()
            simulations = _read_json(args.simulation_summaries)
            result = _calibrate_engine(engine, simulations)
            _validate_calibration(result)
            _exclusive_json(args.output, {"schema": "full_spd_calibration/v1", "created_utc": _utc_now(), "input": _path_record(args.simulation_summaries), "calibration": result.get("calibration", result), "engine_result": result, "git_commit": _git_commit(), "runtime": _runtime_metadata()})
            return 0
        if args.command == "evaluate-cached":
            engine = _engine()
            cells = _load_cells(args.cell_artifacts, development_only=True)
            result = _evaluate_engine(engine, cells=cells)
            _exclusive_json(args.output, {"schema": "full_spd_cached_evaluation/v1", "created_utc": _utc_now(), "cell_artifacts": [_path_record(path) for path in args.cell_artifacts], "engine_result": result, "git_commit": _git_commit(), "runtime": _runtime_metadata()})
            return 0
        if args.command == "authorize-real":
            _authorize_real(Path(args.preflight_manifest), Path(args.output))
            return 0
        if not args.authorize_real_run:
            raise ValidationGateError("run-real requires the literal --authorize-real-run flag")
        _run_real(Path(args.authorization), args.cell_artifacts, Path(args.output))
        return 0
    except ValidationGateError as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
