"""Fail-closed cached-cell construction CLI; no endpoint is evaluated here."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.full_spd_cell_builder import CellBuilderError, build_from_manifest, load_bank_manifest


ROOT = Path(__file__).resolve().parents[1]
REAL_MARKER = ROOT / "outputs" / "full_spd_tensor" / "full_spd_real_cell_construction.marker.json"


def _git_sha() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise CellBuilderError("unable to resolve git HEAD")
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise CellBuilderError(f"refusing to overwrite output: {path}") from exc


def _authorization(path: Path, bank_manifest: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CellBuilderError("real construction needs a valid authorization JSON") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != "full_spd_real_authorization/v1":
        raise CellBuilderError("real construction needs full_spd_real_authorization/v1")
    if payload.get("git_commit") != _git_sha():
        raise CellBuilderError("authorization commit does not bind current HEAD")
    preflight_record = payload.get("preflight")
    if not isinstance(preflight_record, Mapping) or not isinstance(preflight_record.get("path"), str):
        raise CellBuilderError("authorization lacks a preflight path/hash record")
    preflight_path = Path(preflight_record["path"]).resolve()
    if not preflight_path.is_file() or _sha256(preflight_path) != preflight_record.get("sha256"):
        raise CellBuilderError("authorization preflight hash mismatch")
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CellBuilderError("authorization preflight is invalid JSON") from exc
    if not isinstance(preflight, Mapping) or preflight.get("schema") != "full_spd_preflight/v1":
        raise CellBuilderError("authorization does not bind a full-SPD preflight")
    if _canonical_hash(preflight) != payload.get("preflight_hash"):
        raise CellBuilderError("authorization preflight canonical hash mismatch")
    if preflight.get("git_commit") != payload.get("git_commit") or preflight.get("git_dirty") is not False:
        raise CellBuilderError("authorization/preflight git binding mismatch")
    inputs = preflight.get("inputs")
    manifest_record = inputs.get("bank_manifest") if isinstance(inputs, Mapping) else None
    if not isinstance(manifest_record, Mapping) or not isinstance(manifest_record.get("path"), str):
        raise CellBuilderError("preflight lacks bank-manifest path/hash input")
    supplied_record = _path_record(bank_manifest)
    normalized_record = {"path": str(Path(manifest_record["path"]).resolve()), "sha256": manifest_record.get("sha256")}
    if supplied_record != normalized_record:
        raise CellBuilderError("supplied bank manifest is not the exact preflight path/hash input")
    if payload.get("bank_manifest") != normalized_record:
        raise CellBuilderError("authorization bank-manifest binding mismatch")
    binding = {
        "authorization": _path_record(path),
        "preflight": _path_record(preflight_path),
        "bank_manifest": supplied_record,
    }
    return payload, binding


def _dry_run(manifest: Path, synthetic: bool) -> Mapping[str, Any]:
    banks = load_bank_manifest(manifest, synthetic=synthetic)
    return {
        "dry_run": True,
        "mode": "synthetic" if synthetic else "real",
        "banks": list(banks),
        "cohort_size": len(banks["Q0"].ordered_ids),
        "cells_planned": 45,
        "no_model_inference": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--build", action="store_true", help="perform numerical cached-cell construction")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = args.bank_manifest.resolve()
        if not args.build:
            print(json.dumps(_dry_run(manifest, args.synthetic), indent=2, sort_keys=True))
            return 0
        if args.output is None:
            raise CellBuilderError("--build requires --output")
        output = args.output.resolve()
        if args.synthetic:
            if args.authorization is not None:
                raise CellBuilderError("synthetic construction must not use real authorization")
            cells = build_from_manifest(manifest, synthetic=True, device=args.device)
            _exclusive_json(output, {"schema": "full_spd_cached_cells/v1", "mode": "synthetic", "cells": cells})
            return 0
        if args.authorization is None:
            raise CellBuilderError("real construction requires --authorization")
        if REAL_MARKER.exists():
            raise CellBuilderError("real cell construction marker exists; refusing a second construction")
        authorization, binding = _authorization(args.authorization.resolve(), manifest)
        # Reserve the one-shot before any numerical construction or potential training.
        _exclusive_json(REAL_MARKER, {
            "schema": "full_spd_real_cell_construction_marker/v1",
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "authorization": str(args.authorization.resolve()),
            "authorization": binding["authorization"],
            "preflight": binding["preflight"],
            "bank_manifest": binding["bank_manifest"],
            "git_commit": authorization["git_commit"],
            "output": str(output),
        })
        cells = build_from_manifest(manifest, synthetic=False, device=args.device, git_commit=authorization["git_commit"], real_binding=binding)
        _exclusive_json(output, {"schema": "full_spd_cached_cells/v1", "mode": "real", "authorization": binding["authorization"], "preflight": binding["preflight"], "bank_manifest": binding["bank_manifest"], "cells": cells})
        return 0
    except CellBuilderError as exc:
        print(f"CELL BUILD FAILED CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
