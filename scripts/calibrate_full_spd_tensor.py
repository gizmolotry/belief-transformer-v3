"""Write-once CLI for deterministic pre-real full-SPD synthetic calibration."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from core.full_spd_geometry import canonical_json_sha256
from core.full_spd_simulation import (
    DEFAULT_RECORDS_PER_KIND,
    calibrate_synthetic_records,
    frozen_configuration,
    generate_simulation_records,
    run_learned_potential_recovery,
)


RECORDS_FILENAME = "simulation_records.json"
CALIBRATION_FILENAME = "calibration_manifest.json"


class CalibrationCLIError(RuntimeError):
    """Fail-closed CLI error."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-per-kind", type=int, default=DEFAULT_RECORDS_PER_KIND)
    parser.add_argument("--device", default="cpu", help="cpu or an available CUDA device")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually run synthetic simulations and create write-once JSON outputs",
    )
    return parser


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _exclusive_json(path: Path, payload: Any) -> str:
    raw = _canonical_bytes(payload)
    with path.open("xb") as handle:
        handle.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": "execute" if args.execute else "dry-run",
        "real_artifacts_read": False,
        "records_per_kind": int(args.records_per_kind),
        "total_records": 4 * int(args.records_per_kind),
        "output_files": [
            str(args.output_dir / RECORDS_FILENAME),
            str(args.output_dir / CALIBRATION_FILENAME),
        ],
        "configuration": frozen_configuration(),
    }


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    if args.records_per_kind <= 0:
        raise CalibrationCLIError("--records-per-kind must be positive")
    plan = _plan(args)
    if not args.execute:
        return plan

    records_path = args.output_dir / RECORDS_FILENAME
    calibration_path = args.output_dir / CALIBRATION_FILENAME
    existing = [str(path) for path in (records_path, calibration_path) if path.exists()]
    if existing:
        raise CalibrationCLIError(f"exclusive outputs already exist: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = generate_simulation_records(records_per_kind=args.records_per_kind)
    recovery = run_learned_potential_recovery(device=args.device)
    calibration = calibrate_synthetic_records(records)

    records_payload = {
        "schema": "full_spd_synthetic_records/v1",
        "configuration": frozen_configuration(),
        "records_per_kind": args.records_per_kind,
        "records": [asdict(record) for record in records],
    }
    records_payload["content_hash"] = canonical_json_sha256(records_payload)
    records_file_sha256 = _exclusive_json(records_path, records_payload)

    calibration_payload = {
        "schema": "full_spd_synthetic_calibration/v1",
        "real_artifacts_read": False,
        "records_file": RECORDS_FILENAME,
        "records_file_sha256": records_file_sha256,
        "records_content_hash": records_payload["content_hash"],
        "configuration": frozen_configuration(),
        "recovery": asdict(recovery),
        "calibration": asdict(calibration.thresholds),
        "rates": {
            "catch_rate": calibration.catch_rate,
            "planted_retention": calibration.planted_retention,
            "type_i_error": calibration.type_i_error,
            "power": calibration.power,
        },
    }
    calibration_payload["content_hash"] = canonical_json_sha256(calibration_payload)
    _exclusive_json(calibration_path, calibration_payload)
    return calibration_payload


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(argv)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
