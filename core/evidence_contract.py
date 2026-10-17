from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from core.reflector_script_registry import (
    build_reflector_orchestration_contract,
    get_reflector_script_spec,
    reflector_payload_awareness,
)


NODE_MANIFEST_NAME = "node_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(json_safe(dict(payload)), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> Dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_state(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(repo_root or _repo_root())

    def run_git(args: Sequence[str]) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return None
        return proc.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    branch = run_git(["branch", "--show-current"])
    return {
        "repo_root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def _artifact_paths(artifacts: Mapping[str, Any]) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for name, raw_path in artifacts.items():
        if name == "node_manifest":
            continue
        if raw_path is None:
            continue
        path = Path(str(raw_path))
        if path.exists() and path.is_file():
            paths[str(name)] = path
    return paths


def _manifest_fingerprint_payload(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    script = manifest.get("script", {})
    return {
        "node_id": manifest.get("node_id"),
        "script_sha256": script.get("sha256") if isinstance(script, Mapping) else None,
        "outputs": manifest.get("outputs", {}),
        "config": manifest.get("config", {}),
        "upstream_script_ids": manifest.get("upstream_script_ids", []),
        "parent_manifest_fingerprints": manifest.get("parent_manifest_fingerprints", []),
    }


def _compute_manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    return sha256_payload(_manifest_fingerprint_payload(manifest))


def _parent_manifest_fingerprint(path: Path) -> Dict[str, Any]:
    path = Path(path)
    fingerprint = file_fingerprint(path)
    try:
        parent_manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        parent_manifest = {}
    if isinstance(parent_manifest, Mapping):
        fingerprint["script_id"] = parent_manifest.get("script_id")
        fingerprint["node_id"] = parent_manifest.get("node_id")
        fingerprint["manifest_fingerprint"] = parent_manifest.get("manifest_fingerprint")
    return fingerprint


def attach_reflector_awareness(payload: Dict[str, Any], script_id: str, output_dir: Path) -> None:
    artifacts = payload.setdefault("artifacts", {})
    artifacts["node_manifest"] = str(Path(output_dir) / NODE_MANIFEST_NAME)
    payload["evidence_node"] = reflector_payload_awareness(script_id)


def build_node_manifest(
    *,
    script_id: str,
    output_dir: Path,
    artifacts: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    generated_at_utc: Optional[str] = None,
    parent_manifests: Sequence[Path | str] = (),
    claim_matrix: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    spec = get_reflector_script_spec(script_id)
    script_path = spec.path
    output_paths = _artifact_paths(artifacts)
    present_outputs = {
        name: file_fingerprint(path)
        for name, path in sorted(output_paths.items())
    }
    missing_declared_outputs = [
        name
        for name in spec.produced_artifacts
        if name not in present_outputs and name != "node_manifest"
    ]
    parent_fingerprints = [
        _parent_manifest_fingerprint(Path(path))
        for path in parent_manifests
        if Path(path).exists() and Path(path).is_file()
    ]
    claim_rows = []
    if isinstance(claim_matrix, Mapping):
        raw_claims = claim_matrix.get("claims", [])
        if isinstance(raw_claims, list):
            for claim in raw_claims:
                if isinstance(claim, Mapping):
                    claim_rows.append(
                        {
                            "claim_id": claim.get("claim_id") or claim.get("variant") or claim.get("gate_variant"),
                            "pass": bool(claim.get("pass")),
                            "thesis_safe": bool(claim.get("thesis_safe", claim.get("pass", False))),
                        }
                    )
    manifest = {
        "schema_version": "1.0",
        "manifest_type": "reflector_evidence_node",
        "node_id": spec.script_id,
        "script_id": spec.script_id,
        "diagnostic_type": spec.diagnostic_type,
        "generated_at_utc": generated_at_utc or utc_now(),
        "output_dir": str(Path(output_dir)),
        "script": {
            "module": spec.module,
            "path": str(script_path),
            "sha256": sha256_file(script_path) if script_path.exists() else None,
        },
        "git": git_state(),
        "execution": {
            "argv": list(sys.argv),
        },
        "registry": reflector_payload_awareness(script_id),
        "config": json_safe(dict(config or {})),
        "declared_outputs": list(spec.produced_artifacts),
        "outputs": present_outputs,
        "missing_declared_outputs": missing_declared_outputs,
        "parent_manifest_fingerprints": parent_fingerprints,
        "upstream_script_ids": list(spec.upstream_script_ids),
        "claim_families": list(spec.claim_families),
        "claim_rows": claim_rows,
        "pass": not missing_declared_outputs,
    }
    manifest["manifest_fingerprint"] = _compute_manifest_fingerprint(manifest)
    return manifest


def write_node_manifest(
    *,
    script_id: str,
    output_dir: Path,
    artifacts: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    generated_at_utc: Optional[str] = None,
    parent_manifests: Sequence[Path | str] = (),
    claim_matrix: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    manifest = build_node_manifest(
        script_id=script_id,
        output_dir=output_dir,
        artifacts=artifacts,
        config=config,
        generated_at_utc=generated_at_utc,
        parent_manifests=parent_manifests,
        claim_matrix=claim_matrix,
    )
    manifest_path = output_dir / NODE_MANIFEST_NAME
    manifest_path.write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_node_manifest(manifest_path: Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    failures = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "manifest_path": str(manifest_path),
            "pass": False,
            "failure_reasons": [f"manifest_unreadable:{exc}"],
        }
    script_id = str(manifest.get("script_id") or "")
    try:
        spec = get_reflector_script_spec(script_id)
    except Exception:
        spec = None
        failures.append(f"unknown_script_id:{script_id}")
    if manifest.get("manifest_type") != "reflector_evidence_node":
        failures.append("wrong_manifest_type")
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        failures.append("outputs_not_mapping")
        outputs = {}
    for name, record in outputs.items():
        if not isinstance(record, Mapping):
            failures.append(f"bad_output_record:{name}")
            continue
        path = Path(str(record.get("path") or ""))
        if not path.exists():
            failures.append(f"missing_output:{name}")
            continue
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            failures.append(f"sha256_mismatch:{name}")
    if spec is not None:
        declared = set(manifest.get("declared_outputs") or [])
        expected = set(spec.produced_artifacts)
        if not expected.issubset(declared):
            failures.append("declared_outputs_do_not_cover_registry")
        if list(manifest.get("upstream_script_ids") or []) != list(spec.upstream_script_ids):
            failures.append("upstream_registry_mismatch")
    missing_declared = manifest.get("missing_declared_outputs") or []
    if missing_declared:
        failures.append("missing_declared_outputs")
    expected_manifest_fingerprint = _compute_manifest_fingerprint(manifest)
    if manifest.get("manifest_fingerprint") != expected_manifest_fingerprint:
        failures.append("manifest_fingerprint_mismatch")
    return {
        "manifest_path": str(manifest_path),
        "script_id": script_id,
        "pass": not failures,
        "failure_reasons": failures,
        "manifest": manifest,
    }


def verify_manifest_chain(
    manifest_paths: Iterable[Path | str],
    *,
    require_dependencies: bool = True,
) -> Dict[str, Any]:
    rows = [verify_node_manifest(Path(path)) for path in manifest_paths]
    row_by_script = {str(row.get("script_id")): row for row in rows if row.get("script_id")}
    present = set(row_by_script)
    failures = []
    for row in rows:
        if not bool(row.get("pass")):
            failures.append(f"node_failed:{row.get('script_id')}")
        manifest = row.get("manifest", {})
        if require_dependencies and isinstance(manifest, Mapping):
            child_id = str(manifest.get("script_id") or "")
            parent_links = {
                str(link.get("script_id")): link
                for link in manifest.get("parent_manifest_fingerprints", []) or []
                if isinstance(link, Mapping) and link.get("script_id")
            }
            for dep in manifest.get("upstream_script_ids", []) or []:
                if dep not in present:
                    failures.append(f"missing_dependency:{child_id}:{dep}")
                    continue
                if dep not in parent_links:
                    failures.append(f"missing_parent_link:{child_id}:{dep}")
                    continue
                parent_row = row_by_script[str(dep)]
                parent_manifest = parent_row.get("manifest", {})
                parent_path = Path(str(parent_row.get("manifest_path") or ""))
                parent_link = parent_links[str(dep)]
                if parent_path.exists() and parent_link.get("sha256") != sha256_file(parent_path):
                    failures.append(f"parent_manifest_sha_mismatch:{child_id}:{dep}")
                expected_parent_fingerprint = (
                    parent_manifest.get("manifest_fingerprint")
                    if isinstance(parent_manifest, Mapping)
                    else None
                )
                if parent_link.get("manifest_fingerprint") != expected_parent_fingerprint:
                    failures.append(f"parent_manifest_fingerprint_mismatch:{child_id}:{dep}")
    contract = build_reflector_orchestration_contract(
        manifest_paths=[row["manifest_path"] for row in rows],
        metadata={"verified_node_count": len(rows), "require_dependencies": require_dependencies},
    )
    return {
        "schema_version": "1.0",
        "verification_type": "reflector_evidence_chain",
        "generated_at_utc": utc_now(),
        "pass": not failures,
        "failure_reasons": failures,
        "node_count": len(rows),
        "nodes": rows,
        "orchestration_contract": contract,
    }
