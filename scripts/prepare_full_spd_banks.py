"""Prepare provenance-bound Q0/Q1/Q2 artifacts for the full-SPD experiment.

This script deliberately does *not* run by default.  ``--execute`` is required
before it creates an output directory or invokes ``run_experiments.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHAPES = {
    "T0_substrate.npy": (320, 8, 3),
    "T1_embeddings_raw.npy": (320, 2048),
    "T1_embeddings_per_bot.npy": (320, 8, 1536),
}
BANK_SOURCES = {
    "Q0": Path("probe_hypotheses.json"),
    "Q1": Path("config/prompt_banks/probe_hypotheses_paraphrase_a.json"),
    "Q2": Path("config/prompt_banks/probe_hypotheses_paraphrase_b.json"),
}
FRAMING_DECLARATION_BASIS = "human_audited_declaration_not_computational_proof"
DEFAULT_FRAMING_PROVENANCE = {
    "Q0": {
        "relation_to_Q0": "anchor",
        "independently_authored_without_q0_exposure": False,
        "author": "unspecified",
        "source_process": "existing project anchor bank",
    },
    "Q1": {
        "relation_to_Q0": "paraphrase_of_Q0",
        "independently_authored_without_q0_exposure": False,
        "author": "unspecified",
        "source_process": "direct paraphrase of Q0",
    },
    "Q2": {
        "relation_to_Q0": "paraphrase_of_Q0",
        "independently_authored_without_q0_exposure": False,
        "author": "unspecified",
        "source_process": "direct paraphrase of Q0",
    },
}


class PreparationError(RuntimeError):
    """A fail-closed preparation error."""


@dataclass(frozen=True)
class Bank:
    name: str
    source: Path
    raw_sha256: str
    pairs: tuple[tuple[str, str], ...]
    pairs_sha256: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(encoded.encode("utf-8"))


def parse_bank(name: str, source: Path) -> Bank:
    raw = source.read_bytes()
    try:
        values = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"{name}: invalid JSON bank: {source}") from exc
    if not isinstance(values, list) or len(values) != 16:
        raise PreparationError(f"{name}: expected exactly 16 strings, got {len(values) if isinstance(values, list) else 'non-list'}")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise PreparationError(f"{name}: every hypothesis must be a non-empty string")
    pairs = tuple((values[index], values[index + 1]) for index in range(0, 16, 2))
    canonical_pairs = [{"A": a, "B": b} for a, b in pairs]
    return Bank(name, source, sha256_bytes(raw), pairs, canonical_json_sha256(canonical_pairs))


def resolve_framing_provenance(source: Path | None) -> dict[str, dict[str, Any]]:
    """Return explicit human-audited framing declarations for all three banks."""

    if source is None:
        payload: Any = DEFAULT_FRAMING_PROVENANCE
    else:
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PreparationError(f"cannot read framing provenance: {source}") from exc
    if not isinstance(payload, Mapping) or set(payload) != set(BANK_SOURCES):
        raise PreparationError("framing provenance must contain exactly Q0, Q1, and Q2")
    resolved: dict[str, dict[str, Any]] = {}
    for bank in BANK_SOURCES:
        declaration = payload[bank]
        if not isinstance(declaration, Mapping):
            raise PreparationError(f"{bank} framing provenance must be an object")
        relation = declaration.get("relation_to_Q0")
        independent = declaration.get("independently_authored_without_q0_exposure")
        author = declaration.get("author")
        process = declaration.get("source_process")
        if not isinstance(relation, str) or not relation.strip():
            raise PreparationError(f"{bank} framing provenance requires relation_to_Q0")
        if type(independent) is not bool:
            raise PreparationError(
                f"{bank} framing provenance requires a boolean independently_authored_without_q0_exposure"
            )
        if not isinstance(author, str) or not isinstance(process, str):
            raise PreparationError(f"{bank} framing provenance author/source_process must be strings")
        resolved[bank] = {
            "relation_to_Q0": relation.strip(),
            "independently_authored_without_q0_exposure": independent,
            "author": author.strip(),
            "source_process": process.strip(),
            "declaration_basis": FRAMING_DECLARATION_BASIS,
        }
    return resolved


def runtime_yaml(bank: Bank) -> dict[str, Any]:
    return {
        "queries": {
            f"{bank.name.lower()}_pair_{index:02d}": {"A": a, "B": b}
            for index, (a, b) in enumerate(bank.pairs, start=1)
        }
    }


def load_corpus_identity(corpus: Path) -> tuple[str, str, list[dict[str, str]]]:
    articles: list[dict[str, str]] = []
    for line_number, line in enumerate(corpus.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise PreparationError(f"corpus contains a blank line at {line_number}")
        try:
            article = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PreparationError(f"corpus line {line_number} is invalid JSON") from exc
        article_id = article.get("canonical_id") or article.get("id")
        publisher = article.get("publisher") or article.get("source")
        if not isinstance(article_id, str) or not article_id or not isinstance(publisher, str) or not publisher:
            raise PreparationError(f"corpus line {line_number} lacks canonical ID/id or publisher/source")
        articles.append({"id": article_id, "publisher": publisher})
    if len(articles) != 320:
        raise PreparationError(f"expected exactly 320 corpus articles, got {len(articles)}")
    ids = [article["id"] for article in articles]
    if len(set(ids)) != len(ids):
        raise PreparationError("corpus article IDs are not unique")
    return sha256_file(corpus), canonical_json_sha256(articles), articles


def _as_int_map(value: Mapping[str, Any], name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise PreparationError(f"{name} must be an object")
    try:
        result = {str(key): int(item) for key, item in value.items()}
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"{name} contains a non-integer class ID") from exc
    if len(result) != 3 or len(set(result.values())) != 3:
        raise PreparationError(f"{name} must specify exactly three unique classes")
    if set(result.values()) != {0, 1, 2}:
        raise PreparationError(f"{name} class IDs must be exactly 0, 1, and 2")
    return result


def _canonical_class_name(label: str) -> str:
    normalized = "".join(character for character in label.lower() if character.isalnum())
    matches = [
        canonical
        for token, canonical in (
            ("entail", "entail"),
            ("neutral", "neutral"),
            ("contradict", "contradict"),
        )
        if token in normalized
    ]
    if len(matches) != 1:
        raise PreparationError(f"ambiguous or unknown NLI class label: {label!r}")
    return matches[0]


def _canonical_axis_order(labels_by_id: Mapping[int, str], source: str) -> list[str]:
    if set(labels_by_id) != {0, 1, 2}:
        raise PreparationError(f"{source} class IDs must be exactly 0, 1, and 2")
    class_names = [_canonical_class_name(labels_by_id[index]) for index in range(3)]
    if set(class_names) != {"entail", "neutral", "contradict"}:
        raise PreparationError(f"{source} must map exactly one entail, neutral, and contradict class")
    return class_names


def resolve_model_provenance(
    *, model_config: Path | None, label_map: str | None, model_name: str | None, model_revision: str | None
) -> dict[str, Any]:
    """Load an actual class map; never infer it from old checkpoint metadata."""
    if bool(model_config) == bool(label_map):
        raise PreparationError("provide exactly one of --model-config or --label-map")
    payload: Mapping[str, Any]
    source: str
    if model_config:
        try:
            payload = json.loads(model_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PreparationError(f"cannot read model config: {model_config}") from exc
        source = str(model_config)
        model_name = model_name or payload.get("_name_or_path") or payload.get("model_name")
        model_revision = model_revision or payload.get("_commit_hash") or payload.get("revision")
        label2id_value = payload.get("label2id")
        id2label_value = payload.get("id2label")
    else:
        try:
            raw_label_map = label_map or ""
            if raw_label_map.lstrip().startswith("{"):
                payload = json.loads(raw_label_map)
            else:
                payload = json.loads(Path(raw_label_map).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PreparationError("--label-map must be JSON text or a JSON file") from exc
        source = "--label-map"
        label2id_value = payload.get("label2id", payload)
        id2label_value = payload.get("id2label")
    if not isinstance(model_name, str) or not model_name or not isinstance(model_revision, str) or not model_revision:
        raise PreparationError("exact model name and revision are required")
    label2id = _as_int_map(label2id_value, "label2id")
    label2id_axis = {index: label for label, index in label2id.items()}
    label2id_class_names = _canonical_axis_order(label2id_axis, "label2id")
    if id2label_value is not None:
        if not isinstance(id2label_value, Mapping):
            raise PreparationError("id2label must be an object")
        try:
            id2label = {int(key): str(value) for key, value in id2label_value.items()}
        except (TypeError, ValueError) as exc:
            raise PreparationError("id2label contains a non-integer class ID") from exc
        id2label_class_names = _canonical_axis_order(id2label, "id2label")
        if id2label_class_names != label2id_class_names:
            raise PreparationError("label2id and id2label disagree about logit-axis semantics")
    else:
        id2label = label2id_axis
    return {
        "source": source,
        "model_name": model_name,
        "model_revision": model_revision,
        "label2id": label2id,
        "id2label": {str(key): value for key, value in sorted(id2label.items())},
        "class_names": label2id_class_names,
    }


def planned_command(*, runtime_yaml_path: Path, output_dir: Path, corpus: Path) -> list[str]:
    return [
        sys.executable,
        "run_experiments.py",
        "--corpus", str(corpus),
        "--mode", "cls_logits",
        "--seeds", "42",
        "--output-root", str(output_dir),
        "--kernel-type", "rbf",
        "--rks-dim", "2048",
        "--kernel-seed", "42",
        "--queries-config", str(runtime_yaml_path),
    ]


def missing_runner_requirements() -> list[str]:
    """Runner options requested by the design but absent from --help/source interface."""
    return []


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise PreparationError(f"refusing overwrite: {path}") from exc


def discover_artifacts(bank_dir: Path) -> dict[str, Path]:
    context = bank_dir / "bank_context.json"
    if not context.is_file():
        raise PreparationError(f"unbound output: missing preparer context in {bank_dir}")
    checkpoint_dir = bank_dir / "checkpoints" / "batch"
    artifacts: dict[str, Path] = {}
    for name in EXPECTED_SHAPES:
        path = checkpoint_dir / name
        if not path.is_file():
            raise PreparationError(f"missing runner artifact: {path}")
        artifacts[name] = path
    return artifacts


def validate_artifacts(artifacts: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, expected_shape in EXPECTED_SHAPES.items():
        path = artifacts[name]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != expected_shape:
            raise PreparationError(f"{name}: expected shape {expected_shape}, got {tuple(array.shape)}")
        if array.dtype != np.dtype("float32"):
            raise PreparationError(f"{name}: expected float32, got {array.dtype}")
        byteorder = array.dtype.byteorder
        endianness = (
            sys.byteorder if byteorder == "=" else "little" if byteorder == "<" else "big"
        )
        digest = sha256_file(path)
        records[name] = {
            "path": str(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "endianness": endianness,
            "sha256": digest,
            "raw_sha256": digest,
        }
    return records


def artifact_aliases(records: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "X": dict(records["T1_embeddings_raw.npy"]),
        "T0": dict(records["T0_substrate.npy"]),
        "T1": dict(records["T1_embeddings_per_bot.npy"]),
    }


def validate_distinct_t1_hashes(bank_records: Mapping[str, Mapping[str, Any]]) -> None:
    hashes = [record["T1_embeddings_raw.npy"]["sha256"] for record in bank_records.values()]
    if len(hashes) != 3 or len(set(hashes)) != 3:
        raise PreparationError("shared-X rejection: Q0/Q1/Q2 T1_embeddings_raw hashes must all differ")


def git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="create artifacts and invoke the runner")
    parser.add_argument("--corpus", type=Path, default=Path("data/controlled_N_dense_corpus.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/full_spd_banks"))
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--label-map")
    parser.add_argument("--model-name")
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--framing-provenance",
        type=Path,
        help="JSON human-audited declaration for Q0/Q1/Q2; defaults describe Q1/Q2 as Q0 paraphrases",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = (REPO_ROOT / args.corpus).resolve() if not args.corpus.is_absolute() else args.corpus
    output_root = (REPO_ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    if not corpus.is_file():
        raise PreparationError(f"corpus not found: {corpus}")
    banks = [parse_bank(name, REPO_ROOT / source) for name, source in BANK_SOURCES.items()]
    framing_source = args.framing_provenance
    if framing_source is not None and not framing_source.is_absolute():
        framing_source = (REPO_ROOT / framing_source).resolve()
    framing = resolve_framing_provenance(framing_source)
    corpus_sha256, article_order_hash, articles = load_corpus_identity(corpus)
    planned = {
        bank.name: planned_command(
            runtime_yaml_path=output_root / bank.name / "runtime_queries.yaml",
            output_dir=output_root / bank.name,
            corpus=corpus,
        )
        for bank in banks
    }
    if not args.execute:
        print(json.dumps({"dry_run": True, "planned_commands": planned, "missing_runner_requirements": missing_runner_requirements()}, indent=2))
        return 0
    model = resolve_model_provenance(
        model_config=args.model_config,
        label_map=args.label_map,
        model_name=args.model_name,
        model_revision=args.model_revision,
    )
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise PreparationError(f"refusing overwrite: {output_root}") from exc
    detailed_records: dict[str, dict[str, Any]] = {}
    bank_entries: dict[str, dict[str, Any]] = {}
    for bank in banks:
        bank_dir = output_root / bank.name
        bank_dir.mkdir(exist_ok=False)
        runtime_path = bank_dir / "runtime_queries.yaml"
        with runtime_path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(runtime_yaml(bank), handle, sort_keys=False, allow_unicode=True)
        context = {
            "bank": bank.name,
            "bank_source": str(bank.source),
            "bank_raw_sha256": bank.raw_sha256,
            "framing_provenance": framing[bank.name],
            "ordered_pairs_sha256": bank.pairs_sha256,
            "corpus": str(corpus),
            "corpus_raw_sha256": corpus_sha256,
            "ordered_article_id_publisher_sha256": article_order_hash,
            "article_order": articles,
            "model": model,
            "command": planned[bank.name],
            "git_sha": git_sha(),
            "python": sys.version,
            "platform": platform.platform(),
        }
        exclusive_json(bank_dir / "bank_context.json", context)
        subprocess.run(planned[bank.name], cwd=REPO_ROOT, check=True)
        artifacts = discover_artifacts(bank_dir)
        detailed = validate_artifacts(artifacts)
        detailed_records[bank.name] = detailed
        entry = {
            "prompt_hash": bank.raw_sha256,
            "q_bank_hash": bank.raw_sha256,
            "prompt_path": str(bank.source.resolve()),
            "framing_provenance": framing[bank.name],
            "ordered_pairs": [{"A": a, "B": b} for a, b in bank.pairs],
            "ordered_pairs_sha256": bank.pairs_sha256,
            "cohort": {
                "path": str(corpus),
                "sha256": corpus_sha256,
                "ordered_article_id_publisher_sha256": article_order_hash,
                "article_order": articles,
            },
            "model": model,
            "artifacts": artifact_aliases(detailed),
            "detailed_artifacts": detailed,
            "command": planned[bank.name],
            "git_sha": context["git_sha"],
            "runtime": {"python": context["python"], "platform": context["platform"]},
        }
        bank_entries[bank.name] = entry
        exclusive_json(bank_dir / "artifact_manifest.json", entry)
    validate_distinct_t1_hashes(detailed_records)
    manifest = {"schema": "full_spd_bank_manifest/v1", "banks": bank_entries}
    exclusive_json(output_root / "full_spd_bank_manifest.json", manifest)
    print(json.dumps({"prepared": str(output_root), "banks": bank_entries}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreparationError as exc:
        print(f"PREPARATION FAILED CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
