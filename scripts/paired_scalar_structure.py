import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Include repository root in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_utils import extract_article_text
from core.nli_extraction import _split_into_paragraphs, _normalize_weights

# Import analysis helpers
import scripts.exploratory_internal_structure_analysis as expl
import scripts.q0_q1_robustness as rob

# Constants
MODEL_PATH = "microsoft/deberta-v2-xlarge-mnli"
SEED = 42
MAX_LENGTH = 512
PARAGRAPH_WEIGHTS = [0.5, 0.3, 0.2]
DEFAULT_PERMUTATIONS = 10_000


class PairedScalarInputError(ValueError):
    """Raised when a run cannot satisfy its declared input contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_array(value: Any, label: str, *, shape_tail: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PairedScalarInputError(f"{label} is not numeric") from exc
    if array.size == 0 or not np.isfinite(array).all():
        raise PairedScalarInputError(f"{label} must be nonempty and finite")
    if shape_tail is not None and array.shape[-len(shape_tail):] != shape_tail:
        raise PairedScalarInputError(f"{label} must end with shape {shape_tail}, got {array.shape}")
    return array


def scores_from_logits(
    logits_a: Any,
    logits_b: Any,
    *,
    entailment_index: int = 2,
    contradiction_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return A-side, B-side, and paired scalar scores without conflating them."""

    a = _finite_array(logits_a, "A logits", shape_tail=(8, 3))
    b = _finite_array(logits_b, "B logits", shape_tail=(8, 3))
    if a.shape != b.shape:
        raise PairedScalarInputError(f"A/B logit shapes differ: {a.shape} versus {b.shape}")
    if (entailment_index, contradiction_index) != (2, 0):
        raise PairedScalarInputError("expected contradiction/entailment class indices 0 and 2")
    probabilities_a = expl.stable_softmax(a)
    probabilities_b = expl.stable_softmax(b)
    score_a = probabilities_a[..., entailment_index] - probabilities_a[..., contradiction_index]
    score_b = probabilities_b[..., entailment_index] - probabilities_b[..., contradiction_index]
    return score_a, score_b, score_a - score_b


def full_pca_spectrum(matrix: Any) -> np.ndarray:
    array = _finite_array(matrix, "PCA matrix", shape_tail=(8,))
    if array.ndim != 2 or array.shape[0] < 3:
        raise PairedScalarInputError("PCA matrix must contain at least three article rows")
    singular_values = np.linalg.svd(
        array - array.mean(axis=0, keepdims=True), full_matrices=False, compute_uv=False
    )
    eigenvalues = singular_values**2 / (array.shape[0] - 1)
    total = float(eigenvalues.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise PairedScalarInputError("PCA matrix has no finite positive variance")
    return eigenvalues / total


def load_checkpoint(path: Path) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    """Load and validate an aggregated-logit checkpoint without trusting its filename."""

    ids: list[str] = []
    values = {bank: {side: [] for side in ("A", "B")} for bank in ("q0", "q1")}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PairedScalarInputError(f"could not read checkpoint: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PairedScalarInputError(f"checkpoint line {line_number} is invalid JSON") from exc
        canonical_id = row.get("canonical_id") if isinstance(row, Mapping) else None
        if not isinstance(canonical_id, str) or not canonical_id.strip():
            raise PairedScalarInputError(f"checkpoint line {line_number} lacks canonical_id")
        canonical_id = canonical_id.strip()
        if canonical_id in ids:
            raise PairedScalarInputError(f"duplicate checkpoint canonical_id: {canonical_id}")
        ids.append(canonical_id)
        for bank in ("q0", "q1"):
            bank_value = row.get(bank)
            if not isinstance(bank_value, Mapping):
                raise PairedScalarInputError(f"checkpoint line {line_number} lacks {bank}")
            for side in ("A", "B"):
                array = _finite_array(
                    bank_value.get(side), f"line {line_number} {bank}/{side}", shape_tail=(8, 3)
                )
                if array.shape != (8, 3):
                    raise PairedScalarInputError(
                        f"line {line_number} {bank}/{side} must have shape (8, 3), got {array.shape}"
                    )
                values[bank][side].append(array)
    if not ids:
        raise PairedScalarInputError("checkpoint is empty")
    stacked = {
        bank: {side: np.stack(rows, axis=0) for side, rows in sides.items()}
        for bank, sides in values.items()
    }
    return ids, stacked

def load_articles(path: Path, expected_n: int | None = None) -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PairedScalarInputError(f"could not read corpus: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            article = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PairedScalarInputError(f"corpus line {line_number} is invalid JSON") from exc
        if not isinstance(article, dict):
            raise PairedScalarInputError(f"corpus line {line_number} is not an object")
        canonical_id = article.get("canonical_id")
        if not isinstance(canonical_id, str) or not canonical_id.strip():
            raise PairedScalarInputError(f"corpus line {line_number} lacks canonical_id")
        canonical_id = canonical_id.strip()
        if canonical_id in seen_ids:
            raise PairedScalarInputError(f"duplicate corpus canonical_id: {canonical_id}")
        seen_ids.add(canonical_id)
        try:
            text = extract_article_text(article)
        except ValueError as exc:
            raise PairedScalarInputError(f"corpus line {line_number} has no article text") from exc
        publisher = article.get("publisher") or article.get("source")
        if not isinstance(publisher, str) or not publisher.strip():
            raise PairedScalarInputError(f"corpus line {line_number} lacks publisher/source")
        articles.append({**article, "canonical_id": canonical_id, "_text": text, "_publisher": publisher.strip()})
    if not articles:
        raise PairedScalarInputError("corpus is empty")
    if expected_n is not None and len(articles) != expected_n:
        raise PairedScalarInputError(f"expected {expected_n} articles, got {len(articles)}")
    return articles


def load_prompt_pairs(q0_path: Path, q1_path: Path) -> dict[str, list[tuple[str, str]]]:
    try:
        q0_payload = yaml.safe_load(q0_path.read_text(encoding="utf-8"))
        q1_payload = json.loads(q1_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise PairedScalarInputError("could not parse prompt banks") from exc
    queries = q0_payload.get("queries") if isinstance(q0_payload, Mapping) else None
    if not isinstance(queries, Mapping):
        raise PairedScalarInputError("Q0 must contain an ordered queries mapping")
    q0_pairs: list[tuple[str, str]] = []
    for name, pair in queries.items():
        if not isinstance(name, str) or not isinstance(pair, Mapping):
            raise PairedScalarInputError("Q0 entries must be named mappings")
        q0_pairs.append((pair.get("A"), pair.get("B")))
    if not isinstance(q1_payload, list) or len(q1_payload) != 16:
        raise PairedScalarInputError("Q1 must be a 16-string alternating A/B list")
    q1_pairs = [(q1_payload[index], q1_payload[index + 1]) for index in range(0, 16, 2)]
    banks = {"q0": q0_pairs, "q1": q1_pairs}
    for bank, pairs in banks.items():
        if len(pairs) != 8:
            raise PairedScalarInputError(f"{bank} must contain exactly eight pairs")
        for index, (hypothesis_a, hypothesis_b) in enumerate(pairs):
            if not all(isinstance(value, str) and value.strip() for value in (hypothesis_a, hypothesis_b)):
                raise PairedScalarInputError(f"{bank} pair {index} must contain nonempty A/B strings")
            if hypothesis_a.strip() == hypothesis_b.strip():
                raise PairedScalarInputError(f"{bank} pair {index} has identical A/B hypotheses")
    return banks


def _class_indices(label2id: Mapping[str, int]) -> tuple[int, int]:
    matches: dict[str, int] = {}
    for label, index in label2id.items():
        lowered = str(label).lower()
        if "contradict" in lowered:
            matches["contradiction"] = int(index)
        elif "entail" in lowered:
            matches["entailment"] = int(index)
        elif "neutral" in lowered:
            matches["neutral"] = int(index)
    if matches != {"contradiction": 0, "neutral": 1, "entailment": 2}:
        raise PairedScalarInputError(f"unexpected NLI class mapping: {dict(label2id)}")
    return matches["entailment"], matches["contradiction"]


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "pca_helper": Path(expl.__file__).resolve(),
        "robustness_helper": Path(rob.__file__).resolve(),
        "data_helper": ROOT / "core" / "data_utils.py",
        "extraction_helper": ROOT / "core" / "nli_extraction.py",
        "protocol": ROOT / "paper" / "paired_scalar_structure_protocol_20260905.md",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _model_record(model_path: str) -> dict[str, Any]:
    requested = Path(model_path)
    if requested.is_dir():
        snapshot = requested.resolve()
    else:
        cached_config = try_to_load_from_cache(model_path, "config.json")
        if not isinstance(cached_config, str):
            raise PairedScalarInputError(
                f"model {model_path!r} is not fully resolved in the local Hugging Face cache"
            )
        snapshot = Path(cached_config).resolve().parent
    files = sorted(path for path in snapshot.iterdir() if path.is_file())
    if not any(path.name == "config.json" for path in files):
        raise PairedScalarInputError("resolved model snapshot lacks config.json")
    if not any(path.name in {"pytorch_model.bin", "model.safetensors", "model.safetensors.index.json"} for path in files):
        raise PairedScalarInputError("resolved model snapshot lacks model weights")
    return {
        "requested_path_or_id": model_path,
        "resolved_snapshot": str(snapshot),
        "local_files_only": True,
        "files_sha256": {path.name: sha256_file(path) for path in files},
    }


def _validate_pre_run_manifest(
    manifest_path: Path,
    *,
    articles: Sequence[Mapping[str, Any]],
    banks: Mapping[str, Sequence[tuple[str, str]]],
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedScalarInputError(f"could not parse pre-run manifest: {manifest_path}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("status") != "frozen_before_extraction":
        raise PairedScalarInputError("pre-run manifest lacks frozen_before_extraction status")

    manifest_inputs = manifest.get("inputs")
    if not isinstance(manifest_inputs, Mapping):
        raise PairedScalarInputError("pre-run manifest lacks input records")
    for name in ("corpus", "q0", "q1"):
        record = manifest_inputs.get(name)
        if not isinstance(record, Mapping) or record.get("sha256") != input_hashes[name]:
            raise PairedScalarInputError(f"pre-run manifest {name} hash does not match analyzed input")

    expected_articles = [
        {
            "canonical_id": article["canonical_id"],
            "publisher": article["_publisher"],
            "text_sha256": hashlib.sha256(article["_text"].encode("utf-8")).hexdigest(),
        }
        for article in articles
    ]
    if manifest.get("articles") != expected_articles:
        raise PairedScalarInputError("pre-run manifest article identities/text hashes do not match corpus")
    expected_pairs = {name: [list(pair) for pair in pairs] for name, pairs in banks.items()}
    if manifest.get("prompt_pairs") != expected_pairs:
        raise PairedScalarInputError("pre-run manifest prompt pairs do not match analyzed banks")

    model = manifest.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("files_sha256"), Mapping):
        raise PairedScalarInputError("pre-run manifest lacks model-file hashes")
    snapshot_value = model.get("resolved_snapshot")
    if not isinstance(snapshot_value, str):
        raise PairedScalarInputError("pre-run manifest lacks resolved model snapshot")
    snapshot = Path(snapshot_value)
    for name, expected_hash in model["files_sha256"].items():
        if not isinstance(name, str) or Path(name).name != name or not isinstance(expected_hash, str):
            raise PairedScalarInputError("pre-run manifest contains an invalid model-file record")
        model_file = snapshot / name
        if not model_file.is_file() or sha256_file(model_file) != expected_hash:
            raise PairedScalarInputError(f"model file changed after manifest freeze: {name}")

    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or not all(
        isinstance(name, str) and isinstance(value, str) for name, value in source_hashes.items()
    ):
        raise PairedScalarInputError("pre-run manifest lacks extraction-source hashes")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "status": manifest["status"],
        "model": model,
        "extraction_source_sha256": dict(source_hashes),
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _pre_run_manifest(
    *,
    corpus_path: Path,
    q0_path: Path,
    q1_path: Path,
    articles: Sequence[Mapping[str, Any]],
    banks: Mapping[str, Sequence[tuple[str, str]]],
    model_path: str,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    return {
        "status": "frozen_before_extraction",
        "claim_scope": "exploratory paired-scalar measurement; not hidden-state displacement",
        "inputs": {
            "corpus": {"path": str(corpus_path), "sha256": sha256_file(corpus_path)},
            "q0": {"path": str(q0_path), "sha256": sha256_file(q0_path)},
            "q1": {"path": str(q1_path), "sha256": sha256_file(q1_path)},
        },
        "articles": [
            {
                "canonical_id": article["canonical_id"],
                "publisher": article["_publisher"],
                "text_sha256": hashlib.sha256(article["_text"].encode("utf-8")).hexdigest(),
            }
            for article in articles
        ],
        "prompt_pairs": {bank: [list(pair) for pair in pairs] for bank, pairs in banks.items()},
        "model": _model_record(model_path),
        "preprocessing": {
            "max_length": MAX_LENGTH,
            "truncation": "only_first",
            "paragraph_limit": 3,
            "paragraph_weights": PARAGRAPH_WEIGHTS,
            "aggregate": "weighted arithmetic mean of logits before softmax",
            "score": "s=P(entailment)-P(contradiction); paired scalar c=s_A-s_B",
        },
        "runtime": {"batch_size": batch_size, "device": device, "seed": SEED},
        "source_sha256": _source_hashes(),
    }


def extract_features(
    corpus_path: Path,
    q0_path: Path,
    q1_path: Path,
    output_dir: Path,
    batch_size: int,
    device: str,
    model_path: str = MODEL_PATH,
    expected_n: int | None = 320,
) -> Path:
    if batch_size < 1:
        raise PairedScalarInputError("batch_size must be positive")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    articles = load_articles(corpus_path, expected_n=expected_n)
    banks = load_prompt_pairs(q0_path, q1_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        output_dir / "pre_run_manifest.json",
        _pre_run_manifest(
            corpus_path=corpus_path,
            q0_path=q0_path,
            q1_path=q1_path,
            articles=articles,
            banks=banks,
            model_path=model_path,
            batch_size=batch_size,
            device=device,
        ),
    )

    print("Loading model and tokenizer from local files...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()
    _class_indices(model.config.label2id)

    def run_batches(texts: list[str], hypotheses: list[str]) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start : start + batch_size],
                hypotheses[start : start + batch_size],
                padding=True,
                truncation="only_first",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                logits = model(**inputs).logits.detach().cpu().numpy()
            pieces.append(_finite_array(logits, "model logits", shape_tail=(3,)))
        return np.concatenate(pieces, axis=0)

    checkpoint_path = output_dir / "extraction_checkpoint.jsonl"
    with checkpoint_path.open("x", encoding="utf-8", newline="\n") as checkpoint:
        for article_index, article in enumerate(articles):
            paragraphs = _split_into_paragraphs(article["_text"], len(PARAGRAPH_WEIGHTS)) or [article["_text"]]
            weights = np.asarray(_normalize_weights(PARAGRAPH_WEIGHTS, len(paragraphs)), dtype=np.float64)
            bank_outputs: dict[str, dict[str, list[list[float]]]] = {}
            for bank, pairs in banks.items():
                texts: list[str] = []
                hypotheses: list[str] = []
                coordinates: list[tuple[int, int, int]] = []
                for pair_index, pair in enumerate(pairs):
                    for side_index, hypothesis in enumerate(pair):
                        for paragraph_index, paragraph in enumerate(paragraphs):
                            texts.append(paragraph)
                            hypotheses.append(hypothesis)
                            coordinates.append((pair_index, side_index, paragraph_index))
                raw = run_batches(texts, hypotheses)
                per_paragraph = np.empty((8, 2, len(paragraphs), 3), dtype=np.float64)
                for coordinate, logits in zip(coordinates, raw, strict=True):
                    per_paragraph[coordinate] = logits
                aggregated = np.sum(per_paragraph * weights[None, None, :, None], axis=2)
                bank_outputs[bank] = {"A": aggregated[:, 0].tolist(), "B": aggregated[:, 1].tolist()}
            checkpoint.write(
                json.dumps(
                    {
                        "canonical_id": article["canonical_id"],
                        "publisher": article["_publisher"],
                        "text_sha256": hashlib.sha256(article["_text"].encode("utf-8")).hexdigest(),
                        "paragraph_weights": weights.tolist(),
                        **bank_outputs,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            checkpoint.flush()
            if (article_index + 1) % 10 == 0:
                print(f"Processed {article_index + 1}/{len(articles)} articles...")
    return checkpoint_path


def analyze_and_compare(
    q0_scores: np.ndarray,
    q1_scores: np.ndarray,
    rng: np.random.Generator,
    perms: int = DEFAULT_PERMUTATIONS,
) -> dict[str, Any]:
    pca0 = expl.deterministic_pca(q0_scores)
    pca1 = expl.deterministic_pca(q1_scores)
    
    obs_var0 = float(np.sum(pca0["explained_variance_ratio"]))
    obs_var1 = float(np.sum(pca1["explained_variance_ratio"]))
    
    var0_null = rob._variance_null(q0_scores, obs_var0, perms, rng)
    var1_null = rob._variance_null(q1_scores, obs_var1, perms, rng)
    
    var0_res = rob._variance_result(obs_var0, var0_null)
    var1_res = rob._variance_result(obs_var1, var1_null)
    
    obs_proc = rob._procrustes_disparity(pca0["scores"], pca1["scores"])
    proc_null = rob._procrustes_null(pca0["scores"], pca1["scores"], perms, rng)
    proc_res = rob._procrustes_result(obs_proc, proc_null)
    
    distances = rob._distance_result(pca0["scores"], pca1["scores"])
    angles = np.asarray(rob.subspace_angles(pca0["loadings"].T, pca1["loadings"].T))
    
    return {
        "q0_variance_ratio": obs_var0,
        "q1_variance_ratio": obs_var1,
        "q0_variance_null": var0_res,
        "q1_variance_null": var1_res,
        "procrustes": proc_res,
        "pairwise_distance_spearman": distances,
        "principal_angles_radians": [float(a) for a in angles],
        "principal_angles_degrees": [float(np.degrees(a)) for a in angles],
        "q0_full_explained_variance_ratio": [float(x) for x in full_pca_spectrum(q0_scores)],
        "q1_full_explained_variance_ratio": [float(x) for x in full_pca_spectrum(q1_scores)],
        "q0_loadings": [[float(x) for x in row] for row in pca0["loadings"]],
        "q1_loadings": [[float(x) for x in row] for row in pca1["loadings"]],
    }


def build_checkpoint_analysis(
    checkpoint_path: Path,
    corpus_path: Path,
    q0_path: Path,
    q1_path: Path,
    *,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 42,
    expected_n: int | None = 320,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if permutations < 1 or isinstance(permutations, bool):
        raise PairedScalarInputError("permutations must be a positive integer")
    ids, logits = load_checkpoint(checkpoint_path)
    articles = load_articles(corpus_path, expected_n=expected_n)
    banks = load_prompt_pairs(q0_path, q1_path)
    corpus_ids = [article["canonical_id"] for article in articles]
    if ids != corpus_ids:
        raise PairedScalarInputError("checkpoint IDs do not exactly match corpus order")
    initial_hashes = {
        "checkpoint": sha256_file(checkpoint_path),
        "corpus": sha256_file(corpus_path),
        "q0": sha256_file(q0_path),
        "q1": sha256_file(q1_path),
    }
    manifest_record = None
    if manifest_path is not None:
        manifest_record = _validate_pre_run_manifest(
            manifest_path,
            articles=articles,
            banks=banks,
            input_hashes=initial_hashes,
        )
    score_sets: dict[str, dict[str, np.ndarray]] = {"A_side": {}, "B_side": {}, "paired_scalar": {}}
    for bank in ("q0", "q1"):
        score_a, score_b, paired = scores_from_logits(logits[bank]["A"], logits[bank]["B"])
        score_sets["A_side"][bank] = score_a
        score_sets["B_side"][bank] = score_b
        score_sets["paired_scalar"][bank] = paired
    rng = np.random.default_rng(seed)
    results = {
        channel: analyze_and_compare(values["q0"], values["q1"], rng, perms=permutations)
        for channel, values in (("A_side", score_sets["A_side"]), ("paired_scalar", score_sets["paired_scalar"]))
    }
    final_hashes = {
        "checkpoint": sha256_file(checkpoint_path),
        "corpus": sha256_file(corpus_path),
        "q0": sha256_file(q0_path),
        "q1": sha256_file(q1_path),
    }
    if initial_hashes != final_hashes:
        raise PairedScalarInputError("analyzed inputs changed during analysis")
    return {
        "schema": {"name": "belief_transformer.paired_scalar_structure", "version": "1.0.0"},
        "status": (
            "fresh_manifest_bound_extraction_and_analysis"
            if manifest_record is not None
            else "exploratory_posthoc_analysis_of_existing_unmanifested_extraction"
        ),
        "claim_boundaries": {
            "primary_measurement": "paired scalar c=s_A-s_B",
            "not_equivalent_to": "hidden-state displacement phi_A-phi_B",
            "q1": "paraphrase-A bank; not independent authorship",
            "historical_results": "preserved separately and not retroactively replaced",
            "extraction_provenance": (
                "The extraction is bound to a validated pre-run manifest."
                if manifest_record is not None
                else (
                    "The existing checkpoint was produced without a hash-bound pre-run manifest. "
                    "Current hashes and exact article-ID order are verified post hoc, not at extraction time."
                )
            ),
        },
        "analysis": {"seed": seed, "permutations_per_null": permutations, "n_articles": len(ids)},
        "formulas": {
            "side_score": "s=P(entailment)-P(contradiction), softmax applied after paragraph-logit aggregation",
            "paired_scalar": "c=s_A-s_B",
            "pca": "covariance PCA on each column-centered article-by-eight score matrix",
            "variance_null": "independently permute each score column and refit PCA",
            "procrustes_null": "hold PCA fits fixed and permute only Q1 article-row correspondence",
        },
        "provenance": {
            "inputs": initial_hashes,
            "paths": {
                "checkpoint": str(checkpoint_path),
                "corpus": str(corpus_path),
                "q0": str(q0_path),
                "q1": str(q1_path),
            },
            "source_sha256": _source_hashes(),
            "pre_run_manifest": manifest_record,
            "runtime": {**expl.runtime_environment_versions(), "platform": platform.platform()},
            "article_ids_exactly_match_corpus_order": True,
        },
        "results": results,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract or analyze paired scalar NLI structure")
    parser.add_argument("--corpus", type=Path, default="data/controlled_N_dense_corpus.jsonl")
    parser.add_argument("--q0", type=Path, default="config/framing_queries.yaml")
    parser.add_argument("--q1", type=Path, default="config/prompt_banks/probe_hypotheses_paraphrase_a.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--expected-n", type=int, default=320)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.checkpoint is not None:
        if args.output_json is None or args.output_dir is not None:
            raise PairedScalarInputError("checkpoint analysis requires --output-json and forbids --output-dir")
        payload = build_checkpoint_analysis(
            args.checkpoint,
            args.corpus,
            args.q0,
            args.q1,
            permutations=args.permutations,
            seed=args.seed,
            expected_n=args.expected_n,
        )
        _write_json_exclusive(args.output_json, payload)
        print(f"Analysis written to {args.output_json}")
        return 0
    if args.output_dir is None or args.output_json is not None:
        raise PairedScalarInputError("fresh extraction requires --output-dir and forbids --output-json")
    started = time.time()
    checkpoint = extract_features(
        args.corpus,
        args.q0,
        args.q1,
        args.output_dir,
        args.batch_size,
        args.device,
        model_path=args.model_path,
        expected_n=args.expected_n,
    )
    payload = build_checkpoint_analysis(
        checkpoint,
        args.corpus,
        args.q0,
        args.q1,
        permutations=args.permutations,
        seed=args.seed,
        expected_n=args.expected_n,
        manifest_path=args.output_dir / "pre_run_manifest.json",
    )
    payload["analysis"]["extraction_and_analysis_runtime_seconds"] = time.time() - started
    _write_json_exclusive(args.output_dir / "paired_scalar_results.json", payload)
    print(f"Extraction and analysis written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PairedScalarInputError as exc:
        print(f"PairedScalarInputError: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
