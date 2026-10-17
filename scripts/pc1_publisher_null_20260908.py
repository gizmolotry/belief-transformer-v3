from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import paired_scalar_dimensionality_v2_20260907 as v2


paired = v2.paired
DRAW_COUNT = 10_000
ALPHA = 0.05
Q0_SEED = 20_260_908
Q1_SEED = 20_260_909
EXPECTED_ROWS = 320
EXPECTED_COLUMNS = 8
EXPECTED_PUBLISHERS = 16
EXPECTED_ROWS_PER_PUBLISHER = 20

PASS_CONCLUSION = (
    "dominant-component dependence detected under the specified publisher-block null."
)
FAIL_CONCLUSION = "not detected under the specified publisher-block null."
ALLOWED_CONCLUSIONS = frozenset((PASS_CONCLUSION, FAIL_CONCLUSION))

DEFAULT_CHECKPOINT = ROOT / "outputs" / "fresh_scalar_structure_run" / "extraction_checkpoint.jsonl"
DEFAULT_CORPUS = ROOT / "data" / "controlled_N_dense_corpus.jsonl"
DEFAULT_Q0 = ROOT / "config" / "framing_queries.yaml"
DEFAULT_Q1 = ROOT / "config" / "prompt_banks" / "probe_hypotheses_paraphrase_a.json"
DEFAULT_PAIRED_STRUCTURE_PROTOCOL = ROOT / "paper" / "paired_scalar_structure_protocol_20260905.md"
DEFAULT_PAIRED_STRUCTURE_RESULT = ROOT / "paper" / "paired_scalar_structure_results_20260906.json"
DEFAULT_V2_PROTOCOL = ROOT / "paper" / "paired_scalar_dimensionality_protocol_v2_20260907.md"
DEFAULT_V2_RESULT = ROOT / "paper" / "paired_scalar_dimensionality_results_v2_20260907.json"
DEFAULT_PROTOCOL = ROOT / "paper" / "pc1_publisher_null_protocol_20260908.md"
DEFAULT_OUTPUT = ROOT / "paper" / "pc1_publisher_null_results_20260908.json"


class PC1PublisherNullInputError(ValueError):
    """Raised when the frozen Gate 1 contract cannot be satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PC1PublisherNullInputError(f"could not hash required file: {path}") from exc
    return digest.hexdigest()


def repository_helper_paths() -> dict[str, Path]:
    """Return every repository helper source imported or called by this runner."""

    return {
        "imported_dimensionality_v2": Path(v2.__file__).resolve(),
        "imported_dimensionality_v1": Path(v2.v1.__file__).resolve(),
        "imported_paired_scalar_structure": Path(paired.__file__).resolve(),
        "imported_pca_helper": Path(paired.expl.__file__).resolve(),
        "imported_robustness_helper": Path(paired.rob.__file__).resolve(),
        "imported_data_helper": Path(paired.extract_article_text.__code__.co_filename).resolve(),
        "imported_extraction_helper": Path(
            paired._split_into_paragraphs.__code__.co_filename
        ).resolve(),
    }


def _hash_records(paths: Mapping[str, Path]) -> dict[str, str]:
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise PC1PublisherNullInputError(
            f"missing required files: {', '.join(sorted(missing))}"
        )
    return {name: sha256_file(path) for name, path in paths.items()}


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PC1PublisherNullInputError(f"could not parse {label}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise PC1PublisherNullInputError(f"{label} must contain a JSON object")
    return payload


def _require_hashes(
    bound: Any, expected: Mapping[str, str], *, label: str
) -> None:
    if not isinstance(bound, Mapping):
        raise PC1PublisherNullInputError(f"{label} lacks hash bindings")
    for name, expected_hash in expected.items():
        actual = bound.get(name)
        if not isinstance(actual, str) or actual.lower() != expected_hash.lower():
            raise PC1PublisherNullInputError(f"{label} does not bind current {name}")


def _validate_paired_structure_result(
    path: Path,
    *,
    input_hashes: Mapping[str, str],
    source_hashes: Mapping[str, str],
) -> None:
    payload = _read_json_object(path, "paired-scalar structure result")
    try:
        schema_name = payload["schema"]["name"]
        measurement = payload["claim_boundaries"]["primary_measurement"]
        provenance = payload["provenance"]
    except (KeyError, TypeError) as exc:
        raise PC1PublisherNullInputError(
            "paired-scalar structure result is malformed"
        ) from exc
    if schema_name != "belief_transformer.paired_scalar_structure":
        raise PC1PublisherNullInputError("paired-scalar structure result schema is not recognized")
    if measurement != "paired scalar c=s_A-s_B":
        raise PC1PublisherNullInputError(
            "paired-scalar structure result does not declare the corrected paired scalar"
        )
    _require_hashes(
        provenance.get("inputs"),
        {name: input_hashes[name] for name in ("checkpoint", "corpus", "q0", "q1")},
        label="paired-scalar structure result",
    )
    _require_hashes(
        provenance.get("source_sha256"),
        {
            "protocol": input_hashes["paired_scalar_structure_protocol"],
            "runner": source_hashes["imported_paired_scalar_structure"],
        },
        label="paired-scalar structure result source provenance",
    )


def _validate_v2_result(
    path: Path,
    *,
    input_hashes: Mapping[str, str],
    source_hashes: Mapping[str, str],
) -> None:
    payload = _read_json_object(path, "corrected dimensionality-v2 result")
    try:
        schema_name = payload["schema"]["name"]
        measurement = payload["claim_boundaries"]["measurement"]
        provenance = payload["provenance"]
    except (KeyError, TypeError) as exc:
        raise PC1PublisherNullInputError(
            "corrected dimensionality-v2 result is malformed"
        ) from exc
    if schema_name != "belief_transformer.paired_scalar_dimensionality_selection_v2":
        raise PC1PublisherNullInputError(
            "corrected dimensionality-v2 result schema is not recognized"
        )
    if measurement != "paired scalar c=s_A-s_B only":
        raise PC1PublisherNullInputError(
            "corrected dimensionality-v2 result does not declare the corrected paired scalar"
        )
    _require_hashes(
        provenance.get("input_sha256"),
        {
            **{name: input_hashes[name] for name in ("checkpoint", "corpus", "q0", "q1")},
            "contextual_paired_scalar_result": input_hashes["paired_scalar_structure_result"],
        },
        label="corrected dimensionality-v2 result",
    )
    _require_hashes(
        provenance.get("source_sha256"),
        {
            "v2_frozen_protocol": input_hashes["paired_scalar_dimensionality_v2_protocol"],
            "v2_runner": source_hashes["imported_dimensionality_v2"],
            "imported_v1_runner": source_hashes["imported_dimensionality_v1"],
            "paired_scalar_scoring_runner": source_hashes[
                "imported_paired_scalar_structure"
            ],
        },
        label="corrected dimensionality-v2 result source provenance",
    )


def _matrix(value: Any, label: str) -> np.ndarray:
    try:
        matrix = v2._matrix(value, label)
    except v2.DimensionalityV2InputError as exc:
        raise PC1PublisherNullInputError(str(exc)) from exc
    return matrix


def centered_unstandardized_matrix(value: Any, label: str) -> np.ndarray:
    matrix = _matrix(value, label)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if centered.shape != matrix.shape or not np.isfinite(centered).all():
        raise PC1PublisherNullInputError(f"{label} could not be centered")
    return centered


def sample_covariance_eigenvalues(value: Any) -> np.ndarray:
    matrix = centered_unstandardized_matrix(value, "sample-covariance matrix")
    covariance = matrix.T @ matrix / (matrix.shape[0] - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    if eigenvalues.shape != (EXPECTED_COLUMNS,) or not np.isfinite(eigenvalues).all():
        raise PC1PublisherNullInputError("sample covariance has an invalid eigenspectrum")
    if eigenvalues[0] <= 0.0:
        raise PC1PublisherNullInputError("sample covariance has no positive leading eigenvalue")
    return eigenvalues


def publisher_block_permutation(
    value: Any, publishers: Sequence[str], rng: np.random.Generator
) -> np.ndarray:
    if not isinstance(rng, np.random.Generator):
        raise PC1PublisherNullInputError("publisher-block null requires a local Generator")
    try:
        return v2.publisher_block_permutation(value, publishers, rng)
    except v2.DimensionalityV2InputError as exc:
        raise PC1PublisherNullInputError(str(exc)) from exc


def summarize_null(
    *, observed: float, null_values: Any, alpha: float = ALPHA
) -> dict[str, Any]:
    values = np.asarray(null_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise PC1PublisherNullInputError("null values must be a finite vector of length at least two")
    if not np.isfinite(observed):
        raise PC1PublisherNullInputError("observed lambda_1 must be finite")
    if not 0.0 < alpha < 1.0:
        raise PC1PublisherNullInputError("alpha must be strictly between zero and one")
    q95 = float(np.quantile(values, 0.95, method="higher"))
    order_one_based = math.ceil(0.95 * (values.size - 1)) + 1
    exceedances = int(np.count_nonzero(values >= observed))
    plus_one_p = (exceedances + 1) / (values.size + 1)
    detected = bool(observed > q95 and plus_one_p <= alpha)
    return {
        "observed_lambda_1": float(observed),
        "null_mean": float(values.mean()),
        "null_standard_deviation_ddof_1": float(values.std(ddof=1)),
        "null_minimum": float(values.min()),
        "null_maximum": float(values.max()),
        "null_q95": q95,
        "null_q95_method": "numpy.quantile(method='higher')",
        "null_q95_order_statistic_one_based": order_one_based,
        "exceedance_comparison": "null_lambda_1 >= observed_lambda_1",
        "exceedances_greater_than_or_equal": exceedances,
        "plus_one_p_value": plus_one_p,
        "alpha": float(alpha),
        "threshold_rule": "observed_lambda_1 > null_q95",
        "detected": detected,
    }


def publisher_block_pc1_null(
    value: Any,
    publishers: Sequence[str],
    *,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
        raise PC1PublisherNullInputError("draw count must be an integer of at least two")
    matrix = centered_unstandardized_matrix(value, "publisher-block PC1 matrix")
    if matrix.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise PC1PublisherNullInputError(
            f"publisher-block PC1 matrix must have shape (320, 8), got {matrix.shape}"
        )
    if not isinstance(rng, np.random.Generator):
        raise PC1PublisherNullInputError("publisher-block null requires a local Generator")
    try:
        v2._balanced_group_indices(publishers, matrix.shape[0])
    except v2.DimensionalityV2InputError as exc:
        raise PC1PublisherNullInputError(str(exc)) from exc

    observed = float(sample_covariance_eigenvalues(matrix)[0])
    null_values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        permuted = publisher_block_permutation(matrix, publishers, rng)
        null_values[draw] = sample_covariance_eigenvalues(permuted)[0]
    return {
        "draws": draws,
        "statistic": "largest eigenvalue lambda_1 of column-centered unstandardized sample covariance",
        "sample_covariance_ddof": 1,
        "null_scheme": (
            "independently reassign complete 20-value source-publisher blocks per column "
            "among 16 target publishers, then independently shuffle values within every "
            "assigned target block"
        ),
        **summarize_null(observed=observed, null_values=null_values),
    }


def validate_conclusion_vocabulary(conclusion: str) -> str:
    if conclusion not in ALLOWED_CONCLUSIONS:
        raise PC1PublisherNullInputError("conclusion vocabulary is outside the frozen contract")
    return conclusion


def gate_decision(*, q0_detected: bool, q1_detected: bool) -> dict[str, Any]:
    passed = bool(q0_detected and q1_detected)
    conclusion = validate_conclusion_vocabulary(PASS_CONCLUSION if passed else FAIL_CONCLUSION)
    return {
        "requires_detection_in_both_banks": True,
        "q0_detected": bool(q0_detected),
        "q1_detected": bool(q1_detected),
        "passed": passed,
        "conclusion": conclusion,
    }


def _numpy_global_states_equal(
    before: tuple[Any, ...], after: tuple[Any, ...]
) -> bool:
    if len(before) != len(after):
        return False
    for left, right in zip(before, after, strict=True):
        if isinstance(left, np.ndarray):
            if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
                return False
        elif left != right:
            return False
    return True


def _article_id_order_sha256(article_ids: Sequence[str]) -> str:
    encoded = json.dumps(list(article_ids), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_analysis(
    *,
    checkpoint_path: Path,
    corpus_path: Path,
    q0_path: Path,
    q1_path: Path,
    paired_structure_protocol_path: Path,
    paired_structure_result_path: Path,
    v2_protocol_path: Path,
    v2_result_path: Path,
    protocol_path: Path,
    draws: int = DRAW_COUNT,
    q0_seed: int = Q0_SEED,
    q1_seed: int = Q1_SEED,
) -> dict[str, Any]:
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
        raise PC1PublisherNullInputError("draw count must be an integer of at least two")
    if isinstance(q0_seed, bool) or isinstance(q1_seed, bool):
        raise PC1PublisherNullInputError("seeds must be distinct integers")
    if not isinstance(q0_seed, int) or not isinstance(q1_seed, int) or q0_seed == q1_seed:
        raise PC1PublisherNullInputError("seeds must be distinct integers")

    input_paths = {
        "checkpoint": checkpoint_path.resolve(),
        "corpus": corpus_path.resolve(),
        "q0": q0_path.resolve(),
        "q1": q1_path.resolve(),
        "paired_scalar_structure_protocol": paired_structure_protocol_path.resolve(),
        "paired_scalar_structure_result": paired_structure_result_path.resolve(),
        "paired_scalar_dimensionality_v2_protocol": v2_protocol_path.resolve(),
        "paired_scalar_dimensionality_v2_result": v2_result_path.resolve(),
    }
    source_paths = {
        "pc1_publisher_null_protocol": protocol_path.resolve(),
        "pc1_publisher_null_runner": Path(__file__).resolve(),
        **repository_helper_paths(),
    }
    python_random_before = random.getstate()
    numpy_random_before = np.random.get_state()
    initial_input_hashes = _hash_records(input_paths)
    initial_source_hashes = _hash_records(source_paths)

    _validate_paired_structure_result(
        paired_structure_result_path,
        input_hashes=initial_input_hashes,
        source_hashes=initial_source_hashes,
    )
    _validate_v2_result(
        v2_result_path,
        input_hashes=initial_input_hashes,
        source_hashes=initial_source_hashes,
    )
    paired.load_prompt_pairs(q0_path, q1_path)
    article_ids, logits = paired.load_checkpoint(checkpoint_path)
    articles = paired.load_articles(corpus_path, expected_n=EXPECTED_ROWS)
    corpus_ids = [article["canonical_id"] for article in articles]
    if len(article_ids) != EXPECTED_ROWS:
        raise PC1PublisherNullInputError(
            f"expected exactly {EXPECTED_ROWS} checkpoint rows, got {len(article_ids)}"
        )
    if len(set(article_ids)) != EXPECTED_ROWS:
        raise PC1PublisherNullInputError("checkpoint article IDs are not unique")
    if article_ids != corpus_ids:
        raise PC1PublisherNullInputError("checkpoint IDs do not exactly match corpus order")

    publishers = [article["_publisher"] for article in articles]
    publisher_counts = Counter(publishers)
    if len(publisher_counts) != EXPECTED_PUBLISHERS or set(publisher_counts.values()) != {
        EXPECTED_ROWS_PER_PUBLISHER
    }:
        raise PC1PublisherNullInputError("publisher balance must be exactly 16 publishers by 20 rows")
    try:
        v2._balanced_group_indices(publishers, len(article_ids))
    except v2.DimensionalityV2InputError as exc:
        raise PC1PublisherNullInputError(str(exc)) from exc

    matrices: dict[str, np.ndarray] = {}
    centering_checks: dict[str, float] = {}
    for bank in ("q0", "q1"):
        _, _, paired_scores = paired.scores_from_logits(
            logits[bank]["A"], logits[bank]["B"]
        )
        matrix = centered_unstandardized_matrix(paired_scores, f"{bank} paired A-minus-B matrix")
        if matrix.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
            raise PC1PublisherNullInputError(
                f"{bank} paired A-minus-B matrix must have shape (320, 8), got {matrix.shape}"
            )
        matrices[bank] = matrix
        centering_checks[bank] = float(np.max(np.abs(matrix.mean(axis=0))))

    results = {
        "q0": {
            "seed": q0_seed,
            **publisher_block_pc1_null(
                matrices["q0"],
                publishers,
                draws=draws,
                rng=np.random.default_rng(q0_seed),
            ),
        },
        "q1": {
            "seed": q1_seed,
            **publisher_block_pc1_null(
                matrices["q1"],
                publishers,
                draws=draws,
                rng=np.random.default_rng(q1_seed),
            ),
        },
    }
    overall_decision = gate_decision(
        q0_detected=bool(results["q0"]["detected"]),
        q1_detected=bool(results["q1"]["detected"]),
    )

    final_input_hashes = _hash_records(input_paths)
    final_source_hashes = _hash_records(source_paths)
    if initial_input_hashes != final_input_hashes or initial_source_hashes != final_source_hashes:
        raise PC1PublisherNullInputError("bound inputs or sources changed during analysis")
    python_random_unchanged = random.getstate() == python_random_before
    numpy_random_unchanged = _numpy_global_states_equal(
        numpy_random_before, np.random.get_state()
    )
    if not python_random_unchanged or not numpy_random_unchanged:
        raise PC1PublisherNullInputError("Python random or NumPy global RNG changed during analysis")

    return {
        "schema": {
            "name": "belief_transformer.pc1_publisher_block_null_gate1",
            "version": "1.0.0",
        },
        "status": "post_hoc_exploratory_gate_1",
        "scope": {
            "measurement": "paired scalar c=s_A-s_B only",
            "gate": "Gate 1 only",
            "gate_2": "out of scope and not executed; requires independent review of Gate 1",
            "model_inference_performed": False,
            "qualitative_article_coding_performed": False,
            "manuscript_modified": False,
        },
        "analysis": {
            "draws_per_bank": draws,
            "seeds": {"q0": q0_seed, "q1": q1_seed},
            "rng": "distinct isolated numpy.random.Generator instances",
            "python_random_global_state_unchanged": python_random_unchanged,
            "numpy_global_random_state_unchanged": numpy_random_unchanged,
            "matrix_preprocessing": "column-centered and unstandardized",
            "sample_covariance_ddof": 1,
            "exceedance_direction_and_ties": "null_lambda_1 >= observed_lambda_1",
            "plus_one_formula": "(exceedances + 1) / (draws + 1)",
            "q95_method": "numpy.quantile(method='higher')",
            "bank_detection_rule": "observed_lambda_1 > null_q95 and plus_one_p_value <= 0.05",
            "overall_gate_rule": "both q0 and q1 detected",
        },
        "validation": {
            "n_rows": len(article_ids),
            "n_columns": EXPECTED_COLUMNS,
            "unique_article_ids": len(set(article_ids)),
            "article_ids_exactly_match_corpus_order": True,
            "article_id_order_sha256": _article_id_order_sha256(article_ids),
            "publisher_count": len(publisher_counts),
            "rows_per_publisher": EXPECTED_ROWS_PER_PUBLISHER,
            "publisher_counts": dict(sorted(publisher_counts.items())),
            "publisher_balance_valid": True,
            "maximum_absolute_centered_column_mean": centering_checks,
        },
        "provenance": {
            "input_paths": {name: str(path) for name, path in input_paths.items()},
            "input_sha256": initial_input_hashes,
            "source_paths": {name: str(path) for name, path in source_paths.items()},
            "source_sha256": initial_source_hashes,
            "hashes_rechecked_after_computation": True,
            "runtime": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_executable": sys.executable,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
        },
        "results": results,
        "overall_decision": overall_decision,
    }


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_real_gate_once() -> Path:
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite: {DEFAULT_OUTPUT}")
    payload = build_analysis(
        checkpoint_path=DEFAULT_CHECKPOINT,
        corpus_path=DEFAULT_CORPUS,
        q0_path=DEFAULT_Q0,
        q1_path=DEFAULT_Q1,
        paired_structure_protocol_path=DEFAULT_PAIRED_STRUCTURE_PROTOCOL,
        paired_structure_result_path=DEFAULT_PAIRED_STRUCTURE_RESULT,
        v2_protocol_path=DEFAULT_V2_PROTOCOL,
        v2_result_path=DEFAULT_V2_RESULT,
        protocol_path=DEFAULT_PROTOCOL,
        draws=DRAW_COUNT,
        q0_seed=Q0_SEED,
        q1_seed=Q1_SEED,
    )
    write_json_exclusive(DEFAULT_OUTPUT, payload)
    return DEFAULT_OUTPUT


def main() -> int:
    output = run_real_gate_once()
    print(f"Gate 1 result written to {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        PC1PublisherNullInputError,
        paired.PairedScalarInputError,
        v2.DimensionalityV2InputError,
        FileExistsError,
    ) as exc:
        print(f"PC1PublisherNullInputError: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
