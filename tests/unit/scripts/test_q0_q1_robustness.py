from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pytest
import torch
from scipy.spatial import procrustes

from scripts import q0_q1_robustness as robustness


def _logits(article_index: int, bank_shift: float = 0.0) -> torch.Tensor:
    axis_a = np.asarray([-1.1, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.0])
    axis_b = np.asarray([0.8, -0.6, 0.7, -0.5, 0.4, -0.2, 0.1, -0.7])
    scale = -2.0 + 0.31 * article_index
    wave = np.sin(article_index + np.arange(8) * 0.73)
    z = scale * axis_a + (0.7 + bank_shift) * axis_b + 0.12 * wave
    return torch.tensor(np.column_stack([-z, 0.07 * wave, z]), dtype=torch.float64)


def _write_artifacts(tmp_path: Path, n: int = 18) -> tuple[Path, Path]:
    paths = []
    for name, shift in (("q0.pt", 0.0), ("q1.pt", 0.08)):
        path = tmp_path / name
        torch.save(
            {"nli_pairs": [{"logits_raw": _logits(i, shift)} for i in range(n)], "n_articles": n},
            path,
        )
        paths.append(path)
    return paths[0], paths[1]


def _committed_cli_repo(tmp_path: Path) -> Path:
    repo = tmp_path / f"committed-harness-{uuid.uuid4().hex}"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    commit_env = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
    }
    shutil.copy2(Path(robustness.__file__).resolve(), scripts / "q0_q1_robustness.py")
    shutil.copy2(Path(robustness.helper.__file__).resolve(), scripts / "exploratory_internal_structure_analysis.py")
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=repo, check=True)
    subprocess.run(["git", "add", "scripts"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "test harness"], cwd=repo, check=True, env=commit_env)
    return repo


def _run_cli(
    repo: Path,
    q0: Path,
    q1: Path,
    output: Path,
    *,
    permutations: int = 35,
    seed: int = 42,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "q0_q1_robustness.py"),
            "--q0-artifact",
            str(q0),
            "--q1-artifact",
            str(q1),
            "--output-json",
            str(output),
            "--permutations",
            str(permutations),
            "--seed",
            str(seed),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def _run(
    tmp_path: Path,
    q0: Path,
    q1: Path,
    *,
    permutations: int = 35,
    seed: int = 42,
) -> dict[str, object]:
    repo = _committed_cli_repo(tmp_path)
    output = tmp_path / "result.json"
    completed = _run_cli(repo, q0, q1, output, permutations=permutations, seed=seed)
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def test_score_extraction_uses_fixed_class_order_and_polarity(tmp_path: Path) -> None:
    logits = np.zeros((1, 8, 3), dtype=np.float64)
    logits[0, 0] = [1000.0, 1001.0, 1002.0]
    path = tmp_path / "score-extraction.pt"
    torch.save({"nli_pairs": [{"logits_raw": torch.tensor(logits[0])}]}, path)
    expected = robustness.helper.observer_polarity(logits)[0, 0]
    assert robustness.load_scores(path)[0, 0] == pytest.approx(expected)
    assert robustness.CLASS_ORDER == ("contradiction", "neutral", "entailment")


def test_snapshot_loader_uses_supplied_bytes_not_a_second_path_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    q0, _ = _write_artifacts(tmp_path, n=3)
    snapshot = q0.read_bytes()
    expected = robustness.load_scores(snapshot)
    calls: list[object] = []
    original = robustness.helper.load_nli_logits

    def recording_loader(source: object) -> np.ndarray:
        calls.append(source)
        assert isinstance(source, io.BytesIO)
        assert source.getvalue() == snapshot
        return original(source)  # type: ignore[arg-type]

    monkeypatch.setattr(robustness.helper, "load_nli_logits", recording_loader)
    replacement = (tmp_path / "q1.pt").read_bytes()
    q0.write_bytes(replacement)
    try:
        actual = robustness.load_scores(snapshot)
    finally:
        q0.write_bytes(snapshot)
    np.testing.assert_array_equal(actual, expected)
    assert len(calls) == 1


def test_same_seed_is_deterministic_and_p_display_has_a_floor(tmp_path: Path) -> None:
    q0, q1 = _write_artifacts(tmp_path)
    first = _run(tmp_path, q0, q1, permutations=45, seed=9)
    second = _run(tmp_path, q0, q1, permutations=45, seed=9)
    first["analysis"] = {**first["analysis"], "runtime_seconds": 0.0}
    second["analysis"] = {**second["analysis"], "runtime_seconds": 0.0}
    assert first == second
    assert robustness._p_display(1.0 / 1001.0) == "<0.001"
    assert robustness._p_display(1.0 / 46.0) == "0.0217"


@pytest.mark.parametrize("seed", [False, None, 1.5, "42"])
def test_invalid_seeds_raise_analysis_input_error(tmp_path: Path, seed: object) -> None:
    q0, q1 = _write_artifacts(tmp_path)
    with pytest.raises(robustness.AnalysisInputError, match="deterministic integer"):
        robustness.build_analysis(q0, q1, permutations=1, seed=seed)  # type: ignore[arg-type]


def test_variance_and_correspondence_nulls_have_distinct_behaviors(tmp_path: Path) -> None:
    q0, q1 = _write_artifacts(tmp_path)
    payload = _run(tmp_path, q0, q1, permutations=25)
    variance_text = payload["formulas_and_null_interpretations"]["variance_null"]
    correspondence_text = payload["formulas_and_null_interpretations"]["procrustes_null"]
    snapshot_text = payload["formulas_and_null_interpretations"][
        "artifact_snapshot_and_hash_limitation"
    ]
    assert "cross-axis dependence/non-independence" in variance_text
    assert "permute only Q1 row correspondence" in correspondence_text
    assert "snapshotted once" in snapshot_text
    assert "exact bytes are analyzed" in snapshot_text
    assert "final path hashes must match" in snapshot_text
    assert "A→B→A filesystem event" in snapshot_text
    assert "count_null_ge_observed" in payload["results"]["variance_null"]["q0"]
    assert "count_null_le_observed" in payload["results"]["procrustes"]


def test_procrustes_null_matches_independent_scipy_oracle() -> None:
    scores0 = np.asarray(
        [[-1.0, 0.5], [0.2, -1.4], [1.3, 0.1], [2.1, 1.7], [-0.4, 2.2]],
        dtype=np.float64,
    )
    scores1 = np.asarray(
        [[0.6, -0.8], [-1.7, 1.1], [1.8, 0.4], [0.3, 2.5], [2.2, -1.3]],
        dtype=np.float64,
    )
    permutations = 7
    seed = 31415
    oracle_rng = np.random.default_rng(seed)
    expected = []
    for _ in range(permutations):
        _, _, disparity = procrustes(scores0, scores1[oracle_rng.permutation(scores1.shape[0])])
        expected.append(disparity)

    actual = robustness._procrustes_null(
        scores0, scores1, permutations, np.random.default_rng(seed)
    )
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0.0, atol=1e-15)


def test_variance_null_matches_independent_column_permutation_oracle() -> None:
    matrix = np.asarray(
        [
            [-2.0, 0.5, 3.1, -1.2, 2.3, 0.7, -0.4, 1.8],
            [-1.0, -0.2, 1.7, 0.3, -1.1, 2.0, 0.9, -0.6],
            [0.2, 1.4, -0.8, 2.1, 0.5, -1.3, 1.6, 0.1],
            [1.1, -1.5, 0.4, -2.2, 1.8, 0.2, -1.7, 2.4],
            [2.0, 0.9, -2.5, 1.5, -0.7, -2.1, 0.3, -1.9],
            [3.2, -0.8, 2.2, -0.4, -2.0, 1.1, -0.9, 0.8],
        ],
        dtype=np.float64,
    )
    permutations = 6
    seed = 2718
    oracle_rng = np.random.default_rng(seed)
    expected = []
    for _ in range(permutations):
        permuted = np.column_stack(
            [matrix[oracle_rng.permutation(matrix.shape[0]), column] for column in range(matrix.shape[1])]
        )
        centered = permuted - permuted.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
        expected.append(float(np.sum(singular_values[:2] ** 2) / np.sum(singular_values**2)))

    actual = robustness._variance_null(
        matrix, 0.0, permutations, np.random.default_rng(seed)
    )
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0.0, atol=1e-15)


def test_procrustes_lower_tail_formula_and_summary_are_reported(tmp_path: Path) -> None:
    q0, q1 = _write_artifacts(tmp_path)
    payload = _run(tmp_path, q0, q1, permutations=19)
    result = payload["results"]["procrustes"]
    expected = (1 + result["count_null_le_observed"]) / 20
    assert result["p_value"] == pytest.approx(expected)
    assert result["null_quantiles"]["2.5%"] <= result["null_quantiles"]["97.5%"]
    assert result["lower_tail_rank"] == result["count_null_le_observed"] + 1
    assert result["lower_tail_rank_out_of"] == 20
    assert 0.0 < result["lower_tail_percentile"] <= 100.0


@pytest.mark.parametrize("case", ["missing", "malformed", "unequal"])
def test_malformed_missing_and_unequal_artifacts_fail_closed(tmp_path: Path, case: str) -> None:
    q0, q1 = _write_artifacts(tmp_path)
    if case == "missing":
        q1 = tmp_path / "missing.pt"
    elif case == "malformed":
        torch.save({"nli_pairs": [{"logits_raw": torch.zeros((8, 2))}]}, q1)
    else:
        torch.save({"nli_pairs": [{"logits_raw": _logits(0)}], "n_articles": 1}, q1)
    repo = _committed_cli_repo(tmp_path)
    output = tmp_path / "result.json"
    completed = _run_cli(repo, q0, q1, output, permutations=5)
    assert completed.returncode != 0
    assert "AnalysisInputError:" in completed.stderr
    assert not output.exists()


def test_provenance_is_recorded_and_global_rng_state_is_unchanged(tmp_path: Path) -> None:
    q0, q1 = _write_artifacts(tmp_path)
    np.random.seed(1234)
    expected = np.random.random(4)
    np.random.seed(1234)
    payload = _run(tmp_path, q0, q1, permutations=10, seed=42)
    actual = np.random.random(4)
    assert actual == pytest.approx(expected)
    assert len(payload["provenance"]["git_commit"]) == 40
    assert payload["provenance"]["harness"]["exact_head_match"] is True
    assert payload["provenance"]["helper"]["exact_head_match"] is True
    for name, path in (("q0_artifact", q0), ("q1_artifact", q1)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert payload["provenance"]["inputs"][name]["sha256_before"] == digest
        assert payload["provenance"]["inputs"][name]["sha256_after"] == digest
    assert payload["provenance"]["alignment"]["mode"] == "row_order_only"
    assert payload["analysis"]["confirmatory_decision"] is None


@pytest.mark.parametrize("collision", ["q0", "q1", "harness", "helper"])
def test_cli_rejects_output_collision_with_inputs_or_sources(
    tmp_path: Path, collision: str
) -> None:
    q0, q1 = _write_artifacts(tmp_path)
    repo = _committed_cli_repo(tmp_path)
    output = {
        "q0": q0,
        "q1": q1,
        "harness": repo / "scripts" / "q0_q1_robustness.py",
        "helper": repo / "scripts" / "exploratory_internal_structure_analysis.py",
    }[collision]
    before = output.read_bytes()
    completed = _run_cli(repo, q0, q1, output, permutations=5)
    assert completed.returncode != 0
    assert "output JSON must not overwrite an input or source artifact" in completed.stderr
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "relative_path", ["scripts/q0_q1_robustness.py", "scripts/exploratory_internal_structure_analysis.py"]
)
def test_exact_head_verifier_rejects_modified_committed_source(
    tmp_path: Path, relative_path: str
) -> None:
    repo = _committed_cli_repo(tmp_path)
    source = repo / relative_path
    source.write_bytes(source.read_bytes() + b"\n# temporary test mutation\n")

    with pytest.raises(robustness.AnalysisInputError, match="running script bytes differ"):
        robustness._verify_exact_head_source(source)


def test_atomic_cli_output_is_json_only(tmp_path: Path) -> None:
    q0, q1 = _write_artifacts(tmp_path)
    repo = _committed_cli_repo(tmp_path)
    output = tmp_path / "result.json"
    completed = _run_cli(repo, q0, q1, output, permutations=5)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["schema"]["name"] == robustness.SCHEMA_NAME
    assert not list(output.parent.glob(f".{output.name}.*"))


def test_direct_cli_missing_q1_fails_closed_without_import_traceback(tmp_path: Path) -> None:
    q0, _ = _write_artifacts(tmp_path)
    q1 = tmp_path / "missing-q1.pt"
    output = tmp_path / "result.json"
    repo = _committed_cli_repo(tmp_path)
    completed = _run_cli(repo, q0, q1, output, permutations=5)
    assert completed.returncode != 0
    assert "AnalysisInputError: Q1 artifact is not a file:" in completed.stderr
    assert "ImportError" not in completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr
    assert not output.exists()


def test_local_rng_does_not_mutate_global_rng_state(tmp_path: Path) -> None:
    q0, _ = _write_artifacts(tmp_path)
    scores = robustness.load_scores(q0)
    np.random.seed(1234)
    expected = np.random.random(4)
    np.random.seed(1234)
    robustness._variance_null(scores, 0.5, 5, np.random.default_rng(42))
    assert np.random.random(4) == pytest.approx(expected)
