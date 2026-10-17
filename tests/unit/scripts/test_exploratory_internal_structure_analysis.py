from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import exploratory_internal_structure_analysis as analysis


PUBLISHERS = ["alpha.test", "bravo.test", "charlie.test", "delta.test", "echo.test", "foxtrot.test"]


def _logits(article_index: int) -> torch.Tensor:
    pattern_1 = np.asarray([-1.1, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.0])
    pattern_2 = np.asarray([0.8, -0.6, 0.7, -0.5, 0.4, -0.2, 0.1, -0.7])
    publisher_index = article_index // 2
    a = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])[publisher_index]
    b = np.asarray([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])[publisher_index]
    within_publisher = -0.08 if article_index % 2 == 0 else 0.08
    z = 0.55 * a * pattern_1 + 0.45 * b * pattern_2 + within_publisher * pattern_1
    neutral = 0.1 * np.cos(np.arange(analysis.N_OBSERVERS) + article_index)
    return torch.tensor(np.column_stack([-z, neutral, z]), dtype=torch.float64)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact_path = tmp_path / "nli.pt"
    corpus_path = tmp_path / "corpus.jsonl"
    mbfc_path = tmp_path / "mbfc.json"
    torch.save(
        {
            "nli_pairs": [{"logits_raw": _logits(index)} for index in range(12)],
            "n_articles": 12,
        },
        artifact_path,
    )
    articles = []
    for index in range(12):
        articles.append(
            {
                "canonical_id": f"article-{index:02d}",
                "publisher": PUBLISHERS[index // 2],
                "published_at": f"2025-01-{index + 1:02d}",
            }
        )
    corpus_path.write_text(
        "".join(json.dumps(article, sort_keys=True) + "\n" for article in articles),
        encoding="utf-8",
    )
    mbfc_path.write_text(
        json.dumps(
            {
                publisher: score
                for publisher, score in zip(PUBLISHERS, [-4.0, -2.0, 1.0, 0.5, 4.5, 5.0])
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return artifact_path, corpus_path, mbfc_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _test_provenance(script_path: Path, repo_root: Path) -> dict[str, object]:
    del repo_root
    script_hash = _sha256(script_path)
    return {
        "git_commit": "0" * 40,
        "script": {
            "repo_relative_path": "scripts/exploratory_internal_structure_analysis.py",
            "sha256": script_hash,
            "git_head_sha256": script_hash,
            "exact_head_match": True,
        },
    }


def _build_analysis(
    artifact_path: Path, corpus_path: Path, mbfc_path: Path
) -> dict[str, object]:
    return analysis.build_analysis(
        artifact_path,
        corpus_path,
        mbfc_path,
        provenance_verifier=_test_provenance,
    )


def test_stable_softmax_and_fixed_polarity_formula() -> None:
    logits = np.zeros((1, analysis.N_OBSERVERS, 3), dtype=np.float64)
    logits[0, 0] = [1000.0, 1001.0, 1002.0]
    probabilities = analysis.stable_softmax(logits)
    expected = np.exp([-2.0, -1.0, 0.0]) / np.exp([-2.0, -1.0, 0.0]).sum()

    assert probabilities[0, 0] == pytest.approx(expected)
    assert analysis.observer_polarity(logits)[0, 0] == pytest.approx(expected[2] - expected[0])
    assert analysis.CLASS_ORDER == ("contradiction", "neutral", "entailment")


def test_pca_variance_loadings_and_scores_match_covariance_eigendecomposition() -> None:
    matrix = np.asarray(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0],
            [-1.0, -0.2, 0.7, 0.4, 1.1, 0.1, -0.3, -0.8],
            [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, 1.5, -0.5],
            [1.0, 0.3, -0.8, -0.4, -1.1, -0.1, 0.4, 0.7],
            [2.0, 1.2, 0.2, -0.9, -2.1, -1.0, 0.1, 1.1],
        ]
    )
    result = analysis.deterministic_pca(matrix)
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]

    assert result["explained_variance"] == pytest.approx(eigenvalues[:2])
    assert result["explained_variance_ratio"] == pytest.approx(eigenvalues[:2] / eigenvalues.sum())
    assert result["loadings"] @ result["loadings"].T == pytest.approx(np.eye(2), abs=1e-12)
    assert result["scores"] == pytest.approx(
        (matrix - matrix.mean(axis=0)) @ result["loadings"].T
    )
    for loading in result["loadings"]:
        assert loading[np.argmax(np.abs(loading))] > 0.0


def test_regression_adjusted_r2_and_overall_f_test_use_residual_degrees_of_freedom() -> None:
    x = np.asarray([[-2.0, 1.0], [-1.0, -1.0], [0.0, 2.0], [1.0, -2.0], [2.0, 0.5], [3.0, -0.5]])
    y = np.asarray([-3.2, -0.7, -0.1, 2.6, 3.4, 5.8])
    fit = analysis._fit_linear(x, y)
    test = analysis.overall_f_test(fit)
    expected_adjusted = 1.0 - (1.0 - fit["r2"]) * 5.0 / 3.0
    expected_f = ((fit["sst"] - fit["sse"]) / 2.0) / (fit["sse"] / 3.0)

    assert fit["adjusted_r2"] == pytest.approx(expected_adjusted)
    assert test["f_statistic"] == pytest.approx(expected_f)
    assert test["numerator_df"] == 2
    assert test["denominator_df"] == 3
    assert 0.0 <= test["p_value"] <= 1.0


def test_leave_one_publisher_out_refits_without_the_held_out_outcome() -> None:
    x = np.asarray([[-2.0, 1.0], [-1.0, -1.0], [0.0, 2.0], [1.0, -2.0], [2.0, 0.5], [3.0, -0.5]])
    y = np.asarray([-3.2, -0.7, -0.1, 2.6, 3.4, 5.8])
    result = analysis.leave_one_publisher_out(x, y, PUBLISHERS)

    manual_predictions = []
    for held_out in range(len(y)):
        keep = np.arange(len(y)) != held_out
        design = np.column_stack([np.ones(keep.sum()), x[keep]])
        beta = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        manual_predictions.append(float(np.r_[1.0, x[held_out]] @ beta))
    reported_predictions = [row["predicted"] for row in result["predictions"]]
    expected_r2 = 1.0 - np.sum((y - manual_predictions) ** 2) / np.sum((y - y.mean()) ** 2)

    assert reported_predictions == pytest.approx(manual_predictions)
    assert result["predictive_r2"] == pytest.approx(expected_r2)
    assert [row["publisher"] for row in result["predictions"]] == PUBLISHERS


@pytest.mark.parametrize(
    "failure",
    [
        "observer_count",
        "nonfinite",
        "count",
        "duplicate_id",
        "artifact_row_id",
        "mbfc_nonfinite",
    ],
)
def test_fail_closed_input_validation(tmp_path: Path, failure: str) -> None:
    artifact_path, corpus_path, mbfc_path = _write_fixture(tmp_path)
    if failure in {"observer_count", "nonfinite", "artifact_row_id"}:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if failure == "observer_count":
            payload["nli_pairs"][0]["logits_raw"] = torch.zeros((7, 3))
        elif failure == "nonfinite":
            payload["nli_pairs"][0]["logits_raw"][0, 0] = float("nan")
        else:
            payload["nli_pairs"][0]["canonical_id"] = "article-00"
        torch.save(payload, artifact_path)
    elif failure == "count":
        lines = corpus_path.read_text(encoding="utf-8").splitlines()
        corpus_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    elif failure == "duplicate_id":
        rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
        rows[1]["canonical_id"] = rows[0]["canonical_id"]
        corpus_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    else:
        mbfc = json.loads(mbfc_path.read_text(encoding="utf-8"))
        mbfc[PUBLISHERS[0]] = float("inf")
        mbfc_path.write_text(json.dumps(mbfc), encoding="utf-8")

    with pytest.raises(analysis.AnalysisInputError):
        _build_analysis(artifact_path, corpus_path, mbfc_path)


def test_code_provenance_verifier_accepts_exact_head_bytes_and_rejects_changes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    script = repo / "scripts" / "analysis.py"
    script.parent.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    script.write_bytes(b"print('frozen')\n")
    subprocess.run(["git", "add", "scripts/analysis.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "freeze analysis"], cwd=repo, check=True)

    provenance = analysis.verify_code_provenance(script, repo)
    assert provenance["script"]["sha256"] == _sha256(script)
    assert provenance["script"]["git_head_sha256"] == _sha256(script)
    assert provenance["script"]["exact_head_match"] is True
    assert provenance["script"]["repo_relative_path"] == "scripts/analysis.py"

    script.write_bytes(b"print('changed')\n")
    with pytest.raises(analysis.AnalysisInputError, match="differ from.*git HEAD"):
        analysis.verify_code_provenance(script, repo)


def test_input_change_during_analysis_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path, corpus_path, mbfc_path = _write_fixture(tmp_path)
    original_loader = analysis.load_mbfc

    def mutate_after_loading(path: Path) -> dict[str, float]:
        result = original_loader(path)
        corpus_path.write_bytes(corpus_path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(analysis, "load_mbfc", mutate_after_loading)
    with pytest.raises(analysis.AnalysisInputError, match="corpus_jsonl"):
        _build_analysis(artifact_path, corpus_path, mbfc_path)


def test_default_cli_rejects_unverified_running_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path, corpus_path, mbfc_path = _write_fixture(tmp_path)
    output_path = tmp_path / "must-not-exist.json"

    def reject_unverified(script_path: Path, repo_root: Path) -> dict[str, object]:
        del script_path, repo_root
        raise analysis.AnalysisInputError("uncommitted analysis code")

    monkeypatch.setattr(analysis, "verify_code_provenance", reject_unverified)
    with pytest.raises(analysis.AnalysisInputError, match="uncommitted analysis code"):
        analysis.main(
            [
                "--nli-artifact",
                str(artifact_path),
                "--corpus-jsonl",
                str(corpus_path),
                "--mbfc-json",
                str(mbfc_path),
                "--output-json",
                str(output_path),
            ]
        )
    assert not output_path.exists()


def test_output_is_deterministic_hashed_machine_readable_and_inference_limited(
    tmp_path: Path,
) -> None:
    artifact_path, corpus_path, mbfc_path = _write_fixture(tmp_path)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first_payload = _build_analysis(artifact_path, corpus_path, mbfc_path)
    second_payload = _build_analysis(artifact_path, corpus_path, mbfc_path)
    analysis.write_json_atomic(first_payload, first_path)
    analysis.write_json_atomic(second_payload, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert payload["schema"] == {"name": analysis.SCHEMA_NAME, "version": analysis.SCHEMA_VERSION}
    assert len(payload["provenance"]["git_commit"]) == 40
    assert payload["provenance"]["script"]["sha256"] == _sha256(Path(analysis.__file__))
    assert payload["provenance"]["script"]["exact_head_match"] is True
    assert payload["provenance"]["runtime_environment"] == {
        "numpy": str(np.__version__),
        "python": analysis.platform.python_version(),
        "python_implementation": analysis.platform.python_implementation(),
        "scipy": str(analysis.scipy.__version__),
        "torch": str(torch.__version__),
    }
    assert payload["provenance"]["inputs"]["nli_artifact"]["sha256"] == _sha256(artifact_path)
    assert payload["provenance"]["inputs"]["corpus_jsonl"]["sha256"] == _sha256(corpus_path)
    assert payload["provenance"]["inputs"]["mbfc_json"]["sha256"] == _sha256(mbfc_path)
    assert payload["provenance"]["alignment"]["mode"] == "row_order_only"
    assert payload["provenance"]["alignment"]["artifact_row_ids_present"] is False
    assert payload["formulas_and_limitations"]["temporal_inference"]["status"] == "not_estimated"
    temporal_reason = payload["formulas_and_limitations"]["temporal_inference"]["reason"]
    assert "nested within publishers" in temporal_reason
    assert payload["results"]["n_articles"] == 12
    assert payload["results"]["n_publishers"] == 6
    assert len(payload["results"]["publisher_centroids"]) == 6
    assert payload["results"]["mbfc"]["two_dimensional_fit"]["overall_f_test"]["numerator_df"] == 2
    assert len(payload["results"]["mbfc"]["leave_one_publisher_out"]["predictions"]) == 6
    assert len(payload["results"]["mbfc"]["pc2_leave_one_out_sensitivity"]["records"]) == 6

    serialized = json.dumps(payload).lower()
    assert "silhouette" not in serialized
    assert "permutation" not in serialized
    assert "pass" not in serialized
    assert "fail" not in serialized
