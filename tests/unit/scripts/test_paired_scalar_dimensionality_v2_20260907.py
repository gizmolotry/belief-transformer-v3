from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import paired_scalar_dimensionality_v2_20260907 as v2


PUBLISHERS = [f"publisher-{group:02d}" for group in range(16) for _ in range(20)]


def _logits(scale: np.ndarray) -> np.ndarray:
    scale = np.asarray(scale, dtype=np.float64)
    return np.stack((-scale, np.zeros_like(scale), scale), axis=-1)


def _isotropic_matrix() -> np.ndarray:
    return np.random.default_rng(827_101).normal(size=(320, 8))


def _rank_two_matrix() -> np.ndarray:
    rng = np.random.default_rng(827_202)
    basis, _ = np.linalg.qr(rng.normal(size=(8, 2)))
    latent = rng.normal(size=(320, 2)) * np.asarray([4.0, 2.5])
    return latent @ basis.T + rng.normal(scale=0.45, size=(320, 8))


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    corpus = tmp_path / "corpus.jsonl"
    checkpoint = tmp_path / "checkpoint.jsonl"
    q0 = tmp_path / "q0.yaml"
    q1 = tmp_path / "q1.json"
    context = tmp_path / "context.json"
    v1_protocol = tmp_path / "v1_protocol.md"
    v1_result = tmp_path / "v1_result.json"
    signal = _rank_two_matrix()
    corpus_rows = []
    checkpoint_rows = []
    for index, publisher in enumerate(PUBLISHERS):
        canonical_id = f"article-{index:03d}"
        corpus_rows.append(
            {
                "canonical_id": canonical_id,
                "publisher": publisher,
                "content": f"Synthetic article {index}",
            }
        )
        checkpoint_rows.append(
            {
                "canonical_id": canonical_id,
                "q0": {
                    "A": _logits(signal[index]).tolist(),
                    "B": _logits(-0.7 * signal[index]).tolist(),
                },
                "q1": {
                    "A": _logits(0.9 * signal[index]).tolist(),
                    "B": _logits(-0.6 * signal[index]).tolist(),
                },
            }
        )
    corpus.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in corpus_rows), encoding="utf-8"
    )
    checkpoint.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in checkpoint_rows),
        encoding="utf-8",
    )
    q0.write_text(
        "queries:\n"
        + "".join(
            f'  pair_{axis}:\n    A: "Q0 A {axis}"\n    B: "Q0 B {axis}"\n'
            for axis in range(8)
        ),
        encoding="utf-8",
    )
    q1.write_text(
        json.dumps([text for axis in range(8) for text in (f"Q1 A {axis}", f"Q1 B {axis}")]),
        encoding="utf-8",
    )
    base_hashes = {
        "checkpoint": v2.paired.sha256_file(checkpoint),
        "corpus": v2.paired.sha256_file(corpus),
        "q0": v2.paired.sha256_file(q0),
        "q1": v2.paired.sha256_file(q1),
    }
    context.write_text(
        json.dumps(
            {
                "claim_boundaries": {"primary_measurement": "paired scalar c=s_A-s_B"},
                "provenance": {"inputs": base_hashes},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    v1_protocol.write_text("Synthetic v1 protocol fixture.\n", encoding="utf-8")
    v1_result.write_text(
        json.dumps(
            {
                "schema": {"name": "belief_transformer.paired_scalar_dimensionality_selection"},
                "provenance": {"input_sha256": base_hashes},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "checkpoint_path": checkpoint,
        "corpus_path": corpus,
        "q0_path": q0,
        "q1_path": q1,
        "context_result_path": context,
        "v1_protocol_path": v1_protocol,
        "v1_result_path": v1_result,
    }


def test_isotropic_noise_selects_one_not_three() -> None:
    result = v2.leave_one_publisher_out_ppca(_isotropic_matrix(), PUBLISHERS)

    assert result["selected_k"] == 1
    assert result["best_mean_nll_k"] in {1, 2, 3}
    assert len(result["folds"]) == 16
    assert all(fold["n_train_publishers"] == 15 for fold in result["folds"])
    assert all(fold["n_train_articles"] == 300 for fold in result["folds"])
    assert all(fold["n_test_articles"] == 20 for fold in result["folds"])


def test_strong_rank_two_signal_selects_two() -> None:
    result = v2.leave_one_publisher_out_ppca(_rank_two_matrix(), PUBLISHERS)
    summaries = result["candidate_summaries"]

    assert result["selected_k"] == 2
    assert summaries["2"]["mean_publisher_nll"] < summaries["1"]["mean_publisher_nll"] - 0.5
    assert summaries["2"]["mean_publisher_nll"] <= result["one_standard_error_threshold"]


def test_ppca_floor_produces_spd_covariance_and_finite_nll() -> None:
    rng = np.random.default_rng(33)
    one_factor = rng.normal(size=(320, 1)) @ np.ones((1, 8))
    fit = v2.fit_closed_form_ppca(one_factor, 3)
    nll = v2.ppca_negative_log_likelihood(one_factor[:20], fit)

    assert fit["raw_sigma2"] < fit["numeric_floor"]
    assert fit["floored_sigma2"] == fit["numeric_floor"]
    assert np.all(np.asarray(fit["covariance_eigenvalues"]) > 0.0)
    assert np.isfinite(nll).all()


def test_publisher_block_permutation_preserves_each_complete_block_multiset() -> None:
    row = np.tile(np.arange(20), 16)
    group = np.repeat(np.arange(16), 20)
    matrix = np.column_stack([group * 1000 + row + column * 100_000 for column in range(8)])
    permuted = v2.publisher_block_permutation(matrix, PUBLISHERS, np.random.default_rng(44))

    for column in range(8):
        inferred_sources = []
        reordered_blocks = 0
        for target in range(16):
            target_values = permuted[target * 20 : (target + 1) * 20, column]
            normalized = target_values - column * 100_000
            sources = np.unique(normalized // 1000)
            assert sources.size == 1
            source = int(sources[0])
            inferred_sources.append(source)
            expected_set = set((source * 1000 + np.arange(20)).tolist())
            assert set(normalized.tolist()) == expected_set
            if not np.array_equal(normalized, source * 1000 + np.arange(20)):
                reordered_blocks += 1
        assert sorted(inferred_sources) == list(range(16))
        assert reordered_blocks > 0


def test_primary_null_is_deterministic_plus_one_and_cluster_labeled() -> None:
    matrix = _rank_two_matrix()
    first = v2.publisher_block_eigenvalue_null(
        matrix, PUBLISHERS, permutations=39, rng=np.random.default_rng(55)
    )
    second = v2.publisher_block_eigenvalue_null(
        matrix, PUBLISHERS, permutations=39, rng=np.random.default_rng(55)
    )

    assert first == second
    assert first["role"] == "primary_cluster_aware_null"
    assert "complete 20-value publisher blocks" in first["scheme"]
    for name in ("pc2", "pc3"):
        record = first["components"][name]
        assert record["plus_one_p_value"] == pytest.approx(
            (record["null_exceedances_greater_or_equal"] + 1) / 40
        )
        assert record["plus_one_p_value"] > 0.0


def test_article_null_is_sensitivity_only_and_not_a_decision_input() -> None:
    result = v2.article_level_sensitivity_null(
        _rank_two_matrix(), permutations=19, rng=np.random.default_rng(66)
    )

    assert result["role"] == "article_level_sensitivity_only_not_used_for_primary_verdict"
    assert all("detected" in record and "passes" not in record for record in result["components"].values())


@pytest.mark.parametrize(
    ("pc2", "pc3", "stable", "selected", "expected"),
    [
        (False, False, False, 1, "pc2_not_detected_under_cluster_null"),
        (False, True, True, 3, "pc2_not_detected_under_cluster_null"),
        (True, False, False, 2, "inconclusive_mixed_evidence"),
        (True, False, True, 1, "inconclusive_mixed_evidence"),
        (True, False, True, 2, "evidence_for_at_least_two"),
        (True, True, True, 2, "evidence_for_at_least_two"),
        (True, True, True, 3, "evidence_for_at_least_three"),
    ],
)
def test_decision_vocabulary_never_converts_nondetection_to_absence(
    pc2: bool, pc3: bool, stable: bool, selected: int, expected: str
) -> None:
    assert (
        v2.dimensionality_verdict(
            pc2_detected=pc2,
            pc3_detected=pc3,
            pc2_stable=stable,
            selected_k=selected,
        )
        == expected
    )


def test_build_analysis_is_hash_bound_deterministic_and_preserves_global_rng(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    kwargs = {
        **paths,
        "seed": 77,
        "permutations": 9,
        "bootstrap_replicates": 19,
    }
    state_before = np.random.get_state()
    first = v2.build_analysis(**kwargs)
    second = v2.build_analysis(**kwargs)
    state_after = np.random.get_state()

    assert first == second
    assert first["schema"]["version"] == "2.0.0"
    assert first["claim_boundaries"]["non_detection_is_not_absence_or_equivalence"] is True
    assert set(first["results"]) == {"q0", "q1"}
    assert first["provenance"]["input_sha256"]["checkpoint"] == v2.paired.sha256_file(
        paths["checkpoint_path"]
    )
    for before, after in zip(state_before, state_after, strict=True):
        if isinstance(before, np.ndarray):
            np.testing.assert_array_equal(before, after)
        else:
            assert before == after


def test_hash_mismatch_and_output_overwrite_fail_closed(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    payload = json.loads(paths["context_result_path"].read_text(encoding="utf-8"))
    payload["provenance"]["inputs"]["checkpoint"] = "0" * 64
    paths["context_result_path"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(v2.DimensionalityV2InputError, match="does not bind current checkpoint"):
        v2.build_analysis(**paths, permutations=2, bootstrap_replicates=2)

    output = tmp_path / "result.json"
    v2._write_json_exclusive(output, {"first": True})
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        v2._write_json_exclusive(output, {"second": True})
    assert output.read_bytes() == before


def test_v2_protocol_freezes_review_correction_and_claim_boundaries() -> None:
    text = " ".join(v2.V2_PROTOCOL.read_text(encoding="utf-8").lower().split())

    assert "post-review corrected exploratory reanalysis" in text
    assert "not a preregistration, confirmation, or independent replication" in text
    assert "mechanically reduces squared reconstruction error" in text
    assert "pc2_not_detected_under_cluster_null" in text
    assert "non-detection" in text
