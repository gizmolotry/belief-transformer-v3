from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import paired_scalar_dimensionality as dim


def _logits(scale: np.ndarray) -> np.ndarray:
    scale = np.asarray(scale, dtype=np.float64)
    return np.stack((-scale, np.zeros_like(scale), scale), axis=-1)


def _write_analysis_inputs(
    tmp_path: Path, *, publishers: int = 4, articles_per_publisher: int = 8
) -> tuple[Path, Path, Path, Path, Path]:
    corpus = tmp_path / "corpus.jsonl"
    checkpoint = tmp_path / "checkpoint.jsonl"
    q0 = tmp_path / "q0.yaml"
    q1 = tmp_path / "q1.json"
    context = tmp_path / "context.json"
    corpus_rows = []
    checkpoint_rows = []
    for publisher_index in range(publishers):
        for article_index in range(articles_per_publisher):
            row_index = publisher_index * articles_per_publisher + article_index
            canonical_id = f"article-{row_index:03d}"
            publisher = f"publisher-{publisher_index}"
            corpus_rows.append(
                {
                    "canonical_id": canonical_id,
                    "publisher": publisher,
                    "content": f"Synthetic article {row_index}",
                }
            )
            first = (publisher_index - 1.5) + article_index / 20.0
            second = np.sin(article_index + 0.3 * publisher_index)
            third = np.cos(0.7 * article_index - publisher_index)
            weights = np.asarray(
                [1.2, 1.0, 0.8, 0.6, -0.7, -0.9, 0.4, -0.3], dtype=np.float64
            )
            q0_signal = first * weights + second * np.roll(weights, 2) + 0.1 * third
            q1_signal = 0.92 * q0_signal + 0.08 * third * np.roll(weights, 4)
            checkpoint_rows.append(
                {
                    "canonical_id": canonical_id,
                    "q0": {
                        "A": _logits(q0_signal).tolist(),
                        "B": _logits(-0.75 * q0_signal).tolist(),
                    },
                    "q1": {
                        "A": _logits(q1_signal).tolist(),
                        "B": _logits(-0.70 * q1_signal).tolist(),
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
            f'  pair_{index}:\n    A: "Q0 A {index}"\n    B: "Q0 B {index}"\n'
            for index in range(8)
        ),
        encoding="utf-8",
    )
    q1.write_text(
        json.dumps([value for index in range(8) for value in (f"Q1 A {index}", f"Q1 B {index}")]),
        encoding="utf-8",
    )
    input_hashes = {
        "checkpoint": dim.paired.sha256_file(checkpoint),
        "corpus": dim.paired.sha256_file(corpus),
        "q0": dim.paired.sha256_file(q0),
        "q1": dim.paired.sha256_file(q1),
    }
    context.write_text(
        json.dumps(
            {
                "claim_boundaries": {"primary_measurement": "paired scalar c=s_A-s_B"},
                "provenance": {"inputs": input_hashes},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return checkpoint, corpus, q0, q1, context


def test_grouped_schedule_has_no_publisher_leakage_and_complete_holdout() -> None:
    publishers = [f"publisher-{group}" for group in range(8) for _ in range(3)]
    first = dim.grouped_split_schedule(
        publishers, repeats=5, folds=4, rng=np.random.default_rng(71)
    )
    second = dim.grouped_split_schedule(
        publishers, repeats=5, folds=4, rng=np.random.default_rng(71)
    )

    assert [row["test_publishers"] for row in first] == [row["test_publishers"] for row in second]
    for repeat in range(5):
        rows = [row for row in first if row["repeat"] == repeat]
        held_out = [publisher for row in rows for publisher in row["test_publishers"]]
        assert sorted(held_out) == sorted(set(publishers))
        for row in rows:
            assert set(row["train_publishers"]).isdisjoint(row["test_publishers"])


def test_grouped_reconstruction_reports_nested_decreases_but_uses_one_se_rule() -> None:
    rng = np.random.default_rng(8)
    publishers = [f"publisher-{group}" for group in range(8) for _ in range(10)]
    latent = rng.normal(size=(80, 3))
    matrix = latent @ rng.normal(size=(3, 8)) + 0.05 * rng.normal(size=(80, 8))
    schedule = dim.grouped_split_schedule(
        publishers, repeats=6, folds=4, rng=np.random.default_rng(9)
    )
    result = dim.grouped_reconstruction(matrix, schedule, repeats=6, folds=4)

    assert result["selected_k"] in {1, 2, 3}
    assert len(result["per_fold"]) == 24
    assert "one standard error" in result["selection_rule"]
    for row in result["per_fold"]:
        assert row["paired_relative_error_decreases"]["k1_to_k2"] >= -1e-12
        assert row["paired_relative_error_decreases"]["k2_to_k3"] >= -1e-12
    assert (
        result["paired_improvement_summary"]["k1_to_k2"]["interpretation"]
        == "descriptive_only_nested_components_mechanically_reduce_error"
    )


def test_eigenvalue_null_is_deterministic_and_uses_plus_one_p_values() -> None:
    rng = np.random.default_rng(10)
    latent = rng.normal(size=(120, 2))
    matrix = latent @ rng.normal(size=(2, 8)) + 0.1 * rng.normal(size=(120, 8))
    first = dim.eigenvalue_permutation_null(
        matrix, permutations=99, rng=np.random.default_rng(11)
    )
    second = dim.eigenvalue_permutation_null(
        matrix, permutations=99, rng=np.random.default_rng(11)
    )

    assert first == second
    for component in ("pc2", "pc3"):
        record = first["components"][component]
        expected = (record["null_exceedances_greater_or_equal"] + 1) / 100
        assert record["plus_one_p_value"] == pytest.approx(expected)
        assert record["plus_one_p_value"] > 0.0


def test_cluster_bootstrap_reports_direct_subspace_swap_and_loading_diagnostics() -> None:
    rng = np.random.default_rng(12)
    publishers = [f"publisher-{group}" for group in range(10) for _ in range(12)]
    latent = rng.normal(size=(120, 3)) * np.asarray([3.0, 2.0, 0.8])
    matrix = latent @ np.linalg.qr(rng.normal(size=(8, 8)))[0][:3]
    result = dim.publisher_cluster_bootstrap(
        matrix, publishers, replicates=199, rng=np.random.default_rng(13)
    )

    assert result["valid_replicates"] + result["failed_replicates"] == 199
    assert 0.0 <= result["direct_abs_cosine"]["median"] <= 1.0
    assert 0.0 <= result["reference_pc2_in_replicate_pc2_pc3_subspace"]["median"] <= 1.0
    assert 0.0 <= result["replicate_pc2_in_reference_pc2_pc3_subspace"]["median"] <= 1.0
    assert len(result["sign_aligned_pc2_loading_percentile_intervals"]) == 8
    assert set(result["stability_conditions"]) == {
        "direct_cosine_fraction",
        "reference_pc2_swap_rate",
        "replicate_pc2_swap_rate",
        "failure_rate",
    }


@pytest.mark.parametrize(
    ("pc2", "pc3", "stable", "selected", "expected"),
    [
        (True, False, True, 2, "supports_two_over_one_not_three"),
        (True, True, True, 3, "supports_at_least_three"),
        (False, False, False, 1, "supports_no_more_than_one"),
        (True, False, False, 2, "inconclusive_pc2_unstable"),
        (True, False, True, 3, "inconclusive_mixed_evidence"),
        (False, True, True, 1, "inconclusive_mixed_evidence"),
    ],
)
def test_conjunctive_decision_rules(
    pc2: bool, pc3: bool, stable: bool, selected: int, expected: str
) -> None:
    assert (
        dim.dimensionality_decision(
            pc2_passes=pc2,
            pc3_passes=pc3,
            pc2_stable=stable,
            selected_k=selected,
        )
        == expected
    )


def test_build_analysis_is_deterministic_hash_bound_and_paired_only(tmp_path: Path) -> None:
    checkpoint, corpus, q0, q1, context = _write_analysis_inputs(tmp_path)
    kwargs = {
        "checkpoint_path": checkpoint,
        "corpus_path": corpus,
        "q0_path": q0,
        "q1_path": q1,
        "context_result_path": context,
        "expected_n": 32,
        "expected_publishers": 4,
        "expected_per_publisher": 8,
        "seed": 101,
        "cv_repeats": 3,
        "cv_folds": 2,
        "permutations": 19,
        "bootstrap_replicates": 29,
    }
    legacy_state_before = np.random.get_state()
    first = dim.build_analysis(**kwargs)
    second = dim.build_analysis(**kwargs)
    legacy_state_after = np.random.get_state()

    assert first == second
    assert first["claim_boundaries"]["measurement"] == "paired scalar c=s_A-s_B only"
    assert set(first["results"]) == {"q0", "q1"}
    assert first["provenance"]["article_ids_exactly_match_corpus_order"] is True
    assert first["provenance"]["input_sha256"]["checkpoint"] == dim.paired.sha256_file(
        checkpoint
    )
    for before, after in zip(legacy_state_before, legacy_state_after, strict=True):
        if isinstance(before, np.ndarray):
            np.testing.assert_array_equal(before, after)
        else:
            assert before == after


def test_context_hash_mismatch_and_output_overwrite_fail_closed(tmp_path: Path) -> None:
    checkpoint, corpus, q0, q1, context = _write_analysis_inputs(tmp_path)
    context_payload = json.loads(context.read_text(encoding="utf-8"))
    context_payload["provenance"]["inputs"]["checkpoint"] = "0" * 64
    context.write_text(json.dumps(context_payload), encoding="utf-8")
    with pytest.raises(dim.DimensionalityInputError, match="does not bind current checkpoint"):
        dim.build_analysis(
            checkpoint_path=checkpoint,
            corpus_path=corpus,
            q0_path=q0,
            q1_path=q1,
            context_result_path=context,
            expected_n=32,
            expected_publishers=4,
            expected_per_publisher=8,
            cv_repeats=2,
            cv_folds=2,
            permutations=2,
            bootstrap_replicates=2,
        )

    output = tmp_path / "result.json"
    dim._write_json_exclusive(output, {"first": True})
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        dim._write_json_exclusive(output, {"second": True})
    assert output.read_bytes() == before


def test_frozen_protocol_declares_post_inspection_status_and_no_sign_test() -> None:
    text = " ".join(dim.PROTOCOL.read_text(encoding="utf-8").lower().split())
    assert "**not** a preregistration" in text
    assert "already inspected pca spectra" in text
    assert "positive decrease or sign test is never evidence" in text
    assert "inconclusive_mixed_evidence" in text
    assert (
        "not independent replication" in text
        or "neither branch is independent replication" in text
    )
