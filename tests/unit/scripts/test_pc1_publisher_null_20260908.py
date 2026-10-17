from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from scripts import pc1_publisher_null_20260908 as gate


PUBLISHERS = [f"publisher-{group:02d}" for group in range(16) for _ in range(20)]


def _logits(scale: np.ndarray) -> np.ndarray:
    values = np.asarray(scale, dtype=np.float64)
    return np.stack((-values, np.zeros_like(values), values), axis=-1)


def _write_synthetic_inputs(tmp_path: Path) -> dict[str, Path]:
    checkpoint = tmp_path / "checkpoint.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    q0 = tmp_path / "q0.yaml"
    q1 = tmp_path / "q1.json"
    structure_protocol = tmp_path / "structure_protocol.md"
    structure_result = tmp_path / "structure_result.json"
    v2_protocol = tmp_path / "v2_protocol.md"
    v2_result = tmp_path / "v2_result.json"
    protocol = tmp_path / "gate_protocol.md"

    rng = np.random.default_rng(91_001)
    latent = rng.normal(size=(320, 1))
    loadings = np.asarray([[1.0, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2]])
    signal = latent @ loadings + rng.normal(scale=0.25, size=(320, 8))
    corpus_rows = []
    checkpoint_rows = []
    for index, publisher in enumerate(PUBLISHERS):
        article_id = f"article-{index:03d}"
        corpus_rows.append(
            {"canonical_id": article_id, "publisher": publisher, "content": "synthetic"}
        )
        checkpoint_rows.append(
            {
                "canonical_id": article_id,
                "q0": {
                    "A": _logits(signal[index]).tolist(),
                    "B": _logits(-0.6 * signal[index]).tolist(),
                },
                "q1": {
                    "A": _logits(0.9 * signal[index]).tolist(),
                    "B": _logits(-0.5 * signal[index]).tolist(),
                },
            }
        )
    checkpoint.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in checkpoint_rows),
        encoding="utf-8",
    )
    corpus.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in corpus_rows),
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
        json.dumps(
            [text for axis in range(8) for text in (f"Q1 A {axis}", f"Q1 B {axis}")]
        ),
        encoding="utf-8",
    )
    structure_protocol.write_text("Synthetic structure protocol.\n", encoding="utf-8")
    v2_protocol.write_text("Synthetic corrected-v2 protocol.\n", encoding="utf-8")
    protocol.write_text("Synthetic Gate 1 protocol.\n", encoding="utf-8")

    core_hashes = {
        "checkpoint": gate.sha256_file(checkpoint),
        "corpus": gate.sha256_file(corpus),
        "q0": gate.sha256_file(q0),
        "q1": gate.sha256_file(q1),
    }
    sources = gate.repository_helper_paths()
    structure_result.write_text(
        json.dumps(
            {
                "schema": {"name": "belief_transformer.paired_scalar_structure"},
                "claim_boundaries": {"primary_measurement": "paired scalar c=s_A-s_B"},
                "provenance": {
                    "inputs": core_hashes,
                    "source_sha256": {
                        "protocol": gate.sha256_file(structure_protocol),
                        "runner": gate.sha256_file(sources["imported_paired_scalar_structure"]),
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    v2_result.write_text(
        json.dumps(
            {
                "schema": {
                    "name": "belief_transformer.paired_scalar_dimensionality_selection_v2"
                },
                "claim_boundaries": {"measurement": "paired scalar c=s_A-s_B only"},
                "provenance": {
                    "input_sha256": {
                        **core_hashes,
                        "contextual_paired_scalar_result": gate.sha256_file(structure_result),
                    },
                    "source_sha256": {
                        "v2_frozen_protocol": gate.sha256_file(v2_protocol),
                        "v2_runner": gate.sha256_file(sources["imported_dimensionality_v2"]),
                        "imported_v1_runner": gate.sha256_file(
                            sources["imported_dimensionality_v1"]
                        ),
                        "paired_scalar_scoring_runner": gate.sha256_file(
                            sources["imported_paired_scalar_structure"]
                        ),
                    },
                },
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
        "paired_structure_protocol_path": structure_protocol,
        "paired_structure_result_path": structure_result,
        "v2_protocol_path": v2_protocol,
        "v2_result_path": v2_result,
        "protocol_path": protocol,
    }


def _assert_numpy_state_equal(before: tuple[object, ...], after: tuple[object, ...]) -> None:
    assert len(before) == len(after)
    for left, right in zip(before, after, strict=True):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right


def test_block_null_preserves_blocks_and_destroys_common_correspondence() -> None:
    groups = np.repeat(np.arange(16), 20)
    rows = np.tile(np.arange(20), 16)
    matrix = np.column_stack(
        [column * 1_000_000 + groups * 1_000 + rows for column in range(8)]
    )

    permuted = gate.publisher_block_permutation(
        matrix, PUBLISHERS, np.random.default_rng(20_260_908)
    )
    source_assignments = np.empty((16, 8), dtype=np.int64)
    decoded = np.empty((320, 8, 2), dtype=np.int64)
    for column in range(8):
        np.testing.assert_array_equal(np.sort(permuted[:, column]), np.sort(matrix[:, column]))
        normalized_column = permuted[:, column] - column * 1_000_000
        decoded[:, column, 0] = normalized_column // 1_000
        decoded[:, column, 1] = normalized_column % 1_000
        for target in range(16):
            target_values = normalized_column[target * 20 : (target + 1) * 20]
            source_groups = np.unique(target_values // 1_000)
            assert source_groups.size == 1
            source = int(source_groups[0])
            source_assignments[target, column] = source
            assert set((target_values % 1_000).tolist()) == set(range(20))
        assert sorted(source_assignments[:, column].tolist()) == list(range(16))

    assert len({tuple(source_assignments[:, column]) for column in range(8)}) > 1
    assert any(
        not np.array_equal(decoded[:, 0, :], decoded[:, column, :]) for column in range(1, 8)
    )


def test_pc1_null_is_deterministic_with_an_isolated_local_generator() -> None:
    rng = np.random.default_rng(12_345)
    matrix = rng.normal(size=(320, 8))
    first = gate.publisher_block_pc1_null(
        matrix, PUBLISHERS, draws=17, rng=np.random.default_rng(888)
    )
    second = gate.publisher_block_pc1_null(
        matrix, PUBLISHERS, draws=17, rng=np.random.default_rng(888)
    )

    assert first == second
    assert first["draws"] == 17
    assert first["sample_covariance_ddof"] == 1


def test_p_value_ties_count_as_exceedances_and_direction_is_greater() -> None:
    summary = gate.summarize_null(observed=3.0, null_values=np.asarray([1.0, 2.0, 3.0, 4.0]))

    assert summary["exceedance_comparison"] == "null_lambda_1 >= observed_lambda_1"
    assert summary["exceedances_greater_than_or_equal"] == 2
    assert summary["plus_one_p_value"] == pytest.approx(3 / 5)
    assert summary["detected"] is False


def test_q95_uses_the_frozen_higher_order_statistic() -> None:
    values = np.arange(20, dtype=np.float64)
    summary = gate.summarize_null(observed=100.0, null_values=values)

    assert summary["null_q95_method"] == "numpy.quantile(method='higher')"
    assert summary["null_q95_order_statistic_one_based"] == 20
    assert summary["null_q95"] == 19.0


@pytest.mark.parametrize(
    ("q0", "q1", "passed"),
    [(False, False, False), (True, False, False), (False, True, False), (True, True, True)],
)
def test_gate_requires_detection_in_both_banks(q0: bool, q1: bool, passed: bool) -> None:
    decision = gate.gate_decision(q0_detected=q0, q1_detected=q1)

    assert decision["passed"] is passed
    assert decision["conclusion"] == (
        gate.PASS_CONCLUSION if passed else gate.FAIL_CONCLUSION
    )


def test_conclusion_vocabulary_is_closed_and_guarded() -> None:
    assert gate.validate_conclusion_vocabulary(gate.PASS_CONCLUSION) == gate.PASS_CONCLUSION
    assert gate.validate_conclusion_vocabulary(gate.FAIL_CONCLUSION) == gate.FAIL_CONCLUSION
    with pytest.raises(gate.PC1PublisherNullInputError, match="conclusion vocabulary"):
        gate.validate_conclusion_vocabulary("a real and meaningful axis interpretation")


def test_build_is_hash_bound_deterministic_and_preserves_global_rng(tmp_path: Path) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    random.seed(101)
    np.random.seed(202)
    python_before = random.getstate()
    numpy_before = np.random.get_state()

    first = gate.build_analysis(**paths, draws=3, q0_seed=701, q1_seed=702)
    second = gate.build_analysis(**paths, draws=3, q0_seed=701, q1_seed=702)

    assert first == second
    assert first["validation"]["article_ids_exactly_match_corpus_order"] is True
    assert first["validation"]["publisher_balance_valid"] is True
    assert first["analysis"]["python_random_global_state_unchanged"] is True
    assert first["analysis"]["numpy_global_random_state_unchanged"] is True
    assert set(first["results"]) == {"q0", "q1"}
    assert random.getstate() == python_before
    _assert_numpy_state_equal(numpy_before, np.random.get_state())


def test_hash_tamper_and_output_overwrite_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_synthetic_inputs(tmp_path)
    original = gate.publisher_block_pc1_null
    calls = 0

    def tampering_null(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            checkpoint = paths["checkpoint_path"]
            checkpoint.write_bytes(checkpoint.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(gate, "publisher_block_pc1_null", tampering_null)
    with pytest.raises(gate.PC1PublisherNullInputError, match="changed during analysis"):
        gate.build_analysis(**paths, draws=2, q0_seed=801, q1_seed=802)

    output = tmp_path / "result.json"
    gate.write_json_exclusive(output, {"first": True})
    original_bytes = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gate.write_json_exclusive(output, {"second": True})
    assert output.read_bytes() == original_bytes


def test_frozen_protocol_is_gate1_only_and_records_required_conventions() -> None:
    protocol = " ".join(gate.DEFAULT_PROTOCOL.read_text(encoding="utf-8").split())

    assert "post-hoc exploratory test" in protocol
    assert "sample covariance" in protocol and "ddof=1" in protocol
    assert 'method="higher"' in protocol
    assert "ties count against detection" in protocol
    assert "Gate 2 paraphrase stability is out of scope" in protocol
    assert "must not run unless" in protocol
