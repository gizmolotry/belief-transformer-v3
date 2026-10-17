from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import paired_scalar_pc1_gate_20260908 as gate


PUBLISHERS = [
    f"publisher-{publisher:02d}"
    for publisher in range(gate.EXPECTED_PUBLISHERS)
    for _ in range(gate.EXPECTED_PER_PUBLISHER)
]


def _paired_signal() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1001)
    latent = rng.normal(size=gate.EXPECTED_N)
    q0_loading = np.array([1.0, 0.8, 0.55, 0.35, -0.2, -0.4, -0.65, -0.9])
    q1_loading = q0_loading + np.array([0.03, -0.02, 0.01, 0.0, -0.01, 0.02, -0.03, 0.01])
    q0 = np.outer(latent, q0_loading) + 0.08 * rng.normal(size=(gate.EXPECTED_N, 8))
    q1 = np.outer(latent, q1_loading) + 0.08 * rng.normal(size=(gate.EXPECTED_N, 8))
    return q0, q1


def _assert_rng_states_equal(before: tuple[object, ...], after: tuple[object, ...]) -> None:
    for left, right in zip(before, after, strict=True):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right


def test_exact_v2_block_null_preserves_complete_blocks_and_is_deterministic() -> None:
    row = np.tile(np.arange(gate.EXPECTED_PER_PUBLISHER), gate.EXPECTED_PUBLISHERS)
    group = np.repeat(np.arange(gate.EXPECTED_PUBLISHERS), gate.EXPECTED_PER_PUBLISHER)
    matrix = np.column_stack(
        [group * 1_000 + row + column * 100_000 for column in range(8)]
    ).astype(np.float64)
    permuted = gate.v2.publisher_block_permutation(
        matrix, PUBLISHERS, np.random.default_rng(2002)
    )

    for column in range(8):
        inferred_sources = []
        for target in range(gate.EXPECTED_PUBLISHERS):
            target_slice = slice(
                target * gate.EXPECTED_PER_PUBLISHER,
                (target + 1) * gate.EXPECTED_PER_PUBLISHER,
            )
            normalized = permuted[target_slice, column] - column * 100_000
            source = np.unique(normalized // 1_000)
            assert source.size == 1
            source_number = int(source[0])
            inferred_sources.append(source_number)
            assert set(normalized.tolist()) == set(
                (source_number * 1_000 + np.arange(gate.EXPECTED_PER_PUBLISHER)).tolist()
            )
        assert sorted(inferred_sources) == list(range(gate.EXPECTED_PUBLISHERS))

    first = gate.publisher_block_pc1_null(
        matrix, PUBLISHERS, permutations=9, rng=np.random.default_rng(2003)
    )
    second = gate.publisher_block_pc1_null(
        matrix, PUBLISHERS, permutations=9, rng=np.random.default_rng(2003)
    )
    assert first == second
    assert first["statistic"] == "lambda1"
    assert first["null_q95_method"] == "higher"
    assert first["plus_one_p_value"] == pytest.approx(
        (first["null_exceedances_greater_or_equal"] + 1) / 10
    )


def test_sign_alignment_and_paired_cluster_bootstrap_are_deterministic() -> None:
    reference = np.array([1.0, -2.0, 3.0, -4.0, 2.0, 1.0, -1.0, 0.5])
    aligned, cosine = gate._align_loading(reference, -reference)
    np.testing.assert_allclose(aligned, reference)
    assert cosine == pytest.approx(1.0)

    q0, q1 = _paired_signal()
    first = gate.paired_publisher_cluster_bootstrap(
        q0, q1, PUBLISHERS, replicates=31, rng=np.random.default_rng(3003)
    )
    second = gate.paired_publisher_cluster_bootstrap(
        q0, q1, PUBLISHERS, replicates=31, rng=np.random.default_rng(3003)
    )
    assert first == second
    assert first["requested_replicates"] == 31
    assert first["valid_replicates"] + first["failed_replicates"] == 31
    assert first["full_data_q0_q1_loading_cosine"] > 0.99
    assert first["bootstrap_q0_q1_loading_cosine"]["q025"] > 0.99


def test_within_publisher_permutation_preserves_multisets_means_and_plus_one_p() -> None:
    q0, q1 = _paired_signal()
    q0_fit = gate._pca_fit(q0)
    q1_fit = gate._pca_fit(q1)
    q1_loading, _ = gate._align_loading(
        q0_fit["loadings"][:, 0], q1_fit["loadings"][:, 0]
    )
    q0_scores, _ = gate.center_scores_within_publisher(
        (q0 - q0_fit["mean"]) @ q0_fit["loadings"][:, 0], PUBLISHERS
    )
    q1_scores, _ = gate.center_scores_within_publisher(
        (q1 - q1_fit["mean"]) @ q1_loading, PUBLISHERS
    )
    permuted = gate.within_publisher_permute(
        q1_scores, PUBLISHERS, np.random.default_rng(4004)
    )
    for publisher in sorted(set(PUBLISHERS)):
        indices = np.flatnonzero(np.asarray(PUBLISHERS, dtype=object) == publisher)
        np.testing.assert_allclose(np.sort(permuted[indices]), np.sort(q1_scores[indices]))
        assert permuted[indices].mean() == pytest.approx(q1_scores[indices].mean())

    first = gate.within_publisher_q1_correspondence_null(
        q0_scores,
        q1_scores,
        PUBLISHERS,
        permutations=19,
        rng=np.random.default_rng(4005),
    )
    second = gate.within_publisher_q1_correspondence_null(
        q0_scores,
        q1_scores,
        PUBLISHERS,
        permutations=19,
        rng=np.random.default_rng(4005),
    )
    assert first == second
    assert first["null_q95_method"] == "higher"
    assert first["plus_one_p_value"] == pytest.approx(
        (first["null_exceedances_greater_or_equal"] + 1) / 20
    )


@pytest.mark.parametrize(
    ("q0_detected", "q1_detected", "expected"),
    [
        (True, True, "detected_under_this_null"),
        (True, False, "not_detected_under_this_null"),
        (False, True, "not_detected_under_this_null"),
        (False, False, "not_detected_under_this_null"),
    ],
)
def test_gate1_requires_both_banks_and_uses_frozen_vocabulary(
    q0_detected: bool, q1_detected: bool, expected: str
) -> None:
    assert gate.gate1_joint_outcome(q0_detected, q1_detected) == expected


@pytest.mark.parametrize(
    ("cosine_q025", "rho", "p_value", "expected"),
    [
        (0.80, 0.50, 0.05, "strong_phrasing_stability"),
        (0.79, 0.80, 0.01, "detected_below_practical_threshold"),
        (0.95, 0.49, 0.01, "detected_below_practical_threshold"),
        (0.95, 0.80, 0.051, "not_detected"),
    ],
)
def test_gate2_vocabulary_and_thresholds(
    cosine_q025: float, rho: float, p_value: float, expected: str
) -> None:
    assert (
        gate.phrasing_stability_outcome(
            cosine_q025=cosine_q025, rho=rho, p_value=p_value
        )
        == expected
    )


def test_gate2_is_not_executed_when_either_gate1_bank_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q0, q1 = _paired_signal()
    detections = iter((True, False))

    def fake_gate1(*args: object, **kwargs: object) -> dict[str, object]:
        detected = next(detections)
        return {
            "detected": detected,
            "outcome": (
                "detected_under_this_null"
                if detected
                else "not_detected_under_this_null"
            ),
        }

    def forbidden_gate2(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("Gate 2 must not execute")

    monkeypatch.setattr(gate, "publisher_block_pc1_null", fake_gate1)
    monkeypatch.setattr(gate, "paired_phrasing_stability", forbidden_gate2)
    result = gate.run_gates(
        {"q0": q0, "q1": q1},
        PUBLISHERS,
        seed=5005,
        permutations=3,
        bootstrap_replicates=3,
    )
    assert result["gate1"]["outcome"] == "not_detected_under_this_null"
    assert result["gate2"] == {
        "executed": False,
        "reason": "gate1_not_detected_under_this_null",
    }


def test_hash_change_and_output_overwrite_fail_closed(tmp_path: Path) -> None:
    bound = tmp_path / "bound.txt"
    bound.write_text("first\n", encoding="utf-8")
    before = gate._hash_records({"bound": bound})
    assert before["bound"] == gate.paired.sha256_file(bound)
    bound.write_text("second\n", encoding="utf-8")
    after = gate._hash_records({"bound": bound})
    with pytest.raises(gate.PC1GateInputError, match="bound changed during analysis"):
        gate._assert_hashes_unchanged(before, after, "bound")

    output = tmp_path / "result.json"
    gate._write_json_exclusive(output, {"first": True})
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gate._write_json_exclusive(output, {"second": True})
    assert output.read_bytes() == original


def test_stochastic_helpers_do_not_change_numpy_global_rng() -> None:
    q0, q1 = _paired_signal()
    q0_scores = q0[:, 0] - np.mean(q0[:, 0])
    q1_scores = q1[:, 0] - np.mean(q1[:, 0])
    before = np.random.get_state()
    gate.publisher_block_pc1_null(
        q0, PUBLISHERS, permutations=3, rng=np.random.default_rng(6006)
    )
    gate.paired_publisher_cluster_bootstrap(
        q0, q1, PUBLISHERS, replicates=5, rng=np.random.default_rng(6007)
    )
    gate.within_publisher_q1_correspondence_null(
        q0_scores,
        q1_scores,
        PUBLISHERS,
        permutations=5,
        rng=np.random.default_rng(6008),
    )
    after = np.random.get_state()
    _assert_rng_states_equal(before, after)


def test_protocol_freezes_scope_nulls_and_language() -> None:
    text = " ".join(gate.DEFAULT_PROTOCOL.read_text(encoding="utf-8").lower().split())
    assert "**protocol date:** 2026-09-08" in text
    assert "posthoc exploratory analysis of already inspected data" in text
    assert "corrected paired scalar contrast \\(c=s_a-s_b\\)" in text
    assert "q1 is a paraphrase bank only" in text
    assert "performs no qualitative coding" in text
    assert "assigns no semantic name to any axis" in text
    assert "pc1 has **not** previously passed a publisher-aware null" in text
    assert "10,000 null draws per bank" in text
    assert "gate 2 runs only if gate 1 passes" in text
    assert "not independent replication" in text


def test_result_claim_vocabulary_is_bounded() -> None:
    claim_boundaries = {
        "gate1_allowed_outcomes": [
            "detected_under_this_null",
            "not_detected_under_this_null",
        ]
    }
    serialized = json.dumps(claim_boundaries, sort_keys=True)
    assert "detected_under_this_null" in serialized
    assert "not_detected_under_this_null" in serialized
