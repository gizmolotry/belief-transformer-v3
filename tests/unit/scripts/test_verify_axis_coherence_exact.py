from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import verify_axis_coherence_exact as exact


def _cache(path: Path, shape: tuple[int, int, int] = (4, 2, 3)) -> Path:
    generator = torch.Generator().manual_seed(17)
    torch.save(
        {
            key: torch.randn(shape, generator=generator)
            for key in exact.TENSOR_KEYS
        },
        path,
    )
    return path


def test_global_pc1_removal_and_coherence_reproduce_direct_formula() -> None:
    generator = torch.Generator().manual_seed(5)
    groups = [torch.randn((5, 2, 4), generator=generator) for _ in range(3)]
    residuals, evr = exact.remove_global_pc1(groups)

    flat = torch.cat([group.reshape(-1, 4) for group in groups])
    centered = flat - flat.mean(dim=0, keepdim=True)
    _, singular_values, right_vectors = torch.svd(centered)
    pc1 = right_vectors[:, 0]
    expected_first = groups[0].reshape(-1, 4)
    expected_first = expected_first - (expected_first @ pc1).unsqueeze(1) * pc1

    torch.testing.assert_close(residuals[0], expected_first.reshape(groups[0].shape))
    assert evr == pytest.approx(float(singular_values[0] ** 2 / (singular_values**2).sum()))

    matrix = exact.axis_coherence_matrix(torch.cat(residuals, dim=1))
    normalized = torch.nn.functional.normalize(torch.cat(residuals, dim=1), p=2, dim=-1)
    direct = torch.sum(normalized[:, 0] * normalized[:, 1], dim=-1).mean().item()
    assert matrix[0, 1] == direct
    assert exact.subset_coherence(matrix, (0, 1)) == direct


def test_exhaustive_upper_tail_matches_brute_force() -> None:
    matrix = np.array(
        [
            [0.0, 0.9, 0.2, -0.1, 0.4, 0.0],
            [0.9, 0.0, 0.3, 0.1, 0.2, -0.2],
            [0.2, 0.3, 0.0, 0.8, -0.1, 0.1],
            [-0.1, 0.1, 0.8, 0.0, 0.0, 0.3],
            [0.4, 0.2, -0.1, 0.0, 0.0, 0.7],
            [0.0, -0.2, 0.1, 0.3, 0.7, 0.0],
        ]
    )
    observed = exact.subset_coherence(matrix, (0, 1))
    statistics = [
        exact.subset_coherence(matrix, subset)
        for subset in itertools.combinations(range(6), 2)
    ]

    result = exact.exhaustive_upper_tail(
        matrix, subset_size=2, observed=observed, batch_size=3
    )

    expected_count = sum(value >= observed for value in statistics)
    assert result["total_subsets"] == math.comb(6, 2)
    assert result["upper_tail_count_ge_observed"] == expected_count
    assert result["exact_upper_tail_probability"] == expected_count / len(statistics)
    assert result["null_mean"] == pytest.approx(float(np.mean(statistics)))


@pytest.mark.parametrize("failure", ["missing", "shape", "nonfinite", "integer", "dtype"])
def test_cache_loader_fails_closed(tmp_path: Path, failure: str) -> None:
    path = _cache(tmp_path / "input.pt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if failure == "missing":
        del payload["rand_emb"]
    elif failure == "shape":
        payload["ctrl_emb"] = torch.zeros((4, 3, 3))
    elif failure == "nonfinite":
        payload["ideo_emb"][0, 0, 0] = float("nan")
    elif failure == "integer":
        payload["rand_emb"] = payload["rand_emb"].to(torch.int64)
    else:
        payload["rand_emb"] = payload["rand_emb"].to(torch.float64)
    torch.save(payload, path)

    with pytest.raises(exact.AxisCoherenceVerificationError):
        exact.load_embedding_cache(path, expected_n=4, expected_axes=2, expected_dim=3)


def test_build_verification_hashes_inputs_and_marks_historical_boundary(tmp_path: Path) -> None:
    input_path = _cache(tmp_path / "input.pt")
    result = exact.build_verification(
        input_path, expected_n=4, expected_axes=2, expected_dim=3
    )

    expected_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert result["provenance"]["input_sha256"] == expected_hash
    assert len(result["provenance"]["runner_sha256"]) == 64
    assert result["status"] == "new_verification_conditional_on_surviving_unmanifested_cache"
    assert result["claim_boundary"]["historical_run_reproduced"] is False
    assert set(result["results"]["observed_coherence"]) == {
        "ideology",
        "structure",
        "random",
    }
    assert result["results"]["exact_null"]["total_subsets"] == math.comb(6, 2)


def test_json_writer_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    exact.write_json_exclusive(output, {"first": True})
    before = output.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        exact.write_json_exclusive(output, {"second": True})

    assert output.read_bytes() == before
    assert json.loads(before) == {"first": True}


def test_cli_writes_valid_small_exact_result(tmp_path: Path) -> None:
    input_path = _cache(tmp_path / "input.pt")
    output = tmp_path / "result.json"

    assert exact.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--expected-n",
            "4",
            "--expected-axes",
            "2",
            "--expected-dim",
            "3",
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["results"]["exact_null"]["total_subsets"] == 15
    assert payload["provenance"]["expected_tensor_shape"] == [4, 2, 3]
