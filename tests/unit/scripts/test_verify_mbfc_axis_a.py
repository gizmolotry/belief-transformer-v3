from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import pearsonr

from scripts import verify_mbfc_axis_a as verify


TEST_PUBLISHERS = ("cnn.com", "foxnews.com", "presstv.ir", "example.test")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(verify, "EXPECTED_ARTICLES", 8)
    monkeypatch.setattr(verify, "EXPECTED_OBSERVERS", 2)
    monkeypatch.setattr(verify, "EXPECTED_HIDDEN_DIM", 3)
    monkeypatch.setattr(verify, "EXPECTED_ARTICLES_PER_PUBLISHER", 2)
    monkeypatch.setattr(verify, "EXPECTED_PUBLISHER_ORDER", TEST_PUBLISHERS)

    generator = torch.Generator().manual_seed(1729)
    embeddings = torch.randn((8, 2, 3), generator=generator, dtype=torch.float64)
    input_path = tmp_path / "embeddings.pt"
    corpus_path = tmp_path / "corpus.jsonl"
    mbfc_path = tmp_path / "mbfc.json"
    torch.save(
        {
            "nli_pairs": [{"cls_per_bot": row} for row in embeddings],
            "n_articles": 8,
        },
        input_path,
    )
    publisher_rows = TEST_PUBLISHERS + TEST_PUBLISHERS
    corpus_path.write_text(
        "".join(
            json.dumps({"canonical_id": f"article-{index}", "publisher": publisher}) + "\n"
            for index, publisher in enumerate(publisher_rows)
        ),
        encoding="utf-8",
    )
    mbfc_path.write_text(
        json.dumps(
            {
                "cnn.com": -3.0,
                "foxnews.com": 4.0,
                "presstv.ir": 2.0,
                "example.test": -1.0,
            }
        ),
        encoding="utf-8",
    )
    return input_path, corpus_path, mbfc_path


def test_fixed_axis_matches_historical_arithmetic_and_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, corpus_path, _ = _write_fixture(tmp_path, monkeypatch)
    embeddings = verify.load_cls_per_bot(input_path)
    publishers = verify.load_corpus_publishers(corpus_path)

    first = verify.compute_fixed_axis(embeddings, publishers)
    second = verify.compute_fixed_axis(embeddings, publishers)

    flat = embeddings.reshape(-1, embeddings.shape[-1])
    flat_centered = flat - flat.mean(dim=0)
    _, _, global_v = torch.svd(flat_centered)
    residual = verify.project_out(embeddings, global_v[:, 0])
    article = residual.mean(dim=1)
    article_centered = article - article.mean(dim=0)
    _, _, article_v = torch.svd(article_centered)
    manual_scores = (article * article_v[:, 0]).sum(dim=1).numpy()
    manual_scores, _ = verify.apply_cnn_fox_sign(manual_scores, publishers)

    assert first["article_scores"] == pytest.approx(manual_scores, abs=0.0)
    assert np.array_equal(first["article_scores"], second["article_scores"])
    assert torch.equal(first["semantic_pc1"], second["semantic_pc1"])
    assert torch.equal(first["article_pc1"], second["article_pc1"])


def test_cnn_fox_sign_convention_flips_only_when_required() -> None:
    publishers = ["cnn.com", "foxnews.com", "cnn.com", "foxnews.com"]
    scores = np.asarray([3.0, 1.0, 5.0, 2.0])
    signed, metadata = verify.apply_cnn_fox_sign(scores, publishers)

    assert np.array_equal(signed, -scores)
    assert metadata["flipped"] is True
    assert metadata["cnn_mean_after"] < metadata["fox_mean_after"]

    unchanged, metadata = verify.apply_cnn_fox_sign(-scores, publishers)
    assert np.array_equal(unchanged, -scores)
    assert metadata["flipped"] is False


def test_loo_is_complete_and_uses_fixed_full_sample_scores() -> None:
    publishers = ["a", "b", "c", "d", "e"]
    geometric = np.asarray([-2.0, -0.5, 0.2, 1.4, 3.0])
    external = np.asarray([-4.0, -1.0, 1.0, 2.0, 5.0])
    records = verify.fixed_axis_leave_one_publisher_out(publishers, geometric, external)

    assert [record["held_out_publisher"] for record in records] == publishers
    assert all(record["n_publishers"] == 4 for record in records)
    keep = np.asarray([False, True, True, True, True])
    expected = pearsonr(geometric[keep], external[keep])
    assert records[0]["pearson"]["r"] == pytest.approx(expected.statistic)
    assert records[0]["pearson"]["p_value"] == pytest.approx(expected.pvalue)


@pytest.mark.parametrize(
    "failure",
    [
        "shape",
        "nonfinite",
        "artifact_count",
        "corpus_count",
        "publisher_order",
        "publisher_count",
        "missing_mbfc",
        "nonfinite_mbfc",
    ],
)
def test_fail_closed_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    input_path, corpus_path, mbfc_path = _write_fixture(tmp_path, monkeypatch)
    if failure in {"shape", "nonfinite", "artifact_count"}:
        payload = torch.load(input_path, map_location="cpu", weights_only=True)
        if failure == "shape":
            payload["nli_pairs"][0]["cls_per_bot"] = torch.zeros((2, 4))
        elif failure == "nonfinite":
            payload["nli_pairs"][0]["cls_per_bot"][0, 0] = float("nan")
        else:
            payload["n_articles"] = 7
        torch.save(payload, input_path)
    elif failure in {"corpus_count", "publisher_order", "publisher_count"}:
        rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
        if failure == "corpus_count":
            rows.pop()
        elif failure == "publisher_order":
            rows[0]["publisher"], rows[1]["publisher"] = (
                rows[1]["publisher"],
                rows[0]["publisher"],
            )
        else:
            rows[-1]["publisher"] = "cnn.com"
        corpus_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    else:
        mbfc = json.loads(mbfc_path.read_text(encoding="utf-8"))
        if failure == "missing_mbfc":
            del mbfc["presstv.ir"]
        else:
            mbfc["presstv.ir"] = float("inf")
        mbfc_path.write_text(json.dumps(mbfc), encoding="utf-8")

    with pytest.raises(verify.VerificationInputError):
        verify.build_verification(input_path, corpus_path, mbfc_path)


def test_payload_is_hash_bound_deterministic_and_labels_fixed_axis_loo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, corpus_path, mbfc_path = _write_fixture(tmp_path, monkeypatch)
    first = verify.build_verification(input_path, corpus_path, mbfc_path)
    second = verify.build_verification(input_path, corpus_path, mbfc_path)

    assert first == second
    files = first["provenance"]["files"]
    assert files["input_artifact"]["sha256"] == _sha256(input_path)
    assert files["corpus"]["sha256"] == _sha256(corpus_path)
    assert files["mbfc_mapping"]["sha256"] == _sha256(mbfc_path)
    assert files["source_runner"]["sha256"] == _sha256(Path(verify.__file__))
    assert files["historical_script"]["sha256"] == _sha256(verify.HISTORICAL_SCRIPT)
    assert first["provenance"]["alignment"]["mode"] == "row_order_only"
    assert first["provenance"]["rng"] == {"used": False, "seed": None}
    convention = first["definition"]["loo_convention"]
    assert convention["name"] == "fixed_full_sample_axis_remove_one_publisher"
    assert convention["semantic_pc1_refit_per_exclusion"] is False
    assert convention["article_pc1_refit_per_exclusion"] is False
    assert convention["sign_refit_per_exclusion"] is False
    assert len(first["results"]["leave_one_publisher_out"]) == len(TEST_PUBLISHERS)
    assert first["results"]["press_tv"]["publisher"] == "presstv.ir"
    assert len(first["provenance"]["derived_array_hashes"]) == 4
    assert "not frozen before the result" in " ".join(first["definition"]["caveats"])


def test_cli_defaults() -> None:
    args = verify.parse_args([])
    assert args.input == verify.DEFAULT_INPUT
    assert args.corpus == verify.DEFAULT_CORPUS
    assert args.mbfc == verify.DEFAULT_MBFC
    assert args.output == verify.DEFAULT_OUTPUT


def test_output_is_deterministic_and_never_overwritten(tmp_path: Path) -> None:
    payload = {"value": 1.25, "nested": {"stable": True}}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    verify.write_json_exclusive(payload, first)
    verify.write_json_exclusive(payload, second)

    assert first.read_bytes() == second.read_bytes()
    original = first.read_bytes()
    with pytest.raises(FileExistsError):
        verify.write_json_exclusive({"value": 999}, first)
    assert first.read_bytes() == original
