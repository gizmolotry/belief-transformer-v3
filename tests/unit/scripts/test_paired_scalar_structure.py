from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import paired_scalar_structure as paired


def _logits(scale: np.ndarray) -> np.ndarray:
    scale = np.asarray(scale, dtype=np.float64)
    return np.stack((-scale, np.zeros_like(scale), scale), axis=-1)


def _write_inputs(tmp_path: Path, n: int = 8) -> tuple[Path, Path, Path, Path]:
    corpus = tmp_path / "corpus.jsonl"
    q0 = tmp_path / "q0.yaml"
    q1 = tmp_path / "q1.json"
    checkpoint = tmp_path / "checkpoint.jsonl"
    corpus_rows = []
    checkpoint_rows = []
    for index in range(n):
        canonical_id = f"article-{index:02d}"
        corpus_rows.append(
            {
                "canonical_id": canonical_id,
                "publisher": f"publisher-{index % 3}",
                "content": f"Article text {index}",
            }
        )
        base = np.linspace(-1.0, 1.0, 8) * (index - n / 2)
        q0_a = _logits(base + 0.07 * np.arange(8))
        q0_b = _logits(-0.65 * base + 0.03 * np.arange(8))
        q1_a = _logits(0.9 * base + 0.09 * np.sin(np.arange(8) + index))
        q1_b = _logits(-0.7 * base + 0.05 * np.cos(np.arange(8) + index))
        checkpoint_rows.append(
            {
                "canonical_id": canonical_id,
                "q0": {"A": q0_a.tolist(), "B": q0_b.tolist()},
                "q1": {"A": q1_a.tolist(), "B": q1_b.tolist()},
            }
        )
    corpus.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in corpus_rows), encoding="utf-8"
    )
    q0.write_text(
        "queries:\n"
        + "".join(
            f"  pair_{index}:\n    A: \"Q0 A {index}\"\n    B: \"Q0 B {index}\"\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    q1.write_text(
        json.dumps([value for index in range(8) for value in (f"Q1 A {index}", f"Q1 B {index}")]),
        encoding="utf-8",
    )
    checkpoint.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in checkpoint_rows),
        encoding="utf-8",
    )
    return corpus, q0, q1, checkpoint


def test_paired_scalar_score_responds_to_b_and_has_expected_symmetries() -> None:
    a = _logits(np.linspace(-0.8, 0.8, 8))[None, ...]
    b = _logits(np.linspace(0.6, -0.6, 8))[None, ...]
    score_a, score_b, contrast = paired.scores_from_logits(a, b)
    changed_a, changed_b, changed_contrast = paired.scores_from_logits(a, b + np.array([0.4, 0.0, -0.4]))
    _, _, swapped = paired.scores_from_logits(b, a)
    _, _, self_contrast = paired.scores_from_logits(a, a)

    np.testing.assert_array_equal(changed_a, score_a)
    assert not np.array_equal(changed_b, score_b)
    assert not np.array_equal(changed_contrast, contrast)
    np.testing.assert_allclose(swapped, -contrast, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(self_contrast, np.zeros_like(self_contrast))


def test_score_formula_uses_declared_class_order() -> None:
    a = np.zeros((1, 8, 3), dtype=np.float64)
    b = np.zeros_like(a)
    a[0, 0] = [2.0, 0.0, -2.0]
    score_a, _, _ = paired.scores_from_logits(a, b)
    expected = paired.expl.stable_softmax(a)[0, 0, 2] - paired.expl.stable_softmax(a)[0, 0, 0]
    assert score_a[0, 0] == pytest.approx(expected)
    assert paired._class_indices({"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2}) == (2, 0)
    with pytest.raises(paired.PairedScalarInputError, match="unexpected NLI class mapping"):
        paired._class_indices({"CONTRADICTION": 2, "NEUTRAL": 1, "ENTAILMENT": 0})


def test_full_spectrum_reports_all_components_and_matches_top_two() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(30, 8))
    spectrum = paired.full_pca_spectrum(matrix)
    top_two = paired.expl.deterministic_pca(matrix)["explained_variance_ratio"]

    assert spectrum.shape == (8,)
    assert float(spectrum.sum()) == pytest.approx(1.0)
    np.testing.assert_allclose(spectrum[:2], top_two, rtol=1e-13, atol=1e-15)


def test_checkpoint_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    _, _, _, checkpoint = _write_inputs(tmp_path, n=4)
    rows = checkpoint.read_text(encoding="utf-8").splitlines()
    checkpoint.write_text(rows[0] + "\n" + rows[0] + "\n", encoding="utf-8")
    with pytest.raises(paired.PairedScalarInputError, match="duplicate checkpoint canonical_id"):
        paired.load_checkpoint(checkpoint)


def test_prompt_loader_rejects_malformed_paraphrase_bank(tmp_path: Path) -> None:
    _, q0, q1, _ = _write_inputs(tmp_path, n=4)
    q1.write_text(json.dumps(["only one"]), encoding="utf-8")
    with pytest.raises(paired.PairedScalarInputError, match="16-string"):
        paired.load_prompt_pairs(q0, q1)


def test_checkpoint_analysis_binds_ids_inputs_and_channel_names(tmp_path: Path) -> None:
    corpus, q0, q1, checkpoint = _write_inputs(tmp_path, n=10)
    result = paired.build_checkpoint_analysis(
        checkpoint, corpus, q0, q1, permutations=7, seed=11, expected_n=10
    )

    assert result["status"] == "exploratory_posthoc_analysis_of_existing_unmanifested_extraction"
    assert result["provenance"]["article_ids_exactly_match_corpus_order"] is True
    assert set(result["results"]) == {"A_side", "paired_scalar"}
    assert len(result["results"]["paired_scalar"]["q0_full_explained_variance_ratio"]) == 8
    assert result["claim_boundaries"]["not_equivalent_to"].startswith("hidden-state")
    assert result["analysis"]["permutations_per_null"] == 7


def test_checkpoint_analysis_rejects_wrong_article_order(tmp_path: Path) -> None:
    corpus, q0, q1, checkpoint = _write_inputs(tmp_path, n=6)
    rows = corpus.read_text(encoding="utf-8").splitlines()
    corpus.write_text("\n".join(reversed(rows)) + "\n", encoding="utf-8")
    with pytest.raises(paired.PairedScalarInputError, match="exactly match corpus order"):
        paired.build_checkpoint_analysis(
            checkpoint, corpus, q0, q1, permutations=2, seed=2, expected_n=6
        )


def test_exclusive_json_writer_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    paired._write_json_exclusive(output, {"first": True})
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        paired._write_json_exclusive(output, {"second": True})
    assert output.read_bytes() == before


def test_extraction_batches_both_sides_and_freezes_manifest_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, q0, q1, _ = _write_inputs(tmp_path, n=3)
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "pytorch_model.bin").write_bytes(b"fake model weights")
    model_path = str(model_dir)
    corpus_rows = [
        {
            "canonical_id": f"article-{index:02d}",
            "publisher": "publisher.test",
            "content": f"p0-{index}\n\np1-{index}\n\np2-{index}",
        }
        for index in range(3)
    ]
    corpus.write_text(
        "".join(json.dumps(row) + "\n" for row in corpus_rows), encoding="utf-8"
    )
    call_sizes: list[int] = []

    class FakeBatch(dict):
        def to(self, device: str) -> "FakeBatch":
            assert device == "cpu"
            return self

    class FakeTokenizer:
        def __call__(self, texts: list[str], hypotheses: list[str], **kwargs: object) -> FakeBatch:
            assert kwargs["truncation"] == "only_first"
            call_sizes.append(len(texts))
            marker = []
            for text, hypothesis in zip(texts, hypotheses, strict=True):
                paragraph = int(text[1])
                article = int(text.rsplit("-", 1)[1])
                bank, side_name, axis = hypothesis.split()
                side = 1 if side_name == "A" else -1
                bank_offset = 0.03 if bank == "Q1" else 0.0
                marker.append(
                    side * (paragraph + 1) + 0.2 * article + 0.05 * int(axis) + bank_offset
                )
            return FakeBatch(marker=torch.tensor(marker, dtype=torch.float32))

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, local_files_only: bool) -> FakeTokenizer:
            assert path == model_path and local_files_only is True
            return FakeTokenizer()

    class FakeModel:
        config = SimpleNamespace(
            label2id={"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2}
        )

        def to(self, device: str) -> "FakeModel":
            assert device == "cpu"
            return self

        def eval(self) -> "FakeModel":
            return self

        def __call__(self, marker: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(logits=torch.stack((-marker, torch.zeros_like(marker), marker), dim=1))

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path: str, local_files_only: bool) -> FakeModel:
            assert path == model_path and local_files_only is True
            return FakeModel()

    monkeypatch.setattr(paired, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(paired, "AutoModelForSequenceClassification", FakeAutoModel)
    output_dir = tmp_path / "fresh-output"
    checkpoint = paired.extract_features(
        corpus,
        q0,
        q1,
        output_dir,
        batch_size=2,
        device="cpu",
        model_path=model_path,
        expected_n=3,
    )

    assert checkpoint.exists()
    assert (output_dir / "pre_run_manifest.json").exists()
    assert call_sizes and max(call_sizes) <= 2
    ids, logits = paired.load_checkpoint(checkpoint)
    assert ids == [f"article-{index:02d}" for index in range(3)]
    expected_marker = 0.5 * 1 + 0.3 * 2 + 0.2 * 3
    assert logits["q0"]["A"][0, 0] == pytest.approx(
        [-expected_marker, 0.0, expected_marker]
    )
    assert logits["q0"]["B"][0, 0] == pytest.approx(
        [expected_marker, 0.0, -expected_marker]
    )
    manifest_path = output_dir / "pre_run_manifest.json"
    result = paired.build_checkpoint_analysis(
        checkpoint,
        corpus,
        q0,
        q1,
        permutations=2,
        seed=42,
        expected_n=3,
        manifest_path=manifest_path,
    )
    assert result["status"] == "fresh_manifest_bound_extraction_and_analysis"
    assert result["provenance"]["pre_run_manifest"]["sha256"] == paired.sha256_file(
        manifest_path
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["articles"][0]["text_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(paired.PairedScalarInputError, match="article identities/text hashes"):
        paired.build_checkpoint_analysis(
            checkpoint,
            corpus,
            q0,
            q1,
            permutations=2,
            seed=42,
            expected_n=3,
            manifest_path=manifest_path,
        )
