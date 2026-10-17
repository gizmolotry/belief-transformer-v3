import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "prepare_full_spd_banks.py"
SPEC = importlib.util.spec_from_file_location("prepare_full_spd_banks", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)

VALIDATOR_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "exp_full_spd_tensor_validation.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location("exp_full_spd_tensor_validation_readonly", VALIDATOR_SCRIPT)
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
assert VALIDATOR_SPEC.loader is not None
sys.modules[VALIDATOR_SPEC.name] = validator
VALIDATOR_SPEC.loader.exec_module(validator)


def write_bank(path: Path, prefix: str = "x"):
    path.write_text(json.dumps([f"{prefix}-{index}" for index in range(16)]), encoding="utf-8")


def write_array(path: Path, shape, value=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full(shape, value, dtype=np.float32))


def test_pairing_and_raw_and_ordered_hashes(tmp_path):
    source = tmp_path / "bank.json"
    write_bank(source)
    bank = prepare.parse_bank("Q0", source)
    assert bank.pairs[0] == ("x-0", "x-1")
    assert len(bank.pairs) == 8
    assert bank.raw_sha256 == prepare.sha256_file(source)
    assert bank.pairs_sha256 == prepare.canonical_json_sha256(
        [{"A": f"x-{index}", "B": f"x-{index + 1}"} for index in range(0, 16, 2)]
    )
    assert list(prepare.runtime_yaml(bank)["queries"].values())[0]["A"] == "x-0"


def test_default_framing_provenance_marks_paraphrase_banks_secondary_only():
    framing = prepare.resolve_framing_provenance(None)
    assert framing["Q0"]["relation_to_Q0"] == "anchor"
    for bank in ("Q1", "Q2"):
        assert framing[bank]["relation_to_Q0"] == "paraphrase_of_Q0"
        assert framing[bank]["independently_authored_without_q0_exposure"] is False
        assert framing[bank]["declaration_basis"] == prepare.FRAMING_DECLARATION_BASIS


def test_explicit_independence_declaration_is_preserved_as_human_audited(tmp_path):
    declaration = {
        "Q0": {"relation_to_Q0": "anchor", "independently_authored_without_q0_exposure": False,
               "author": "anchor author", "source_process": "anchor process"},
        "Q1": {"relation_to_Q0": "independent_operationalization", "independently_authored_without_q0_exposure": True,
               "author": "author one", "source_process": "process one"},
        "Q2": {"relation_to_Q0": "independent_operationalization", "independently_authored_without_q0_exposure": True,
               "author": "author two", "source_process": "process two"},
    }
    source = tmp_path / "framing.json"
    source.write_text(json.dumps(declaration), encoding="utf-8")
    resolved = prepare.resolve_framing_provenance(source)
    assert resolved["Q1"]["author"] == "author one"
    assert resolved["Q2"]["source_process"] == "process two"
    assert all(value["declaration_basis"] == prepare.FRAMING_DECLARATION_BASIS for value in resolved.values())


def test_dry_run_prints_commands_without_writes(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps({"canonical_id": str(i), "publisher": "p"}) for i in range(320)), encoding="utf-8")
    monkeypatch.setattr(prepare, "REPO_ROOT", tmp_path)
    for name, relative in prepare.BANK_SOURCES.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        write_bank(target, name)
    output = tmp_path / "new-output"
    assert prepare.main(["--corpus", str(corpus), "--output-root", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert "--queries-config" in report["planned_commands"]["Q0"]
    command = report["planned_commands"]["Q0"]
    assert command[command.index("--mode") + 1] == "cls_logits"
    assert report["missing_runner_requirements"] == []
    assert not output.exists()


def test_wrong_shapes_are_rejected(tmp_path):
    paths = {}
    for name, shape in prepare.EXPECTED_SHAPES.items():
        target = tmp_path / name
        write_array(target, shape)
        paths[name] = target
    write_array(paths["T1_embeddings_raw.npy"], (319, 2048))
    with pytest.raises(prepare.PreparationError, match="expected shape"):
        prepare.validate_artifacts(paths)


def test_label_ambiguity_is_rejected():
    with pytest.raises(prepare.PreparationError, match="ambiguous|exactly one entail"):
        prepare.resolve_model_provenance(
            model_config=None,
            label_map='{"entailment": 0, "entails": 1, "contradiction": 2}',
            model_name="model",
            model_revision="revision",
        )


def test_model_class_names_follow_actual_logit_axis_order():
    model = prepare.resolve_model_provenance(
        model_config=None,
        label_map=json.dumps({
            "label2id": {"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2},
            "id2label": {"0": "contradiction", "1": "neutral", "2": "entailment"},
        }),
        model_name="local-deberta",
        model_revision="revision",
    )
    assert model["class_names"] == ["contradict", "neutral", "entail"]
    assert model["label2id"] == {"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2}
    assert model["id2label"] == {"0": "contradiction", "1": "neutral", "2": "entailment"}


def test_unbound_output_is_rejected(tmp_path):
    with pytest.raises(prepare.PreparationError, match="unbound output"):
        prepare.discover_artifacts(tmp_path)


def test_shared_x_rejection():
    record = {"T1_embeddings_raw.npy": {"sha256": "same", "raw_sha256": "same"}}
    with pytest.raises(prepare.PreparationError, match="shared-X rejection"):
        prepare.validate_distinct_t1_hashes({"Q0": record, "Q1": record, "Q2": record})


def test_overwrite_refusal(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    prepare.exclusive_json(output / "context.json", {})
    with pytest.raises(prepare.PreparationError, match="refusing overwrite"):
        prepare.exclusive_json(output / "context.json", {})


def test_planned_command_expresses_cls_rbf_d2048(tmp_path):
    command = prepare.planned_command(
        runtime_yaml_path=tmp_path / "runtime.yaml",
        output_dir=tmp_path / "output",
        corpus=tmp_path / "corpus.jsonl",
    )
    assert command[command.index("--mode") + 1] == "cls_logits"
    assert command[command.index("--kernel-type") + 1] == "rbf"
    assert command[command.index("--rks-dim") + 1] == "2048"
    assert command[command.index("--kernel-seed") + 1] == "42"
    assert prepare.missing_runner_requirements() == []


def test_execute_mocks_runner_discovers_and_binds_artifacts(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps({"canonical_id": str(i), "publisher": "p"}) for i in range(320)), encoding="utf-8")
    monkeypatch.setattr(prepare, "REPO_ROOT", tmp_path)
    for name, relative in prepare.BANK_SOURCES.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        write_bank(target, name)

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="abc123\n")
        output = Path(command[command.index("--output-root") + 1]) / "checkpoints" / "batch"
        bank_value = float(len(calls))
        for artifact_name, shape in prepare.EXPECTED_SHAPES.items():
            write_array(output / artifact_name, shape, bank_value)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(prepare.subprocess, "run", fake_run)
    output = tmp_path / "output"
    label_map = json.dumps({"contradiction": 0, "neutral": 1, "entailment": 2})
    assert prepare.main([
        "--execute", "--corpus", str(corpus), "--output-root", str(output),
        "--label-map", label_map, "--model-name", "model", "--model-revision", "rev",
    ]) == 0
    manifest = json.loads((output / "full_spd_bank_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["banks"]) == {"Q0", "Q1", "Q2"}
    normalized = validator._validate_bank_manifest(manifest)
    assert set(normalized) == {"Q0", "Q1", "Q2"}
    for name, entry in manifest["banks"].items():
        assert entry["prompt_hash"] == entry["q_bank_hash"]
        assert entry["framing_provenance"] == prepare.resolve_framing_provenance(None)[name]
        assert Path(entry["prompt_path"]).name.endswith(".json")
        assert len(entry["ordered_pairs"]) == 8
        assert len(entry["cohort"]["article_order"]) == 320
        assert entry["model"]["class_names"] == ["contradict", "neutral", "entail"]
        assert set(entry["artifacts"]) == {"X", "T0", "T1"}
        for artifact in entry["artifacts"].values():
            assert set(("path", "sha256", "shape", "dtype", "endianness")) <= set(artifact)
    runner_calls = [call for call in calls if len(call) > 1 and call[1] == "run_experiments.py"]
    assert len(runner_calls) == 3
    assert all(call[call.index("--mode") + 1] == "cls_logits" for call in runner_calls)


def test_artifacts_must_use_output_root_checkpoint_layout(tmp_path):
    (tmp_path / "bank_context.json").write_text("{}", encoding="utf-8")
    for name, shape in prepare.EXPECTED_SHAPES.items():
        write_array(tmp_path / "wrong" / "checkpoints" / "batch" / name, shape)
    with pytest.raises(prepare.PreparationError, match="missing runner artifact"):
        prepare.discover_artifacts(tmp_path)
