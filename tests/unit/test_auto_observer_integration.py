import ast
from pathlib import Path

import pytest

from core.auto_observers import HypothesisPair
from core.auto_observer_integration import install_hypothesis_pairs, normalize_hypothesis_pairs


def _load_probe_flatten():
    tree = ast.parse(Path("nli_probe.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "flatten_hypotheses")
    namespace = {"Any": object, "Sequence": list, "List": list}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "nli_probe.py", "exec"), namespace)
    return namespace["flatten_hypotheses"]


def test_primary_extractor_accepts_generated_pairs():
    pair = HypothesisPair("A one", "B one", 0, (1,), (0,), 2.0)
    assert normalize_hypothesis_pairs([pair, {"A": "A two", "B": "B two"}]) == [
        ("A one", "B one"), ("A two", "B two")]
    class Extractor:
        hidden_size = 10
    extractor = install_hypothesis_pairs(Extractor(), [pair])
    assert extractor.hypothesis_pairs == [("A one", "B one")]
    assert (extractor.n_bots, extractor.logits_dim, extractor.cls_dim) == (1, 3, 10)


def test_sidecar_flattens_structured_pairs_in_order():
    pair = HypothesisPair("A one", "B one", 0, (1,), (0,), 2.0)
    assert _load_probe_flatten()([pair, {"pair_a": "A two", "pair_b": "B two"}, "single"]) == [
        "A one", "B one", "A two", "B two", "single"]


def test_dynamic_input_rejects_empty_collections():
    with pytest.raises(ValueError):
        normalize_hypothesis_pairs([])
