import numpy as np
import pytest

from core.auto_observers import (
    format_hypothesis_pairs,
    generate_auto_observers,
    parse_hypothesis_pair,
    select_svd_extremes,
)


def test_svd_extremes_are_deterministic_and_find_known_axis():
    x = np.array([[-4.0, 0.0], [-2.0, 0.1], [2.0, -0.1], [4.0, 0.0]])
    first = select_svd_extremes(x, n_axes=1, extremes_per_side=1)
    assert first == select_svd_extremes(x, n_axes=1, extremes_per_side=1)
    assert {first[0].positive_indices[0], first[0].negative_indices[0]} == {0, 3}


def test_generation_injects_extreme_texts_and_formats():
    x = np.array([[-4.0, 0.0], [-2.0, 0.1], [2.0, -0.1], [4.0, 0.0]])
    prompts = []
    def fake_llm(prompt):
        prompts.append(prompt)
        return '{"pair_a":"Institutions intervene.","pair_b":"Individuals decide."}'
    pairs = generate_auto_observers(
        x, ["far left", "near left", "near right", "far right"],
        n_pairs=1, extremes_per_side=1, llm=fake_llm,
    )
    assert (pairs[0].pair_a, pairs[0].pair_b) == ("Institutions intervene.", "Individuals decide.")
    assert "far left" in prompts[0] and "far right" in prompts[0]
    assert format_hypothesis_pairs(pairs) == ["A: Institutions intervene.\nB: Individuals decide."]


@pytest.mark.parametrize("response", [
    {"A": "Alpha", "B": "Beta"},
    "```json\n{\"pair_a\": \"Alpha\", \"pair_b\": \"Beta\"}\n```",
    "Pair A: Alpha\nPair B: Beta",
    {"choices": [{"message": {"content": "A: Alpha\nB: Beta"}}]},
])
def test_response_parser(response):
    assert parse_hypothesis_pair(response) == ("Alpha", "Beta")


def test_fallback_alignment_and_strict_mode():
    x = np.array([[-1.0], [1.0]])
    pair = generate_auto_observers(x, ["one", "two"], n_pairs=1)[0]
    assert not pair.generated and "semantic axis 1" in pair.pair_a
    with pytest.raises(ValueError, match="length"):
        generate_auto_observers(x, ["one"])
    bad = lambda _: "not a pair"
    assert not generate_auto_observers(x, ["one", "two"], n_pairs=1, llm=bad)[0].generated
    with pytest.raises(ValueError):
        generate_auto_observers(x, ["one", "two"], n_pairs=1, llm=bad, strict=True)
