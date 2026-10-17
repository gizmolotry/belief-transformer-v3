"""Adapters that install generated observers into existing NLI extractors."""

from typing import Any, List, Sequence, Tuple


def normalize_hypothesis_pairs(pairs: Sequence[Any]) -> List[Tuple[str, str]]:
    result = []
    for index, pair in enumerate(pairs):
        if isinstance(pair, dict):
            a, b = pair.get("pair_a", pair.get("A")), pair.get("pair_b", pair.get("B"))
        elif hasattr(pair, "pair_a") and hasattr(pair, "pair_b"):
            a, b = pair.pair_a, pair.pair_b
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            a, b = pair
        else:
            raise ValueError(f"Invalid hypothesis pair at index {index}")
        if not isinstance(a, str) or not a.strip() or not isinstance(b, str) or not b.strip():
            raise ValueError(f"Hypothesis pair at index {index} must contain non-empty strings")
        result.append((a.strip(), b.strip()))
    if not result:
        raise ValueError("At least one hypothesis pair is required")
    return result


def install_hypothesis_pairs(extractor: Any, pairs: Sequence[Any]) -> Any:
    """Install pairs into a loaded UnifiedNLIExtractor without model reload.

    The extractor's existing forward path reads ``hypothesis_pairs`` and
    ``n_bots`` dynamically, so this adapter preserves its static YAML default
    while enabling explicit generated-pair injection with no network access.
    """
    normalized = normalize_hypothesis_pairs(pairs)
    extractor.hypothesis_pairs = normalized
    extractor.n_bots = len(normalized)
    extractor.logits_dim = extractor.n_bots * 3
    if hasattr(extractor, "hidden_size"):
        extractor.cls_dim = extractor.n_bots * extractor.hidden_size
    return extractor


__all__ = ["install_hypothesis_pairs", "normalize_hypothesis_pairs"]
