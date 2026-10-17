"""Endogenous, SVD-derived NLI observers.

The functions in this module deliberately do not depend on a particular embedding
or LLM package.  Embeddings are supplied by the caller and hypothesis generation
is injectable, which keeps experiment runs reproducible and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib import request

import numpy as np


class LLMGenerator(Protocol):
    """Minimal interface accepted by :func:`generate_auto_observers`."""

    def __call__(self, prompt: str) -> Any: ...


@dataclass(frozen=True)
class AxisExtremes:
    axis: int
    positive_indices: tuple[int, ...]
    negative_indices: tuple[int, ...]
    singular_value: float


@dataclass(frozen=True)
class HypothesisPair:
    """A contrastive observer and the corpus evidence from which it arose."""

    pair_a: str
    pair_b: str
    axis: int
    positive_indices: tuple[int, ...]
    negative_indices: tuple[int, ...]
    singular_value: float
    generated: bool = True

    def as_text(self) -> str:
        return f"A: {self.pair_a}\nB: {self.pair_b}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "A": self.pair_a,
            "B": self.pair_b,
            "axis": self.axis,
            "positive_indices": list(self.positive_indices),
            "negative_indices": list(self.negative_indices),
            "singular_value": self.singular_value,
            "generated": self.generated,
        }


def select_svd_extremes(
    embeddings: np.ndarray,
    n_axes: int = 8,
    extremes_per_side: int = 2,
) -> list[AxisExtremes]:
    """Select documents at both extremes of the leading centered SVD axes.

    SVD vector signs are mathematically arbitrary.  We canonicalize each axis by
    requiring its largest-magnitude feature loading to be positive, producing
    stable positive/negative labels across conforming LAPACK implementations.
    Ties in document scores are resolved by document index.
    """

    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("embeddings must contain at least two documents and one feature")
    if not np.isfinite(x).all():
        raise ValueError("embeddings must contain only finite values")
    if n_axes < 1 or extremes_per_side < 1:
        raise ValueError("n_axes and extremes_per_side must be positive")

    centered = x - x.mean(axis=0, keepdims=True)
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    available = min(n_axes, len(singular_values))
    per_side = min(extremes_per_side, x.shape[0])
    result: list[AxisExtremes] = []
    for axis in range(available):
        scores = u[:, axis] * singular_values[axis]
        pivot = int(np.argmax(np.abs(vt[axis])))
        if vt[axis, pivot] < 0:
            scores = -scores
        indices = np.arange(x.shape[0])
        ascending = np.lexsort((indices, scores))
        descending = np.lexsort((indices, -scores))
        result.append(
            AxisExtremes(
                axis=axis,
                positive_indices=tuple(int(i) for i in descending[:per_side]),
                negative_indices=tuple(int(i) for i in ascending[:per_side]),
                singular_value=float(singular_values[axis]),
            )
        )
    return result


def build_contrastive_prompt(
    positive_texts: Sequence[str], negative_texts: Sequence[str], axis: int
) -> str:
    """Build a bounded, explicit prompt whose output contract is JSON."""

    def render(side: str, texts: Sequence[str]) -> str:
        blocks = []
        for number, text in enumerate(texts, 1):
            cleaned = " ".join(str(text).split())[:4000]
            blocks.append(f"[{side}{number}] {cleaned}")
        return "\n".join(blocks)

    return (
        "You are constructing a neutral NLI observer from two extremes of a semantic "
        f"variance axis (axis {axis}). Identify the central ideological or framing tension. "
        "Write two concise, declarative, mutually contrastive hypotheses. Do not mention "
        "the articles, groups, axis, or this prompt. Return ONLY a JSON object with string "
        'fields "pair_a" and "pair_b".\n\nPOSITIVE EXTREME:\n'
        f"{render('P', positive_texts)}\n\nNEGATIVE EXTREME:\n{render('N', negative_texts)}"
    )


def parse_hypothesis_pair(response: Any) -> tuple[str, str]:
    """Parse common local-LLM response shapes, JSON, or labelled plain text."""

    if isinstance(response, Mapping):
        # OpenAI-compatible response or an already decoded pair.
        if "choices" in response:
            response = response["choices"][0].get("message", {}).get("content", "")
        else:
            a = response.get("pair_a", response.get("A"))
            b = response.get("pair_b", response.get("B"))
            if isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip():
                return a.strip(), b.strip()
    if not isinstance(response, str):
        raise ValueError("LLM response is not text or a supported mapping")
    text = response.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        payload = json.loads(fenced)
        if isinstance(payload, Mapping):
            return parse_hypothesis_pair(payload)
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.search(
        r"(?:Pair\s*)?A\s*:\s*(.+?)(?=\n\s*(?:Pair\s*)?B\s*:)(?:\n\s*(?:Pair\s*)?B\s*:\s*)(.+)$",
        text,
        flags=re.I | re.S,
    )
    if not match:
        raise ValueError("LLM response did not contain a contrastive hypothesis pair")
    a, b = (part.strip().strip('"') for part in match.groups())
    if not a or not b:
        raise ValueError("LLM returned an empty hypothesis")
    return a, b


class OpenAICompatibleLocalClient:
    """Small stdlib client for local OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "Qwen2.5-Coder-14B-Instruct",
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(self.url, data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=self.timeout) as response:  # nosec: caller selects URL
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"])


def _fallback_pair(axis: int) -> tuple[str, str]:
    # Deliberately generic and deterministic: malformed/offline model responses
    # must not silently introduce article text as an NLI hypothesis.
    return (
        f"The document supports the positive framing of semantic axis {axis + 1}.",
        f"The document supports the negative framing of semantic axis {axis + 1}.",
    )


def generate_auto_observers(
    embeddings: np.ndarray,
    article_texts: Sequence[str],
    n_pairs: int = 8,
    *,
    extremes_per_side: int = 2,
    llm: LLMGenerator | None = None,
    strict: bool = False,
) -> list[HypothesisPair]:
    """Generate contrastive NLI observers for the corpus's leading SVD axes.

    If ``llm`` is omitted, deterministic placeholder observers are returned.  If
    an injected generator fails or produces malformed output, the same fallback
    is used unless ``strict=True``.
    """

    texts = [str(text) for text in article_texts]
    if len(texts) != np.asarray(embeddings).shape[0]:
        raise ValueError("article_texts length must match the embedding row count")
    extremes = select_svd_extremes(embeddings, n_pairs, extremes_per_side)
    pairs: list[HypothesisPair] = []
    for item in extremes:
        generated = llm is not None
        if llm is None:
            a, b = _fallback_pair(item.axis)
        else:
            prompt = build_contrastive_prompt(
                [texts[i] for i in item.positive_indices],
                [texts[i] for i in item.negative_indices],
                item.axis,
            )
            try:
                a, b = parse_hypothesis_pair(llm(prompt))
            except Exception:
                if strict:
                    raise
                a, b = _fallback_pair(item.axis)
                generated = False
        pairs.append(HypothesisPair(a, b, item.axis, item.positive_indices, item.negative_indices, item.singular_value, generated))
    return pairs


def format_hypothesis_pairs(pairs: Sequence[HypothesisPair]) -> list[str]:
    """Compatibility adapter for consumers expecting ``A: ...\nB: ...`` strings."""

    return [pair.as_text() for pair in pairs]


__all__ = [
    "AxisExtremes",
    "HypothesisPair",
    "OpenAICompatibleLocalClient",
    "build_contrastive_prompt",
    "format_hypothesis_pairs",
    "generate_auto_observers",
    "parse_hypothesis_pair",
    "select_svd_extremes",
]
