from __future__ import annotations

from pathlib import Path

import pytest

from core import complete_pipeline


QUERY_BANKS = (
    "config/prompt_banks/probe_hypotheses_paraphrase_a.json",
    "config/prompt_banks/probe_hypotheses_paraphrase_b.json",
)


@pytest.mark.parametrize("queries_config", QUERY_BANKS)
def test_multi_observer_wrapper_forwards_exact_query_bank(monkeypatch, queries_config):
    received = []

    class ReachedInitializer(RuntimeError):
        pass

    def recording_initializer(**kwargs):
        received.append(kwargs["queries_config"])
        raise ReachedInitializer

    monkeypatch.setattr(complete_pipeline, "initialize_full_pipeline", recording_initializer)

    with pytest.raises(ReachedInitializer):
        complete_pipeline.run_multi_observer_experiment_simple(
            articles=[],
            seeds=[42],
            queries_config=queries_config,
            output_dir=Path("."),
        )

    assert received == [queries_config]


@pytest.mark.parametrize("queries_config", QUERY_BANKS)
def test_initialize_pipeline_passes_exact_query_bank_to_nli(monkeypatch, queries_config):
    received = []

    class RecordingNLIExtractor:
        def __init__(
            self,
            device="cuda",
            queries_config="config/framing_queries.yaml",
            paragraph_aware=True,
            paragraph_weights=None,
        ):
            received.append(queries_config)
            self.output_dim = 24

    monkeypatch.setattr(complete_pipeline, "NLIExtractor", RecordingNLIExtractor)

    complete_pipeline.initialize_full_pipeline(
        device="cpu",
        use_gru=False,
        use_rks=False,
        use_attention=False,
        queries_config=queries_config,
    )

    assert received == [queries_config]


def test_initialize_pipeline_rejects_unhonored_query_bank(monkeypatch):
    class LegacyNLIExtractor:
        def __init__(self, device="cuda", paragraph_aware=True, paragraph_weights=None):
            raise AssertionError("constructor must not run after query-bank downgrade")

    monkeypatch.setattr(complete_pipeline, "NLIExtractor", LegacyNLIExtractor)

    with pytest.raises(TypeError, match="queries_config cannot be honored"):
        complete_pipeline.initialize_full_pipeline(
            device="cpu",
            use_gru=False,
            use_rks=False,
            use_attention=False,
            queries_config=QUERY_BANKS[0],
        )
