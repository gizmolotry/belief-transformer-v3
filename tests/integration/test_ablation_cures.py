import pytest
from core.pipeline_config import PipelineRuntimeConfig

def test_pipeline_config_ablation_toggles():
    """Verify that the 80/20 optimization toggles are present and backward compatible."""
    config = PipelineRuntimeConfig()
    assert config.use_nli_contrast_gate is False
    assert config.align_observer_gauges is False
    assert config.transport_metric_type == "euclidean"

    active_config = PipelineRuntimeConfig.from_mode_config({
        "use_nli_contrast_gate": True,
        "align_observer_gauges": True,
        "transport_metric_type": "riemannian"
    })
    
    assert active_config.use_nli_contrast_gate is True
    assert active_config.align_observer_gauges is True
    assert active_config.transport_metric_type == "riemannian"

    kwargs = active_config.to_initialize_kwargs()
    assert kwargs.get("use_nli_contrast_gate") is True
    assert kwargs.get("align_observer_gauges") is True
    assert kwargs.get("transport_metric_type") == "riemannian"
