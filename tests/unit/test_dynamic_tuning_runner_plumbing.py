from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_full_suite_exposes_and_forwards_dynamic_tuning_flags():
    source = (ROOT / "run_full_experiment_suite.py").read_text(encoding="utf-8")
    expected = [
        "--dynamic-observer-tuning",
        "--dynamic-tuning-max-iterations",
        "--dynamic-tuning-sensitivity",
        "--dynamic-tuning-damping",
        "--dynamic-tuning-tolerance",
        "--dynamic-tuning-min-alpha",
        "--dynamic-tuning-max-alpha",
    ]
    for flag in expected:
        assert source.count(flag) >= 1, f"{flag} must be declared"
    assert 'env["ASTER_DYNAMIC_OBSERVER_TUNING"]' in source
    assert '*(["--dynamic-observer-tuning"]' not in source


def test_pipeline_resolves_environment_bridge():
    source = (ROOT / "core" / "complete_pipeline.py").read_text(encoding="utf-8")
    assert "ASTER_DYNAMIC_OBSERVER_TUNING" in source
    assert 'dynamic_tuning_config_source = "environment"' in source
