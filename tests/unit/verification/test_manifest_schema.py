import pytest
import torch
import numpy as np

# We expect Codex to implement this strict verifier in analysis.verification.verify_run
from analysis.verification.verify_run import (
    verify_manifest, 
    ManifestVerificationError, 
    ArtifactSchemaVersionError, 
    ProvenanceMissingError
)

def get_valid_provenance():
    return {
        "schema_version": "1.0",
        "pipeline_stage": "track_2_fusion",
        "target_field": "dirichlet_fused",
        "tensor_sha256": "fakehash123",
        "tensor_shape": [30, 8, 2048],
        "tensor_dtype": "float32",
        "article_id_hash": "hash_of_ordered_ids",
        "dataset_hash": "dataset_hash_xyz",
        "resolved_config_hash": "config_hash_456",
        "model_identity": "deberta-v2-xlarge-mnli",
        "git_commit": "abcdef123456"
    }

def test_valid_manifest():
    data = {
        "manifest": get_valid_provenance(),
        "dirichlet_fused": torch.randn(30, 8, 2048)
    }
    # Should pass without error
    assert verify_manifest(data, expected_stage="track_2_fusion") is True

def test_missing_provenance():
    data = {"dirichlet_fused": torch.randn(30, 8, 2048)}
    # Entire manifest is missing
    with pytest.raises(ProvenanceMissingError):
        verify_manifest(data, expected_stage="track_2_fusion")

def test_unknown_schema_version():
    data = {
        "manifest": get_valid_provenance(),
        "dirichlet_fused": torch.randn(30, 8, 2048)
    }
    data["manifest"]["schema_version"] = "v999.0"
    with pytest.raises(ArtifactSchemaVersionError):
        verify_manifest(data, expected_stage="track_2_fusion")

def test_missing_declared_field():
    data = {
        "manifest": get_valid_provenance(),
        "features": torch.randn(30, 8, 2048) # Has legacy field instead of declared field
    }
    # Should not fall back to 'features'
    with pytest.raises(ManifestVerificationError, match="Missing declared target_field: dirichlet_fused"):
        verify_manifest(data, expected_stage="track_2_fusion")

def test_corrupted_tensors_nan():
    data = {
        "manifest": get_valid_provenance(),
        "dirichlet_fused": torch.full((30, 8, 2048), float('nan'))
    }
    with pytest.raises(ManifestVerificationError, match="Tensor contains NaNs or Infs"):
        verify_manifest(data, expected_stage="track_2_fusion")

def test_wrong_pipeline_stage():
    data = {
        "manifest": get_valid_provenance(),
        "dirichlet_fused": torch.randn(30, 8, 2048)
    }
    with pytest.raises(ManifestVerificationError, match="Stage mismatch"):
        verify_manifest(data, expected_stage="track_5_assembly")
