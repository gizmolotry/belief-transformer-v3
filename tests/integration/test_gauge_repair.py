import pytest
import torch
from core.observer_slice_transport import observer_slice_commutator

def test_gauge_repair_orthogonal_procrustes():
    """
    Asserts that the gauge repair mechanism successfully computes the optimal
    orthogonal transformation between observer frames, reducing phantom distance.
    """
    # Create mock observer charts (N, H) -> (50, 8) to ensure overdetermined system for robust IRLS
    source_chart = torch.randn(50, 8)
    # Target chart is a rotated version of source
    rotation_matrix = torch.nn.init.orthogonal_(torch.empty(8, 8))
    target_chart = source_chart @ rotation_matrix

    # Random target metric (N)
    target_metric = torch.randn(50)

    from core.observer_slice_transport import ObserverSliceTransportConfig
    cfg_unaligned = ObserverSliceTransportConfig(align_observer_gauges=False)
    cfg_aligned = ObserverSliceTransportConfig(align_observer_gauges=True)

    # Convert to expected format (dictionary of numpy arrays)
    slices = {
        "src": source_chart.numpy(),
        "tgt": target_chart.numpy()
    }

    # Without alignment, distance is high
    commutator_unaligned = observer_slice_commutator(
        slices,
        source_idx=0,
        target_idx=1,
        source_slice="src",
        target_slice="tgt",
        config=cfg_unaligned
    )
    
    # With alignment, Procrustes should undo the rotation
    commutator_aligned = observer_slice_commutator(
        slices,
        source_idx=0,
        target_idx=1,
        source_slice="src",
        target_slice="tgt",
        config=cfg_aligned
    )
    
    # The absolute length of the path should be strictly smaller when gauges are aligned.
    # We check the total path action rather than the holonomy (difference in path lengths),
    # since two unaligned paths can accidentally have similar lengths resulting in a small unaligned holonomy.
    assert commutator_aligned["semantic_first_action"] < commutator_unaligned["semantic_first_action"]
