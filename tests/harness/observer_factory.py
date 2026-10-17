import numpy as np

def generate_deterministic_embedding_slice(seed=42, shape=(10, 128)):
    """
    Shared utility for generating deterministic embedding slices.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape)

def generate_adversarial_trap():
    """
    Generates deterministic adversarial traps for gating probes.
    """
    return {"trap": "entity_swap_salient", "severity": 0.9}
