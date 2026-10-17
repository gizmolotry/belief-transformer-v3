from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class TopologicalManifest:
    status: str = "PENDING"
    run_id: str = ""
    n_observers: int = 0
    n_articles: int = 0
    dimensionality: int = 0
    ripser_budget_exceeded: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
