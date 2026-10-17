from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from core.ablation_dag import build_orchestration_contract


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ReflectorScriptSpec:
    script_id: str
    diagnostic_type: str
    module: str
    script_path: str
    summary_artifact: str
    produced_artifacts: Sequence[str]
    upstream_script_ids: Sequence[str]
    claim_families: Sequence[str]
    description: str

    @property
    def path(self) -> Path:
        return REPO_ROOT / self.script_path

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


REFLECTOR_SCRIPT_ORDER: Sequence[str] = (
    "counterfactual_intervention_suite",
    "reflector_graph_fix_ablation",
    "reflector_transport_ablation",
    "reflector_adversarial_transport_suite",
    "reflector_semantic_gate_ablation",
)


REFLECTOR_SCRIPTS: Dict[str, ReflectorScriptSpec] = {
    "counterfactual_intervention_suite": ReflectorScriptSpec(
        script_id="counterfactual_intervention_suite",
        diagnostic_type="counterfactual_intervention_suite",
        module="scripts.run_counterfactual_intervention_suite",
        script_path="scripts/run_counterfactual_intervention_suite.py",
        summary_artifact="summary_json",
        produced_artifacts=("counterfactual_rows_csv", "summary_json"),
        upstream_script_ids=(),
        claim_families=(
            "controlled_counterfactual_recovery",
            "observer_force_vs_content",
            "track4_traversal_baseline",
        ),
        description="Root controlled counterfactual packet for Reflector tests.",
    ),
    "reflector_graph_fix_ablation": ReflectorScriptSpec(
        script_id="reflector_graph_fix_ablation",
        diagnostic_type="reflector_graph_fix_ablation",
        module="scripts.run_reflector_graph_fix_ablation",
        script_path="scripts/run_reflector_graph_fix_ablation.py",
        summary_artifact="json",
        produced_artifacts=("json", "csv", "claim_matrix", "legacy_summary_json", "counterfactual_rows_csv"),
        upstream_script_ids=("counterfactual_intervention_suite",),
        claim_families=("graph_support_repair", "track4_path_separation"),
        description="Ablates graph-support fixes over the controlled counterfactual packet.",
    ),
    "reflector_transport_ablation": ReflectorScriptSpec(
        script_id="reflector_transport_ablation",
        diagnostic_type="reflector_transport_ablation",
        module="scripts.run_reflector_transport_ablation",
        script_path="scripts/run_reflector_transport_ablation.py",
        summary_artifact="json",
        produced_artifacts=("json", "csv", "claim_matrix", "legacy_summary_json", "counterfactual_rows_csv"),
        upstream_script_ids=("counterfactual_intervention_suite", "reflector_graph_fix_ablation"),
        claim_families=("observer_transport", "content_roads_observer_costs"),
        description="Tests content-topology roads with observer displacement as transport cost.",
    ),
    "reflector_adversarial_transport_suite": ReflectorScriptSpec(
        script_id="reflector_adversarial_transport_suite",
        diagnostic_type="reflector_adversarial_transport_suite",
        module="scripts.run_reflector_adversarial_transport_suite",
        script_path="scripts/run_reflector_adversarial_transport_suite.py",
        summary_artifact="json",
        produced_artifacts=("json", "csv", "claim_matrix", "legacy_summary_json", "counterfactual_rows_csv"),
        upstream_script_ids=("counterfactual_intervention_suite", "reflector_transport_ablation"),
        claim_families=("adversarial_trap_resistance", "anchor_salience_failure_modes"),
        description="Stress-tests transport against fake frame signals and nuisance traps.",
    ),
    "reflector_semantic_gate_ablation": ReflectorScriptSpec(
        script_id="reflector_semantic_gate_ablation",
        diagnostic_type="reflector_semantic_gate_ablation",
        module="scripts.run_reflector_semantic_gate_ablation",
        script_path="scripts/run_reflector_semantic_gate_ablation.py",
        summary_artifact="json",
        produced_artifacts=("json", "csv", "claim_matrix", "legacy_summary_json", "counterfactual_rows_csv"),
        upstream_script_ids=(
            "counterfactual_intervention_suite",
            "reflector_transport_ablation",
            "reflector_adversarial_transport_suite",
        ),
        claim_families=("semantic_use_gate", "mention_vs_use_discount", "adversarial_trap_discount"),
        description="Ablates semantic-use gates that discount frame mentions without frame use.",
    ),
}


def get_reflector_script_spec(script_id: str) -> ReflectorScriptSpec:
    try:
        return REFLECTOR_SCRIPTS[str(script_id)]
    except KeyError as exc:
        raise KeyError(f"Unknown Reflector script_id: {script_id}") from exc


def iter_reflector_script_specs() -> Iterable[ReflectorScriptSpec]:
    for script_id in REFLECTOR_SCRIPT_ORDER:
        yield REFLECTOR_SCRIPTS[script_id]


def downstream_script_ids(script_id: str) -> List[str]:
    return [
        spec.script_id
        for spec in iter_reflector_script_specs()
        if str(script_id) in set(spec.upstream_script_ids)
    ]


def reflector_payload_awareness(script_id: str) -> Dict[str, object]:
    spec = get_reflector_script_spec(script_id)
    return {
        "contract_schema_version": "1.0",
        "script_id": spec.script_id,
        "diagnostic_type": spec.diagnostic_type,
        "module": spec.module,
        "script_path": str(spec.path),
        "summary_artifact": spec.summary_artifact,
        "produced_artifacts": list(spec.produced_artifacts),
        "upstream_script_ids": list(spec.upstream_script_ids),
        "downstream_script_ids": downstream_script_ids(spec.script_id),
        "claim_families": list(spec.claim_families),
        "description": spec.description,
    }


def build_reflector_orchestration_contract(
    *,
    manifest_paths: Sequence[Path | str] = (),
    metadata: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    nodes = [
        {
            "node_id": spec.script_id,
            "diagnostic_type": spec.diagnostic_type,
            "module": spec.module,
            "script_path": spec.script_path,
            "claim_families": list(spec.claim_families),
        }
        for spec in iter_reflector_script_specs()
    ]
    dependencies = {spec.script_id: list(spec.upstream_script_ids) for spec in iter_reflector_script_specs()}
    return build_orchestration_contract(
        dag_id="reflector_evidence_spine",
        nodes=nodes,
        dependencies=dependencies,
        executor="manifest",
        manifest_paths=manifest_paths,
        metadata=metadata or {},
    )
