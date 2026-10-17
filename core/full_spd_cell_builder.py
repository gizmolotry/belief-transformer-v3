"""Fail-closed construction of cached full-SPD experiment cells."""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from core import full_spd_experiment as experiment
from core import full_spd_geometry as geometry

BANKS = ("Q0", "Q1", "Q2")
SEEDS = (42, 420, 4200)
ARMS = ("identity", "oof_conformal", "scalar_trace", "diagonal", "full", "article_shuffled", "observer_shuffled", "spectrum_matched_random")
GATE_ARMS = ("identity", "oof_conformal", "scalar_trace", "diagonal", "full")

class CellBuilderError(RuntimeError): pass

@dataclass(frozen=True)
class BankArrays:
    name: str; prompt_hash: str; ordered_ids: tuple[str, ...]; publisher_by_id: Mapping[str, str]
    class_names: tuple[str, str, str]; x: np.ndarray; t0: np.ndarray; artifact_hashes: Mapping[str, str]
    manifest_path: str; manifest_sha256: str

Trainer = Callable[..., Any]

def _file_hash(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): d.update(b)
    return d.hexdigest()

def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def _artifact(entry: Mapping[str, Any], name: str) -> tuple[Path, str]:
    artifacts = entry.get("artifacts")
    record = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str): raise CellBuilderError(f"bank artifact {name} is missing")
    path = Path(record["path"]).resolve()
    if not path.is_file(): raise CellBuilderError(f"missing {name} artifact")
    expected = record.get("raw_sha256", record.get("sha256")); actual = _file_hash(path)
    if not isinstance(expected, str) or actual != expected: raise CellBuilderError(f"bank artifact {name} hash mismatch")
    return path, actual

def _identity(entry: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, str]]:
    cohort = entry.get("cohort")
    rows = cohort.get("article_order") if isinstance(cohort, Mapping) else None
    if not isinstance(rows, list): raise CellBuilderError("bank cohort must contain article_order")
    ids, pubs = [], {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str) or not isinstance(row.get("publisher"), str): raise CellBuilderError("invalid cohort row")
        ids.append(row["id"]); pubs[row["id"]] = row["publisher"]
    if not ids or len(ids) != len(set(ids)): raise CellBuilderError("cohort IDs must be unique")
    return tuple(ids), pubs

def load_bank_manifest(path: str | Path, *, synthetic: bool = False) -> Mapping[str, BankArrays]:
    manifest_path = Path(path).resolve()
    try: manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise CellBuilderError("invalid bank manifest") from exc
    raw = manifest.get("banks") if isinstance(manifest, Mapping) else None
    if not isinstance(raw, Mapping) or set(raw) != set(BANKS): raise CellBuilderError("bank manifest must contain exactly Q0/Q1/Q2")
    found, x_hashes, foundation = {}, set(), None
    for name in BANKS:
        entry = raw[name]
        if not isinstance(entry, Mapping): raise CellBuilderError("invalid bank entry")
        prompt = entry.get("q_bank_hash", entry.get("prompt_hash"))
        if not isinstance(prompt, str) or len(prompt) != 64 or entry.get("prompt_hash") not in (None, prompt): raise CellBuilderError("bank prompt provenance mismatch")
        ids, pubs = _identity(entry)
        if foundation is None: foundation = (ids, pubs)
        elif foundation != (ids, pubs): raise CellBuilderError("banks must bind identical ordered IDs and publishers")
        model = entry.get("model"); labels = model.get("class_names") if isinstance(model, Mapping) else None
        names = tuple(str(x).lower() for x in labels) if isinstance(labels, list) else ()
        if len(names) != 3 or set(names) != {"entail", "neutral", "contradict"}: raise CellBuilderError("canonical NLI class order is required")
        xp, xh = _artifact(entry, "X"); tp, th = _artifact(entry, "T0")
        x, t0 = np.load(xp, allow_pickle=False), np.load(tp, allow_pickle=False)
        if not synthetic and (tuple(x.shape), tuple(t0.shape)) != ((320, 2048), (320, 8, 3)): raise CellBuilderError("real arrays require X=(320,2048), T0=(320,8,3)")
        if synthetic and (x.ndim != 2 or t0.ndim != 3 or tuple(t0.shape[1:]) != (8, 3)): raise CellBuilderError("synthetic arrays require X=(N,D), T0=(N,8,3)")
        if len(ids) != x.shape[0] or len(ids) != t0.shape[0] or not np.isfinite(x).all() or not np.isfinite(t0).all(): raise CellBuilderError("array/cohort mismatch")
        found[name] = BankArrays(
            name, prompt, ids, pubs, names,
            np.asarray(x, dtype=np.float64), np.asarray(t0, dtype=np.float64),
            {"X": xh, "T0": th}, str(manifest_path), _file_hash(manifest_path),
        ); x_hashes.add(xh)
    if len(x_hashes) != 3: raise CellBuilderError("Q0/Q1/Q2 must not share an X artifact hash")
    return found

def _synthetic_trainer(train_x: np.ndarray, train_y: np.ndarray, held_x: np.ndarray, **kwargs: Any) -> Any:
    direction = train_x.T @ (train_y - train_y.mean()); norm = np.linalg.norm(direction)
    direction = direction / norm if norm else np.ones(train_x.shape[1]) / math.sqrt(train_x.shape[1])
    return type("Fit", (), {"predictions": held_x @ direction, "train_predictions": train_x @ direction, "node_gradients": np.tile(direction, (len(held_x), 1)), "provenance": {"seed": kwargs["seed"]}})()

def _metrics(e: np.ndarray, train_y: np.ndarray, train_p: np.ndarray, hold_y: np.ndarray, hold_p: np.ndarray, w: np.ndarray, identity: np.ndarray) -> tuple[dict[str, float], float]:
    g = geometry.potential_generalization_diagnostics(train_y, train_p, hold_y, hold_p)
    tolerance = geometry.near_tie_tolerance(e)
    return ({"dispersion": geometry.eccentricity_dispersion_cv(e), "tie_fraction": geometry.near_tie_fraction(e, tolerance=tolerance), "generalization": g.g_hold, "generalization_gap": g.g_gap, "amplification": geometry.edge_amplification_q95(w, identity)}, tolerance)

def _sha256_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CellBuilderError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_training_result(
    result: Any, *, train_ids: Sequence[str], held_ids: Sequence[str], targets: np.ndarray,
    q_bank_hash: str, seed: int, git_commit: str, target_source_kind: str,
) -> dict[str, Any]:
    provenance = getattr(result, "provenance", None)
    if not isinstance(provenance, Mapping):
        raise CellBuilderError("potential result lacks provenance")
    try:
        validated = geometry.validate_empirical_provenance(provenance, expected={
            "target_source_kind": target_source_kind,
            "train_id_hash": geometry.ordered_ids_hash(train_ids),
            "heldout_id_hash": geometry.ordered_ids_hash(held_ids),
            "target_hash": geometry.canonical_array_sha256(targets),
            "q_bank_hash": q_bank_hash,
            "seed": int(seed),
            "git_commit": git_commit,
            "node_gradient_hash": geometry.canonical_array_sha256(np.asarray(result.node_gradients, dtype=np.float64)),
        })
    except (geometry.ProvenanceError, AttributeError, TypeError, ValueError) as exc:
        raise CellBuilderError(f"invalid potential-training provenance: {exc}") from exc
    _sha256_string(validated.get("model_state_hash"), "model_state_hash")
    _sha256_string(validated.get("loss_trace_hash"), "loss_trace_hash")
    training = validated.get("training")
    hashes = training.get("prediction_hashes") if isinstance(training, Mapping) else None
    expected_predictions = {
        "train": getattr(result, "train_predictions", None),
        "train_z": getattr(result, "train_z_predictions", None),
        "heldout": getattr(result, "predictions", None),
        "heldout_z": getattr(result, "z_predictions", None),
    }
    if not isinstance(hashes, Mapping) or set(hashes) != set(expected_predictions):
        raise CellBuilderError("potential provenance lacks the four prediction hashes")
    for name, values in expected_predictions.items():
        if values is None or hashes.get(name) != geometry.canonical_array_sha256(np.asarray(values, dtype=np.float64)):
            raise CellBuilderError(f"potential {name} prediction hash mismatch")
    return dict(validated)


def _provenance(
    bank: BankArrays, support: geometry.SupportGraph, weights: Mapping[str, np.ndarray],
    distances: Mapping[str, np.ndarray], train_ids: Sequence[str], held_ids: Sequence[str],
    train_y: np.ndarray, seed: int, git_commit: str, *, target_source_kind: str,
    normal_provenance: Mapping[str, Any], shuffled_provenance: Mapping[str, Any],
    observer_shuffled_seed: int, near_tie_tolerances: Mapping[str, float],
    real_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {"schema": "full_spd_cached_cell/v1", "target_source_kind": target_source_kind, "q_bank_hash": bank.prompt_hash, "git_commit": git_commit, "seed": seed, "observer_shuffled_seed": observer_shuffled_seed, "ordered_ids_hash": geometry.ordered_ids_hash(held_ids), "train_id_hash": geometry.ordered_ids_hash(train_ids), "heldout_id_hash": geometry.ordered_ids_hash(held_ids), "target_hash": geometry.canonical_array_sha256(train_y), "x_hash": bank.artifact_hashes["X"], "t0_hash": bank.artifact_hashes["T0"], "bank_manifest": {"path": bank.manifest_path, "sha256": bank.manifest_sha256}, "support_hash": support.support_hash, "weight_hashes": {a: geometry.canonical_array_sha256(v) for a,v in weights.items()}, "distance_hashes": {a: geometry.canonical_array_sha256(v) for a,v in distances.items()}, "near_tie_tolerance_rule": geometry.NEAR_TIE_TOLERANCE_RULE, "near_tie_tolerances": dict(near_tie_tolerances), "potential_training": {"base": dict(normal_provenance), "observer_shuffled": dict(shuffled_provenance)}}
    if real_binding is not None:
        payload["authorization_binding"] = dict(real_binding)
    return {**payload, "provenance_hash": _json_hash(payload)}

def validate_cell_provenance(cell: Mapping[str, Any]) -> bool:
    provenance = cell.get("provenance")
    if not isinstance(provenance, Mapping): raise CellBuilderError("cell lacks provenance")
    copied = dict(provenance); recorded = copied.pop("provenance_hash", None)
    if not isinstance(recorded, str) or recorded != _json_hash(copied): raise CellBuilderError("cell provenance hash mismatch")
    mode = cell.get("mode")
    expected_kind = "synthetic" if mode == "synthetic" else "observer_derived" if mode == "real" else None
    if expected_kind is None or provenance.get("target_source_kind") != expected_kind:
        raise CellBuilderError("cell target_source_kind does not match mode")
    for label in ("ordered_ids_hash", "train_id_hash", "heldout_id_hash", "target_hash", "x_hash", "t0_hash", "support_hash"):
        _sha256_string(provenance.get(label), label)
    for label in ("weight_hashes", "distance_hashes"):
        hashes = provenance.get(label)
        if not isinstance(hashes, Mapping) or set(hashes) != set(ARMS):
            raise CellBuilderError(f"cell provenance {label} must cover exactly eight arms")
        for arm, value in hashes.items():
            _sha256_string(value, f"{label}.{arm}")
    tolerances = cell.get("near_tie_tolerances")
    if not isinstance(tolerances, Mapping) or set(tolerances) != set(GATE_ARMS):
        raise CellBuilderError("cell must record exactly five near-tie tolerances")
    if provenance.get("near_tie_tolerances") != tolerances:
        raise CellBuilderError("near-tie tolerance provenance mismatch")
    if provenance.get("near_tie_tolerance_rule") != geometry.NEAR_TIE_TOLERANCE_RULE:
        raise CellBuilderError("near-tie tolerance rule provenance mismatch")
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0 for value in tolerances.values()):
        raise CellBuilderError("near-tie tolerances must be positive finite numbers")
    eccentricities = cell.get("eccentricities")
    if not isinstance(eccentricities, Mapping) or set(eccentricities) != set(ARMS):
        raise CellBuilderError("cell must contain exactly eight eccentricity arms")
    for arm in GATE_ARMS:
        if float(tolerances[arm]) != geometry.near_tie_tolerance(np.asarray(eccentricities[arm], dtype=np.float64)):
            raise CellBuilderError(f"near-tie tolerance mismatch for {arm}")
    manifest = provenance.get("bank_manifest")
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("path"), str):
        raise CellBuilderError("cell lacks bank-manifest provenance")
    _sha256_string(manifest.get("sha256"), "bank manifest hash")
    potential = provenance.get("potential_training")
    if not isinstance(potential, Mapping) or set(potential) != {"base", "observer_shuffled"}:
        raise CellBuilderError("cell lacks both potential-training provenances")
    if mode == "real":
        binding = provenance.get("authorization_binding")
        if not isinstance(binding, Mapping) or set(binding) != {"authorization", "preflight", "bank_manifest"}:
            raise CellBuilderError("real cell lacks authorization/preflight/manifest binding")
        if binding.get("bank_manifest") != manifest:
            raise CellBuilderError("real cell bank-manifest binding mismatch")
        for label in ("authorization", "preflight", "bank_manifest"):
            record = binding[label]
            if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
                raise CellBuilderError(f"invalid {label} binding")
            _sha256_string(record.get("sha256"), f"{label} hash")
        for label, expected_seed in (("base", provenance.get("seed")), ("observer_shuffled", provenance.get("observer_shuffled_seed"))):
            try:
                geometry.validate_empirical_provenance(potential[label], expected={
                    "target_source_kind": "observer_derived",
                    "q_bank_hash": provenance.get("q_bank_hash"),
                    "git_commit": provenance.get("git_commit"),
                    "seed": expected_seed,
                })
            except geometry.ProvenanceError as exc:
                raise CellBuilderError(f"invalid embedded {label} potential provenance: {exc}") from exc
            _sha256_string(potential[label].get("model_state_hash"), f"{label} model-state hash")
            _sha256_string(potential[label].get("loss_trace_hash"), f"{label} loss-trace hash")
            training = potential[label].get("training")
            hashes = training.get("prediction_hashes") if isinstance(training, Mapping) else None
            if not isinstance(hashes, Mapping) or set(hashes) != {"train", "train_z", "heldout", "heldout_z"}:
                raise CellBuilderError(f"invalid embedded {label} prediction hashes")
            for name, value in hashes.items():
                _sha256_string(value, f"{label} {name} prediction hash")
    return True

def build_cached_cells(banks: Mapping[str, BankArrays], *, synthetic: bool=False, trainer: Trainer|None=None, device: str="cpu", k: int=10, git_commit: str="synthetic", real_binding: Mapping[str, Any] | None=None) -> list[dict[str, Any]]:
    if set(banks) != set(BANKS): raise CellBuilderError("exactly Q0/Q1/Q2 banks required")
    if not synthetic and k != 10: raise CellBuilderError("real construction requires k=10")
    if not synthetic and real_binding is None: raise CellBuilderError("real construction requires authorization binding")
    first = banks["Q0"]
    if not synthetic and (len(first.ordered_ids) != 320 or first.x.shape != (320,2048)): raise CellBuilderError("real construction requires 320x2048")
    if any(b.ordered_ids != first.ordered_ids or dict(b.publisher_by_id) != dict(first.publisher_by_id) for b in banks.values()): raise CellBuilderError("bank identity mismatch")
    folds = experiment.deterministic_publisher_folds(first.ordered_ids, first.publisher_by_id); fit = trainer or (_synthetic_trainer if synthetic else geometry.train_oof_potential)
    output=[]
    for bn in BANKS:
      bank=banks[bn]; targets=geometry.observer_target_from_nli_logits(bank.t0, bank.class_names)
      for fold in range(5):
        held_ids=folds.heldout_ids[fold]; held_set=set(held_ids); held_mask=np.array([x in held_set for x in bank.ordered_ids]); train_mask=~held_mask
        train_ids=tuple(np.asarray(bank.ordered_ids)[train_mask]); train_x,held_x=bank.x[train_mask],bank.x[held_mask]; train_y,held_y=targets[train_mask],targets[held_mask]
        if set(train_ids)&held_set: raise CellBuilderError("train/held-out leakage")
        support=geometry.build_euclidean_union_knn_support(held_x, held_ids, k=(min(k,len(held_ids)-1) if synthetic else k))
        train_groups=[bank.publisher_by_id[x] for x in train_ids]; held_groups=[bank.publisher_by_id[x] for x in held_ids]
        for seed in SEEDS:
          target_kind="synthetic" if synthetic else "observer_derived"
          normal=fit(train_x,train_y,held_x,train_ids=train_ids,heldout_ids=held_ids,q_bank_hash=bank.prompt_hash,seed=seed,git_commit=git_commit,target_source_kind=target_kind,device=device)
          shuffled_seed=geometry.derive_deterministic_seed(seed,"observer_shuffled",bn,fold)
          shuffled_target,_=geometry.observer_shuffled_target_helper(train_y,seed=shuffled_seed,groups=train_groups)
          shuffled=fit(train_x,shuffled_target,held_x,train_ids=train_ids,heldout_ids=held_ids,q_bank_hash=bank.prompt_hash,seed=shuffled_seed,git_commit=git_commit,target_source_kind=target_kind,device=device)
          if synthetic:
            normal_provenance=dict(getattr(normal,"provenance",{})); shuffled_provenance=dict(getattr(shuffled,"provenance",{}))
          else:
            normal_provenance=_validate_training_result(normal,train_ids=train_ids,held_ids=held_ids,targets=train_y,q_bank_hash=bank.prompt_hash,seed=seed,git_commit=git_commit,target_source_kind=target_kind)
            shuffled_provenance=_validate_training_result(shuffled,train_ids=train_ids,held_ids=held_ids,targets=shuffled_target,q_bank_hash=bank.prompt_hash,seed=shuffled_seed,git_commit=git_commit,target_source_kind=target_kind)
          grad=np.asarray(normal.node_gradients,float); identity=geometry.edge_costs(held_x,grad,support,arm="identity")
          article=geometry.article_shuffle_gradients_within_group(grad,held_groups,seed=geometry.derive_deterministic_seed(seed,"article_shuffled",bn,fold)); spectrum=geometry.spectrum_matched_random_rank_one_directions(grad,seed=geometry.derive_deterministic_seed(seed,"spectrum_matched_random",bn,fold))
          weights={"identity":identity,"oof_conformal":geometry.edge_costs(held_x,grad,support,arm="oof_conformal",yhat=normal.predictions),"scalar_trace":geometry.edge_costs(held_x,grad,support,arm="scalar_trace"),"diagonal":geometry.edge_costs(held_x,grad,support,arm="diagonal"),"full":geometry.edge_costs(held_x,grad,support,arm="full"),"article_shuffled":geometry.edge_costs(held_x,article,support,arm="full"),"observer_shuffled":geometry.edge_costs(held_x,np.asarray(shuffled.node_gradients,float),support,arm="full"),"spectrum_matched_random":geometry.edge_costs(held_x,spectrum,support,arm="full")}
          distances={a:geometry.all_pairs_dijkstra(support,w) for a,w in weights.items()}; ecc={a:geometry.article_eccentricity(d) for a,d in distances.items()}
          metric_pairs={a:_metrics(ecc[a],train_y,np.asarray(normal.train_predictions),held_y,np.asarray(normal.predictions),weights[a],identity) for a in GATE_ARMS}
          metrics={a:pair[0] for a,pair in metric_pairs.items()}; tolerances={a:pair[1] for a,pair in metric_pairs.items()}
          cell={"mode":"synthetic" if synthetic else "real","bank":bn,"fold":fold,"seed":seed,"ordered_ids":list(first.ordered_ids),"publisher_by_id":dict(first.publisher_by_id),"fold_for_publisher":dict(folds.fold_for_publisher),"eccentricities":{a:v.tolist() for a,v in ecc.items()},"gate_metrics":metrics,"near_tie_tolerances":tolerances,"connected":{a:True for a in GATE_ARMS},"construction_valid":True,"corpus_control_passed":False}
          cell["provenance"]=_provenance(bank,support,weights,distances,train_ids,held_ids,train_y,seed,git_commit,target_source_kind=target_kind,normal_provenance=normal_provenance,shuffled_provenance=shuffled_provenance,observer_shuffled_seed=shuffled_seed,near_tie_tolerances=tolerances,real_binding=real_binding)
          cell["provenance_valid"]=validate_cell_provenance(cell); output.append(cell)
    if len(output)!=45: raise CellBuilderError("incomplete bank/fold/seed grid")
    return output

def build_from_manifest(manifest_path: str|Path, *, synthetic: bool=False, trainer: Trainer|None=None, device: str="cpu", k: int=10, git_commit: str="synthetic", real_binding: Mapping[str, Any] | None=None) -> list[dict[str,Any]]:
    return build_cached_cells(load_bank_manifest(manifest_path,synthetic=synthetic),synthetic=synthetic,trainer=trainer,device=device,k=k,git_commit=git_commit,real_binding=real_binding)
