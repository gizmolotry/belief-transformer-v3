"""Fail-closed external validation for the real 60-article pilot."""

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats
import scipy.optimize
from sklearn.metrics import cohen_kappa_score
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REAL_EMBEDDINGS_PATH = ROOT / "data" / "real_pilot_embeddings.pt"
PILOT_MANIFEST_PATH = ROOT / "data" / "real_pilot_embeddings.manifest.json"
RESULTS_PATH = ROOT / "external_validation_results.json"
N_MANTEL_PERMUTATIONS = 1000
RNG_SEED = 1729

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _path(name):
    """Resolve legacy root-level artifacts independently of the launch cwd."""
    return ROOT / name


def _finite_correlation(function, x, y):
    if len(x) < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return None, None
    result = function(x, y)
    return float(result.statistic), float(result.pvalue)


def partial_correlation_permutation_test(x, y, covar, n_permutations=10000, rng=None):
    if len(x) < 3 or np.std(y) == 0 or np.std(x) == 0:
        return np.nan, np.nan
    rng = rng or np.random.default_rng(RNG_SEED)
    res_x = x - np.polyval(np.polyfit(covar, x, 1), covar)
    res_y = y - np.polyval(np.polyfit(covar, y, 1), covar)
    actual_corr = stats.pearsonr(res_x, res_y).statistic
    null_corrs = np.array([
        stats.pearsonr(res_x, rng.permutation(res_y)).statistic
        for _ in range(n_permutations)
    ])
    p_val = (np.sum(np.abs(null_corrs) >= abs(actual_corr)) + 1) / (n_permutations + 1)
    return float(actual_corr), float(p_val)


def test_crack_density_external_validity(system_scores, blind_scores, human_scores):
    blind_rho, blind_p = _finite_correlation(stats.spearmanr, system_scores, blind_scores)
    human_mask = np.isfinite(human_scores)
    human_rho, human_p = _finite_correlation(
        stats.spearmanr, system_scores[human_mask], human_scores[human_mask]
    ) if human_mask.sum() >= 3 else (None, None)
    return {
        "n_matched": int(len(system_scores)),
        "rho_vs_blind_llm": blind_rho,
        "p_vs_blind_llm": blind_p,
        "rho_vs_human": human_rho,
        "p_vs_human": human_p,
    }


def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


def _load_pilot_articles(n_articles=60):
    articles = []
    with (ROOT / "data" / "real_corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            articles.append(json.loads(line))
            if len(articles) == n_articles:
                break
    if len(articles) != n_articles:
        raise ValueError(f"Expected {n_articles} real pilot articles, found {len(articles)}")
    return articles


def _article_id(article):
    return str(article.get("article_id") or article.get("id") or article.get("canonical_id") or "")


def _tensor_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _git_dirty():
    return bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip())


def _script_sha256():
    return _tensor_sha256(Path(__file__).resolve())


def _load_pilot_manifest():
    if not PILOT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing required tensor-order provenance: {PILOT_MANIFEST_PATH}. "
            "It must map all 60 tensor rows to article_ids and identify manual/auto prompt artifacts."
        )
    manifest = json.loads(PILOT_MANIFEST_PATH.read_text(encoding="utf-8"))
    ids = manifest.get("article_ids")
    if not isinstance(ids, list) or len(ids) != 60 or len(set(map(str, ids))) != 60:
        raise ValueError("Pilot manifest article_ids must contain 60 unique IDs in tensor row order")
    return manifest


def validate_required_provenance():
    """Aggregate unavailable scientific inputs instead of silently substituting proxies."""
    blockers = []
    try:
        manifest = _load_pilot_manifest()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        manifest = None
        blockers.append(str(exc))
    if manifest is not None:
        for key in ("manual_embeddings", "auto_embeddings"):
            value = manifest.get(key)
            if not value or not (ROOT / value).exists():
                blockers.append(f"Pilot manifest lacks an existing {key} artifact")
    human = pd.read_csv(_path("human_tension_rating_subsample.csv"))
    if pd.to_numeric(human["human_tension_score_0_to_4"], errors="coerce").notna().sum() == 0:
        blockers.append("Human tension rating column contains zero completed ratings")
    import inspect
    from core.riemannian_geometry import SemanticWorkIntegrator
    source = inspect.getsource(SemanticWorkIntegrator.compute_work)
    if "TODO: True geodesic" in source or 'path: str = "linear"' in source:
        blockers.append(
            "core/riemannian_geometry.py has no continuous geodesic engine; "
            "SemanticWorkIntegrator is explicitly a linear-interpolation approximation"
        )
    if blockers:
        raise RuntimeError("External validation prerequisites unavailable:\n- " + "\n- ".join(blockers))
    return manifest


def run_test_a(rng):
    publishers = pd.read_csv(_path("publisher_scores.csv"))
    pub_dict = dict(zip(publishers["publisher"], publishers["allsides_score"]))
    cp_data = np.load(_path("cyclic_paths.npz"), allow_pickle=True)
    path_indices = cp_data["path_indices"]
    work_riem = np.asarray(cp_data["work_integral"], dtype=np.float64)
    work_euclid = np.array([
        np.linalg.norm(np.diff(path, axis=0), axis=1).sum() for path in cp_data["path_xyz"]
    ])
    corpus = _load_pilot_articles(max(max(map(int, path)) for path in path_indices) + 1)

    matched, dropped = [], []
    for row, path in enumerate(path_indices):
        i, j = int(path[0]), int(path[len(path) // 2])
        pub_i, pub_j = corpus[i].get("publisher"), corpus[j].get("publisher")
        if pub_i in pub_dict and pub_j in pub_dict:
            matched.append((row, abs(float(pub_dict[pub_i]) - float(pub_dict[pub_j]))))
        else:
            dropped.append((pub_i, pub_j))

    total = len(path_indices)
    coverage_pct = 100.0 * len(matched) / total if total else None
    print(f"Matched {len(matched)}/{total} publisher pairs ({coverage_pct:.1f}%). Dropped {len(dropped)} unmatched")

    matched_stances = [float(pub_dict[p]) for row, _ in matched for p in (
        corpus[int(path_indices[row][0])].get("publisher"),
        corpus[int(path_indices[row][len(path_indices[row]) // 2])].get("publisher"),
    )]
    dropped_known = [float(pub_dict[p]) for pair in dropped for p in pair if p in pub_dict]
    stance_counts = lambda values: {
        "left": sum(v < 0 for v in values), "center": sum(v == 0 for v in values), "right": sum(v > 0 for v in values)
    }
    skew = {
        "matched_endpoint_stances": stance_counts(matched_stances),
        "dropped_known_endpoint_stances": stance_counts(dropped_known),
        "dropped_unknown_endpoints": int(2 * len(dropped) - len(dropped_known)),
    }
    print("Coverage skew analysis:", json.dumps(skew, sort_keys=True))
    print("  Missing-score endpoints cannot be assigned a stance; known dropped endpoints are shown explicitly (no imputation).")

    if len(matched) < 3:
        raise AssertionError("Test A fail-closed: fewer than three publisher pairs have known stances")
    if not matched_stances or (dropped and not dropped_known):
        raise AssertionError("Test A fail-closed: dropped-publisher stance skew is not assessable")
    rows = np.array([row for row, _ in matched])
    bias_dist = np.array([distance for _, distance in matched])
    corr, pval = partial_correlation_permutation_test(
        work_riem[rows], bias_dist, work_euclid[rows], rng=rng
    )
    print(f"Partial Correlation (Linear): {corr:.3f} (p={pval:.4f})")

    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    delta_work = work_riem[rows] - work_euclid[rows]
    popt, pcov = scipy.optimize.curve_fit(exp_func, bias_dist, delta_work, p0=[1, 1, 0], maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    print(f"Exponential fit b={popt[1]:.3f} +/- {perr[1] * 1.96:.3f} (95% CI)")
    return {"coverage_pct": coverage_pct, "matched_pairs": len(matched), "dropped_pairs": len(dropped), "coverage_skew": skew,
            "partial_correlation": corr, "partial_p_value": pval}


def _track3_density(embeds):
    """Re-run production Track 3 Dirichlet fusion and conformal density."""
    from core.dirichlet_fusion import DirichletFusion, DirichletFusionConfig
    from core.hadamard_fusion import ConformalMetric
    config = DirichletFusionConfig(n_bots=8, hidden_dim=1536, rks_dim=2048)
    fusion = DirichletFusion(config)
    with torch.no_grad():
        fused = fusion(embeds, compute_curvature=False)
    rho = ConformalMetric().compute_density(fused["fused_std"])
    if tuple(rho.shape) != (60,):
        raise AssertionError(f"Track 3 rho must be [60], got {tuple(rho.shape)}")
    return rho.detach().cpu().numpy()


def run_test_b(embeds, manifest):
    blind = pd.read_csv(_path("gemini_blind_ratings.csv"))
    human = pd.read_csv(_path("human_tension_rating_subsample.csv"))
    merged = blind.merge(human[["article_id", "human_tension_score_0_to_4"]], on="article_id", how="inner")
    rated = merged.dropna(subset=["gemini_score", "human_tension_score_0_to_4"])
    if len(rated):
        kappa = cohen_kappa_score(np.round(rated["gemini_score"]), np.round(rated["human_tension_score_0_to_4"]))
        inter_rho, inter_p = _finite_correlation(stats.spearmanr, rated["gemini_score"].to_numpy(), rated["human_tension_score_0_to_4"].to_numpy())
        print(f"Inter-rater reliability (n={len(rated)}): kappa={kappa:.3f}, Spearman={inter_rho:.3f}")
    else:
        kappa, inter_rho = None, None
        print("Human rating column contains no completed ratings; human correlation is fail-closed as null.")

    rho = _track3_density(embeds)
    pilot_ids = list(map(str, manifest["article_ids"]))
    density_df = pd.DataFrame({"article_id": pilot_ids, "rho": rho})
    matched = density_df.merge(blind, on="article_id").merge(
        human[["article_id", "human_tension_score_0_to_4"]], on="article_id"
    )
    if matched.empty:
        raise AssertionError("No article IDs match Track 3 density to blind/human ratings")
    result = test_crack_density_external_validity(
        matched["rho"].to_numpy(), matched["gemini_score"].to_numpy(), matched["human_tension_score_0_to_4"].to_numpy()
    )
    print(f"Track 3 per-article rho correlations (matched n={result['n_matched']}): ")
    print(f"  blind LLM={result['rho_vs_blind_llm']} (p={result['p_vs_blind_llm']})")
    print(f"  human={result['rho_vs_human']} (p={result['p_vs_human']})")
    return {"inter_rater_kappa": None if kappa is None else float(kappa), "inter_rater_rho": inter_rho, **result}


def run_test_c(manifest, rng):
    manual_payload = torch.load(ROOT / manifest["manual_embeddings"], map_location="cpu", weights_only=False)
    auto_payload = torch.load(ROOT / manifest["auto_embeddings"], map_location="cpu", weights_only=False)
    manual_embeds = torch.as_tensor(manual_payload["embeddings"]).numpy()
    auto_embeds = torch.as_tensor(auto_payload["embeddings"]).numpy()
    manual_ids = list(map(str, manual_payload["article_ids"]))
    auto_ids = list(map(str, auto_payload["article_ids"]))
    if manual_ids != auto_ids or manual_ids != list(map(str, manifest["article_ids"])):
        raise AssertionError("Manual/auto article IDs are not identical and in proven pilot row order")
    if manual_embeds.shape != auto_embeds.shape or manual_embeds.shape[0] != 60:
        raise AssertionError("Manual/auto embedding arrays must be shape-matched with 60 articles")
    sims = np.array([cosine_similarity(manual_embeds[i], auto_embeds[i]) for i in range(len(manual_embeds))])
    print(f"Same-article manual/auto cosine (n={len(sims)}): mean={sims.mean():.3f}, median={np.median(sims):.3f}")
    l2_actual = float(np.mean(np.linalg.norm(manual_embeds - auto_embeds, axis=1)))
    null = np.array([
        np.mean(np.linalg.norm(manual_embeds - auto_embeds[rng.permutation(len(auto_embeds))], axis=1))
        for _ in range(1000)
    ])
    p_value = float((np.sum(null <= l2_actual) + 1) / (len(null) + 1))
    print(f"Matched L2={l2_actual:.3f}; permutation p={p_value:.4f}")
    return {"n_articles": len(sims), "article_ids_sha256": hashlib.sha256("\n".join(manual_ids).encode()).hexdigest(),
            "mean_cosine": float(sims.mean()), "median_cosine": float(np.median(sims)),
            "matched_l2": l2_actual, "permutation_p": p_value}


def mantel_test(matrix_a, matrix_b, n_permutations=1000, rng=None):
    """Mantel correlations on upper triangles with joint row/column permutations."""
    rng = rng or np.random.default_rng(RNG_SEED)
    if matrix_a.shape != matrix_b.shape or matrix_a.ndim != 2:
        raise ValueError("Mantel matrices must be square and shape-matched")
    upper = np.triu_indices(matrix_a.shape[0], k=1)
    a = matrix_a[upper]
    b = matrix_b[upper]
    pearson = float(stats.pearsonr(a, b).statistic)
    spearman = float(stats.spearmanr(a, b).statistic)
    null = np.empty(n_permutations)
    for index in range(n_permutations):
        permutation = rng.permutation(matrix_b.shape[0])
        permuted = matrix_b[np.ix_(permutation, permutation)]
        null[index] = stats.pearsonr(a, permuted[upper]).statistic
    p_value = float((np.sum(np.abs(null) >= abs(pearson)) + 1) / (n_permutations + 1))
    return pearson, spearman, p_value


def run_test_d():
    raise NotImplementedError(
        "Test D requires a continuous geodesic engine in core/riemannian_geometry.py, but that module's "
        "SemanticWorkIntegrator explicitly uses linear interpolation and contains a TODO for true geodesics. "
        "core/fmm_geodesic.py exists, but substituting a different module would violate the requested contract."
    )


def main():
    print("==================================================\n Running External Validation Suite\n==================================================")
    if not REAL_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Required real embeddings file not found: {REAL_EMBEDDINGS_PATH}; mock fallback prohibited")
    embeds = torch.load(REAL_EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
    if tuple(embeds.shape) != (60, 8, 1536):
        raise AssertionError(f"Expected real pilot tensor [60,8,1536], got {tuple(embeds.shape)}")
    manifest = validate_required_provenance()
    rng = np.random.default_rng(RNG_SEED)

    print("\n--- Test A: Riemannian Work vs External Bias ---")
    test_a = run_test_a(rng)
    print("\n--- Test B: Dynamic Crack Density vs Blind Ratings ---")
    test_b = run_test_b(embeds, manifest)
    print("\n--- Test C: Auto-Observer Falsification ---")
    test_c = run_test_c(manifest, rng)
    print("\n--- Test D: Paper Geometry vs Production Hadamard Fidelity ---")
    test_d = run_test_d()

    coverage_pct = test_a["coverage_pct"]
    mantel_p_value = test_d["mantel_p_value"]
    reduced_riemannian_separates_from_euclidean = test_d["reduced_riemannian_separates_from_euclidean"]
    assert coverage_pct is not None
    assert mantel_p_value is not None
    assert reduced_riemannian_separates_from_euclidean

    output = {
        "manifest": {"schema_version": "1.0", "git_commit": _git_commit(), "tensor_sha256": _tensor_sha256(REAL_EMBEDDINGS_PATH),
                     "script_sha256": _script_sha256(), "git_dirty": _git_dirty()},
        "test_a": test_a, "test_b": test_b, "test_c": test_c, "test_d": test_d,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nStamped validation output: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
