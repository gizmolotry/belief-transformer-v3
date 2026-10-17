#!/usr/bin/env python3
"""Independent adversarial audit of the cross-instrument scrutiny battery.

This script intentionally reconstructs every requested statistic from the two
corpora, the two NLI-logit caches, the canonical hypothesis strings, and fresh
DeBERTa mean-pooled hidden states.  It does not consume either paper results
JSON file.

The CCA cross-validation path is deliberately explicit: scalers and CCA are fit
only on each training fold.  The corpus-wide anisotropy ablation is labelled
transductive because its document mean and principal directions use the full
audit corpus; that is valid for a descriptive ablation, but not leak-free for
an out-of-sample prediction claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.stats import f_oneway, kruskal, pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "microsoft/deberta-v2-xlarge-mnli"
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def article_text(row: dict) -> str:
    for field in ("text", "content", "body", "article", "raw_content"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"No article text in record with keys {sorted(row)}")


def publisher(row: dict) -> str:
    value = row.get("publisher", row.get("publication"))
    if not value:
        raise ValueError(f"No publisher in record with keys {sorted(row)}")
    return str(value)


def hypothesis_pairs(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8") as handle:
        hypotheses = json.load(handle)
    if not isinstance(hypotheses, list) or len(hypotheses) != 16:
        raise ValueError("Expected exactly 16 canonical hypotheses (8 pairs)")
    return [(hypotheses[i], hypotheses[i + 1]) for i in range(0, 16, 2)]


def nli_matrix_from_logits(la: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """Directed score: (P_entail-P_contradict)_A minus the same for B."""
    # Preserve the project's declared cache-analysis convention: fp16 cache
    # values are promoted to fp32 before softmax (not directly to fp64).
    a = torch.softmax(torch.as_tensor(la, dtype=torch.float32), dim=-1)
    b = torch.softmax(torch.as_tensor(lb, dtype=torch.float32), dim=-1)
    if a.ndim != 3 or a.shape[-2:] != (8, 3) or b.shape != a.shape:
        raise ValueError(f"Unexpected paired-logit shapes: {tuple(a.shape)}, {tuple(b.shape)}")
    return ((a[..., 2] - a[..., 0]) - (b[..., 2] - b[..., 0])).numpy()


def load_nli(path: Path, prospective: bool) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    ka, kb = ("q0_la", "q0_lb") if prospective else ("la", "lb")
    if not isinstance(obj, dict) or ka not in obj or kb not in obj:
        raise ValueError(f"{path} lacks expected keys {ka!r}, {kb!r}")
    return nli_matrix_from_logits(np.asarray(obj[ka]), np.asarray(obj[kb]))


def mean_pool_embeddings(
    texts: Sequence[str], batch_size: int, device: torch.device, local_only: bool
) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=local_only)
    classifier = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, local_files_only=local_only
    ).to(device)
    classifier.eval()
    encoder = getattr(classifier, "deberta", classifier.base_model)
    rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            rows.append(pooled.float().cpu())
            print(f"  embedded {min(start + batch_size, len(texts)):3d}/{len(texts)}", flush=True)
    del encoder, classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(rows).numpy().astype(np.float64, copy=False)


def obtain_embeddings(
    prospective_texts: Sequence[str],
    controlled_texts: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = Path(args.embedding_cache).resolve() if args.embedding_cache else None
    expected = len(prospective_texts) + len(controlled_texts) + 16
    if cache and cache.exists() and not args.recompute_embeddings:
        data = np.load(cache)
        required = ("prospective", "controlled", "hyp_a", "hyp_b")
        if not all(key in data for key in required):
            raise ValueError(f"Embedding cache {cache} lacks {required}")
        print(f"Loaded audit embedding cache: {cache}")
        return tuple(np.asarray(data[key], dtype=np.float64) for key in required)  # type: ignore[return-value]

    device = torch.device(args.device)
    flat_hyp = [text for pair in pairs for text in pair]
    all_texts = list(prospective_texts) + list(controlled_texts) + flat_hyp
    print(f"Extracting {expected} raw mean-pooled DeBERTa embeddings on {device}...")
    all_emb = mean_pool_embeddings(all_texts, args.batch_size, device, not args.allow_download)
    n_p, n_c = len(prospective_texts), len(controlled_texts)
    prospective = all_emb[:n_p]
    controlled = all_emb[n_p : n_p + n_c]
    hyp = all_emb[n_p + n_c :]
    hyp_a, hyp_b = hyp[0::2], hyp[1::2]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(
                cache,
                prospective=prospective,
                controlled=controlled,
                hyp_a=hyp_a,
                hyp_b=hyp_b,
            )
            print(f"Wrote audit embedding cache: {cache}")
        except OSError as exc:
            print(f"WARNING: optional embedding cache could not be written ({exc}); continuing")
    return prospective, controlled, hyp_a, hyp_b


def unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("Cannot L2-normalize a zero embedding")
    return x / norms


def frameaxis(docs: np.ndarray, hyp_a: np.ndarray, hyp_b: np.ndarray) -> np.ndarray:
    d, a, b = unit_rows(docs), unit_rows(hyp_a), unit_rows(hyp_b)
    return d @ a.T - d @ b.T


def standardized_pca(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    z = StandardScaler().fit_transform(x)
    # SVD is deterministic and avoids covariance eigensolver sign/order quirks.
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    scores = z @ vt.T
    eig = singular**2 / (len(z) - 1)
    evr = eig / eig.sum()
    participation_ratio = float(eig.sum() ** 2 / np.square(eig).sum())
    return scores, evr, participation_ratio


def orient_to_reference(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return scores if pearsonr(scores, reference).statistic >= 0 else -scores


def remove_top_pcs(
    docs: np.ndarray, hyp_a: np.ndarray, hyp_b: np.ndarray, n_remove: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-corpus (transductive) mean subtraction and exact top-PC removal."""
    mean = docs.mean(axis=0, keepdims=True)
    docs_c = docs - mean
    _, _, vt = np.linalg.svd(docs_c, full_matrices=False)
    directions = vt[:n_remove]

    def transform(x: np.ndarray) -> np.ndarray:
        centered = x - mean
        return centered - (centered @ directions.T) @ directions

    return transform(docs), transform(hyp_a), transform(hyp_b)


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    grand = values.mean()
    ss_total = np.square(values - grand).sum()
    ss_between = sum(
        (labels == group).sum() * (values[labels == group].mean() - grand) ** 2
        for group in np.unique(labels)
    )
    return float(ss_between / ss_total)


def publisher_subset(labels: Sequence[str], minimum: int) -> np.ndarray:
    labels_a = np.asarray(labels)
    unique, counts = np.unique(labels_a, return_counts=True)
    retained = unique[counts >= minimum]
    return np.isin(labels_a, retained)


def group_tests(values: np.ndarray, labels: Sequence[str], minimum: int) -> dict[str, float | int]:
    labels_a = np.asarray(labels)
    keep = publisher_subset(labels, minimum)
    y, groups_l = values[keep], labels_a[keep]
    groups = [y[groups_l == label] for label in np.unique(groups_l)]
    f_stat, f_p = f_oneway(*groups)
    kw_stat, kw_p = kruskal(*groups)
    return {
        "n": int(len(y)),
        "groups": int(len(groups)),
        "F": float(f_stat),
        "p": float(f_p),
        "eta2": eta_squared(y, groups_l),
        "KW_H": float(kw_stat),
        "KW_p": float(kw_p),
    }


def residualize(y: np.ndarray, predictors: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    design = np.column_stack((np.ones(len(y)), predictors))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ beta
    residual = y - fitted
    r2 = 1.0 - np.square(residual).sum() / np.square(y - y.mean()).sum()
    return residual, float(r2), beta


def sentiment_table(
    name: str,
    pc1: np.ndarray,
    texts: Sequence[str],
    pubs: Sequence[str],
    minimum_group_size: int,
) -> None:
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(text) for text in texts]
    pos = np.asarray([score["pos"] for score in scores])
    neg = np.asarray([score["neg"] for score in scores])
    compound = np.asarray([score["compound"] for score in scores])
    conditions: list[tuple[str, np.ndarray, float, np.ndarray]] = [
        ("raw", pc1, 0.0, np.asarray([]))
    ]
    for label, predictors in (
        ("residual_compound", compound[:, None]),
        ("residual_negative", neg[:, None]),
        ("residual_all", np.column_stack((pos, neg, compound))),
    ):
        residual, r2, beta = residualize(pc1, predictors)
        conditions.append((label, residual, r2, beta))
    print(f"\n{name}")
    print(
        "VADER summaries: "
        f"pos mean={pos.mean():.6f}, sd={pos.std(ddof=0):.6f}; "
        f"neg mean={neg.mean():.6f}, sd={neg.std(ddof=0):.6f}; "
        f"compound mean={compound.mean():.6f}, sd={compound.std(ddof=0):.6f}"
    )
    print("condition                  F          p       eta2       KW_H       KW_p   sentiment_R2")
    for label, values, r2, beta in conditions:
        test = group_tests(values, pubs, minimum_group_size)
        print(
            f"{label:23s} {test['F']:9.4f} {test['p']:10.6f} {test['eta2']:10.4f}"
            f" {test['KW_H']:10.4f} {test['KW_p']:10.6f} {r2:14.4f}"
            f"  (n={test['n']}, publishers={test['groups']})"
        )
        if beta.size:
            print(f"  regression beta [intercept, predictors...] = {beta.tolist()}")


def anisotropy_table(
    name: str,
    docs: np.ndarray,
    hyp_a: np.ndarray,
    hyp_b: np.ndarray,
    nli_pc1: np.ndarray,
    pubs: Sequence[str],
    minimum_group_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    variants: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        ("baseline", docs, hyp_a, hyp_b)
    ]
    for count in (1, 2):
        variants.append((f"remove_top_{count}", *remove_top_pcs(docs, hyp_a, hyp_b, count)))
    results: list[tuple[str, np.ndarray, np.ndarray]] = []
    baseline_pc1: np.ndarray | None = None
    norms = np.linalg.norm(docs, axis=1)
    print(f"\n{name}")
    print("condition          PC1%       PR   r(norm)   r(NLI-PC1)  r(baseline)       F          p      eta2")
    for label, d, a, b in variants:
        matrix = frameaxis(d, a, b)
        scores, evr, pr = standardized_pca(matrix)
        pc1 = orient_to_reference(scores[:, 0], nli_pc1)
        if baseline_pc1 is None:
            baseline_pc1 = pc1
        order_r = pearsonr(pc1, baseline_pc1).statistic
        test = group_tests(pc1, pubs, minimum_group_size)
        print(
            f"{label:16s} {100*evr[0]:7.2f} {pr:8.3f}"
            f" {pearsonr(pc1, norms).statistic:9.4f} {pearsonr(pc1, nli_pc1).statistic:12.4f}"
            f" {order_r:12.4f} {test['F']:8.3f} {test['p']:10.6f} {test['eta2']:9.4f}"
        )
        results.append((label, matrix, pc1))
    return results[0][1], results[0][2]


def validated_cca(
    x: np.ndarray, y: np.ndarray, permutations: int, repeats: int, folds: int
) -> dict[str, float]:
    # In-sample CCA/permutation: scaling on the full sample is appropriate here.
    x_full = StandardScaler().fit_transform(x)
    y_full = StandardScaler().fit_transform(y)
    cca = CCA(n_components=1, scale=False, max_iter=5000)
    x_c, y_c = cca.fit_transform(x_full, y_full)
    observed = float(pearsonr(x_c[:, 0], y_c[:, 0]).statistic)

    rng = np.random.RandomState(SEED)
    null = np.empty(permutations)
    for index in range(permutations):
        y_perm = y_full[rng.permutation(len(y_full))]
        model = CCA(n_components=1, scale=False, max_iter=5000)
        xp, yp = model.fit_transform(x_full, y_perm)
        null[index] = pearsonr(xp[:, 0], yp[:, 0]).statistic

    # Strictly leak-free CV: raw matrices enter each split; both scalers and CCA
    # are fit exclusively on train_idx.  Test data are transform-only.
    cv_r = np.empty(repeats)
    cv_fold_mean = np.empty(repeats)
    all_fold_r: list[float] = []
    for repeat in range(repeats):
        splitter = KFold(n_splits=folds, shuffle=True, random_state=SEED + repeat)
        x_oof, y_oof = np.empty(len(x)), np.empty(len(y))
        repeat_fold_r: list[float] = []
        for train_idx, test_idx in splitter.split(x):
            x_scaler = StandardScaler().fit(x[train_idx])
            y_scaler = StandardScaler().fit(y[train_idx])
            x_train, y_train = x_scaler.transform(x[train_idx]), y_scaler.transform(y[train_idx])
            model = CCA(n_components=1, scale=False, max_iter=5000).fit(x_train, y_train)
            train_x_c, train_y_c = model.transform(x_train, y_train)
            sign = 1.0 if pearsonr(train_x_c[:, 0], train_y_c[:, 0]).statistic >= 0 else -1.0
            test_x_c, test_y_c = model.transform(
                x_scaler.transform(x[test_idx]), y_scaler.transform(y[test_idx])
            )
            x_oof[test_idx] = test_x_c[:, 0]
            y_oof[test_idx] = sign * test_y_c[:, 0]
            repeat_fold_r.append(
                float(pearsonr(test_x_c[:, 0], sign * test_y_c[:, 0]).statistic)
            )
        cv_r[repeat] = pearsonr(x_oof, y_oof).statistic
        cv_fold_mean[repeat] = np.mean(repeat_fold_r)
        all_fold_r.extend(repeat_fold_r)

    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_q95": float(np.quantile(null, 0.95)),
        "perm_p_plus_one": float((np.count_nonzero(null >= observed) + 1) / (permutations + 1)),
        "cv_mean": float(cv_r.mean()),
        "cv_sd_population": float(cv_r.std(ddof=0)),
        "cv_q025": float(np.quantile(cv_r, 0.025)),
        "cv_q975": float(np.quantile(cv_r, 0.975)),
        "foldwise_mean": float(cv_fold_mean.mean()),
        "foldwise_repeat_mean_sd": float(cv_fold_mean.std(ddof=0)),
        "individual_fold_sd": float(np.std(all_fold_r, ddof=0)),
    }


def print_cca(name: str, result: dict[str, float]) -> None:
    print(f"\n{name}")
    print(f"  in-sample canonical r : {result['observed']:.6f}")
    print(
        "  permutation null      : "
        f"mean={result['null_mean']:.6f}, q95={result['null_q95']:.6f}, "
        f"plus-one p={result['perm_p_plus_one']:.6f}"
    )
    print(
        "  pooled OOF CV (legacy): "
        f"r={result['cv_mean']:.6f} +/- {result['cv_sd_population']:.6f}, "
        f"empirical 95% interval=[{result['cv_q025']:.6f}, {result['cv_q975']:.6f}]"
    )
    print(
        "  within-fold test r     : "
        f"mean={result['foldwise_mean']:.6f}, "
        f"SD of 50 repeat means={result['foldwise_repeat_mean_sd']:.6f}, "
        f"SD across 250 folds={result['individual_fold_sd']:.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--embedding-cache", default="", help="Optional .npz audit cache")
    parser.add_argument("--recompute-embeddings", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face network lookup")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--cv-repeats", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.set_printoptions(precision=6, suppress=True)
    prospective_rows = read_jsonl(ROOT / "data/prospective_confirmatory_120_corpus.jsonl")
    controlled_rows = read_jsonl(ROOT / "data/controlled_N60_corpus.jsonl")
    if len(prospective_rows) != 120 or len(controlled_rows) != 60:
        raise ValueError(f"Corpus sizes changed: {len(prospective_rows)=}, {len(controlled_rows)=}")
    prospective_texts = [article_text(row) for row in prospective_rows]
    controlled_texts = [article_text(row) for row in controlled_rows]
    prospective_pubs = [publisher(row) for row in prospective_rows]
    controlled_pubs = [publisher(row) for row in controlled_rows]
    pairs = hypothesis_pairs(ROOT / "probe_hypotheses.json")

    prospective_nli = load_nli(ROOT / "data/cache_prospective_120_q0_q1_logits.pt", True)
    controlled_nli = load_nli(
        ROOT / "data/cache_paired_logits_deberta-v2-xlarge-mnli_N60.pt", False
    )
    prospective_nli_scores, _, _ = standardized_pca(prospective_nli)
    controlled_nli_scores, _, _ = standardized_pca(controlled_nli)
    prospective_nli_pc1 = prospective_nli_scores[:, 0]
    controlled_nli_pc1 = controlled_nli_scores[:, 0]

    prospective_emb, controlled_emb, hyp_a, hyp_b = obtain_embeddings(
        prospective_texts, controlled_texts, pairs, args
    )
    expected_dim = prospective_emb.shape[1]
    if not (
        prospective_emb.shape == (120, expected_dim)
        and controlled_emb.shape == (60, expected_dim)
        and hyp_a.shape == hyp_b.shape == (8, expected_dim)
    ):
        raise ValueError("Embedding row order/count/dimension validation failed")

    print("\n=== 1. ANISOTROPY REMOVAL ===")
    prospective_fa, prospective_fa_pc1 = anisotropy_table(
        "Prospective N=120 (publisher tests retain groups with n>=2)",
        prospective_emb,
        hyp_a,
        hyp_b,
        prospective_nli_pc1,
        prospective_pubs,
        2,
    )
    controlled_fa, controlled_fa_pc1 = anisotropy_table(
        "Controlled Balanced N=60 (12 x 5)",
        controlled_emb,
        hyp_a,
        hyp_b,
        controlled_nli_pc1,
        controlled_pubs,
        1,
    )
    print(
        "\nTop-PC audit verdict: subtraction uses an orthonormal SVD basis, applies the same"
        " document-derived mean/basis to documents and hypotheses, then re-normalizes inside"
        " FrameAxis: mathematically sound. It is transductive, not prediction-leak-free, because"
        " the full corpus defines the mean/PCs. No labels or NLI outcomes enter that transform."
    )

    print("\n=== 2. VADER SENTIMENT CONTROLS ===")
    sentiment_table(
        "Controlled NLI PC1", controlled_nli_pc1, controlled_texts, controlled_pubs, 1
    )
    sentiment_table(
        "Controlled FrameAxis PC1", controlled_fa_pc1, controlled_texts, controlled_pubs, 1
    )
    sentiment_table(
        "Prospective NLI PC1 (groups with n>=2)",
        prospective_nli_pc1,
        prospective_texts,
        prospective_pubs,
        2,
    )
    sentiment_table(
        "Prospective FrameAxis PC1 (groups with n>=2)",
        prospective_fa_pc1,
        prospective_texts,
        prospective_pubs,
        2,
    )

    print("\n=== 3. CCA ===")
    print_cca(
        "Prospective N=120",
        validated_cca(
            prospective_nli, prospective_fa, args.permutations, args.cv_repeats, 5
        ),
    )
    print_cca(
        "Controlled N=60",
        validated_cca(controlled_nli, controlled_fa, args.permutations, args.cv_repeats, 5),
    )
    print(
        "\nCCA leakage audit verdict (this independent pipeline): PASS. Every CV fold fits X/Y"
        " StandardScaler objects and the CCA model on train_idx only; test_idx is used only by"
        " transform and final scoring. The legacy experiment implementation FAILS the audit's"
        " literal pipeline requirement because it creates globally standardized matrices before"
        " KFold. Its CCA(scale=True) then re-standardizes on each training fold, algebraically"
        " cancelling almost all of that outer affine transform; this explains why the legacy"
        " pooled-OOF number survives the corrected implementation, but is not a reason to retain"
        " the globally pre-standardized code. Also, pooled OOF correlation combines fold-specific"
        " canonical coordinate systems; the within-fold test-r mean is reported separately."
    )


if __name__ == "__main__":
    main()
