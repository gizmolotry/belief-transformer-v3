"""Experiment A: direct tests of framing-space isomorphism."""
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import subprocess

import torch

N_FRAMINGS, HIDDEN_DIM, SEED = 8, 1536, 42
RBF_SIGMA = 1.0
TRAIN_FRACTION = 0.8
ROOT = Path(__file__).resolve().parents[1]
REAL_EMBEDDINGS_PATH = ROOT / "data" / "real_pilot_embeddings.pt"
MANIFEST_PATH = ROOT / "results" / "exp_A_manifest.json"

def load_embeddings(path=REAL_EMBEDDINGS_PATH):
    """Load and validate the required embeddings, failing closed on every error."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required real embeddings file not found: {path}\n"
            "FAIL-CLOSED POLICY: Required production data is unavailable. "
            "Please run the embedding extraction pipeline on your corpus to produce real embeddings."
        )
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Could not load real embeddings tensor from {path}: {exc}") from exc
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, got {type(x).__name__}")
    if x.ndim != 3 or tuple(x.shape[1:]) != (N_FRAMINGS, HIDDEN_DIM):
        raise ValueError(
            f"Expected tensor shaped (N, {N_FRAMINGS}, {HIDDEN_DIM}), got {tuple(x.shape)}"
        )
    if x.shape[0] < 2:
        raise ValueError(f"Expected at least 2 articles, got {x.shape[0]}")
    if not x.is_floating_point():
        raise TypeError(f"Expected floating-point embeddings, got dtype {x.dtype}")
    if not torch.isfinite(x).all():
        raise ValueError("Embeddings contain non-finite values")
    return x.float()


def sha256_file(path):
    """Return the full SHA-256 digest of an input artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    """Return the exact repository commit used for this run."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_manifest(config, results, path=MANIFEST_PATH):
    """Stamp a successful experiment output with reproducibility metadata."""
    path = Path(path)
    payload = {
        "provenance": {
            "git_commit": git_commit(),
            "embeddings_sha256": sha256_file(REAL_EMBEDDINGS_PATH),
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "config": config,
        },
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def compute_cosine_gram(x):
    x = torch.nn.functional.normalize(x, dim=-1, eps=1e-12)
    return x @ x.T

def compute_rbf_gram(x, sigma=1.0):
    if sigma <= 0: raise ValueError("sigma must be positive")
    return torch.exp(-torch.cdist(x, x).square() / (2 * sigma**2))


def center_gram(k):
    return k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()

def centered_kernel_alignment(k1, k2):
    a, b = center_gram(k1), center_gram(k2)
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float("nan") if denominator <= torch.finfo(a.dtype).eps else ((a * b).sum() / denominator).item()


def qr_factors(train_x):
    """Cache an implicit *complete* QR basis and its train-sized nonzero core.

    ``geqrf`` stores all Householder reflectors needed to apply the full 1536-D
    orthogonal Q, but avoids materializing a 1536x1536 matrix.
    """
    reflectors, tau = torch.geqrf(train_x.T)
    core = torch.triu(reflectors[: train_x.shape[0], :])
    return reflectors, tau, core


def apply_complete_q(rows, factors, transpose=False):
    """Apply the complete Q (or Q.T) represented by QR Householders to rows."""
    reflectors, tau, _ = factors
    return torch.ormqr(reflectors, tau, rows, left=False, transpose=transpose)


def procrustes_predict(x_factors, y_factors, test_x, shuffled_core_y=None):
    """Apply a genuine full-dimensional orthogonal Procrustes completion.

    In complete QR bases, A.T B has an active block sized by the training set.
    If its polar factor is S=U V.T, use block_diag(S, I), then transform between
    the two complete bases. Every factor is orthogonal, hence R.T R=I exactly
    (up to floating-point error). Householder application keeps computation small.
    """
    rx, ry = x_factors[2], y_factors[2] if shuffled_core_y is None else shuffled_core_y
    u, _, vh = torch.linalg.svd(rx @ ry.T, full_matrices=False)
    coordinates = apply_complete_q(test_x, x_factors, transpose=False)
    coordinates[:, : rx.shape[0]] = coordinates[:, : rx.shape[0]] @ (u @ vh)
    return apply_complete_q(coordinates, y_factors, transpose=True)


def main():
    print("Running Experiment A: Original Premise Verification")
    x = load_embeddings()
    N_ARTICLES = x.shape[0]
    config = {
        "n_articles": N_ARTICLES,
        "n_framings": N_FRAMINGS,
        "hidden_dim": HIDDEN_DIM,
        "seed": SEED,
        "rbf_sigma": RBF_SIGMA,
        "train_fraction": TRAIN_FRACTION,
    }
    cosine = [compute_cosine_gram(x[:, f]) for f in range(N_FRAMINGS)]
    rbf = [compute_rbf_gram(x[:, f], RBF_SIGMA) for f in range(N_FRAMINGS)]
    pairs = list(combinations(range(N_FRAMINGS), 2))
    cosine_cka = [centered_kernel_alignment(cosine[i], cosine[j]) for i, j in pairs]
    rbf_cka = [centered_kernel_alignment(rbf[i], rbf[j]) for i, j in pairs]
    n_pairs = len(pairs)
    mean_cosine_cka = sum(cosine_cka) / n_pairs
    mean_rbf_cka = sum(rbf_cka) / n_pairs
    print(
        f"Computed {N_FRAMINGS} cosine and {N_FRAMINGS} exact RBF Gram matrices, "
        f"each shaped [{N_ARTICLES}, {N_ARTICLES}]."
    )
    print(f"Average cosine CKA over {n_pairs} pairs: {mean_cosine_cka:.6f}")
    print(f"Average RBF CKA over {n_pairs} pairs:    {mean_rbf_cka:.6f}")

    # Reuse one seeded article split across pairs so pair results are comparable.
    split = torch.randperm(N_ARTICLES, generator=torch.Generator().manual_seed(SEED))
    train_n = int(TRAIN_FRACTION * N_ARTICLES)
    train_idx, test_idx = split[:train_n], split[train_n:]
    # Factor each framing only once. The shuffled baselines alter the small
    # row-space core, not the complete bases/spans.
    factors = [qr_factors(x[train_idx, frame]) for frame in range(N_FRAMINGS)]
    aligned, shuffled = [], []
    for pair_number, (i, j) in enumerate(pairs):
        train_x, train_y = x[train_idx, i], x[train_idx, j]
        test_x, test_y = x[test_idx, i], x[test_idx, j]
        aligned.append(torch.linalg.matrix_norm(procrustes_predict(factors[i], factors[j], test_x)-test_y).item())
        g = torch.Generator().manual_seed(SEED + pair_number + 1)
        shuffle = torch.randperm(train_n, generator=g)
        # B_shuffled.T = Q_B (R_B P.T), so only its core changes.
        shuffled_core_y = factors[j][2] @ torch.eye(train_n)[shuffle].T
        shuffled.append(torch.linalg.matrix_norm(procrustes_predict(factors[i], factors[j], test_x, shuffled_core_y)-test_y).item())

    # Focused numerical check: norms must be preserved by the completed R.
    probe = x[test_idx, 0]
    mapped_probe = procrustes_predict(factors[0], factors[1], probe)
    norm_error = (torch.linalg.vector_norm(probe, dim=1)-torch.linalg.vector_norm(mapped_probe, dim=1)).abs().max().item()
    mean_aligned_error = sum(aligned) / n_pairs
    mean_shuffled_error = sum(shuffled) / n_pairs
    print(f"Maximum norm error from full orthogonal map: {norm_error:.3g}")
    print(f"Mean held-out Frobenius error (aligned):  {mean_aligned_error:.6f}")
    print(f"Mean held-out Frobenius error (shuffled): {mean_shuffled_error:.6f}")
    manifest_path = write_manifest(
        config,
        {
            "mean_cosine_cka": mean_cosine_cka,
            "mean_rbf_cka": mean_rbf_cka,
            "max_norm_error": norm_error,
            "mean_aligned_frobenius_error": mean_aligned_error,
            "mean_shuffled_frobenius_error": mean_shuffled_error,
        },
    )
    print(f"Stamped output manifest: {manifest_path}")


if __name__ == "__main__":
    main()
