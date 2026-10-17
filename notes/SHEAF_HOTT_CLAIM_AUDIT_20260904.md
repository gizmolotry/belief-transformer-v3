# Sheaf and HoTT claim audit

Audit started September 4, 2026; concluded September 5, 2026. The filename
retains September 4 so the manuscript reference remains valid. The audit
examined source code, artifact provenance, transport algebra, and statistical
inference, reproducing the saved calculations and checking file hashes.
Algebraic diagnostics and replays occurred; no new empirical experiment or
production code fix occurred.

## Scope and disposition

The stored calculations are reproducible, but they do not confirm sheaf
gluing failure, semantic curvature, path-dependent holonomy, or HoTT
annealing-path obstructions. The sheaf calculation reverses composition;
the HoTT significance calculation treats dependent dyads as independent.
Input provenance and the operational definitions further limit both claims.

The original scripts and JSON files entered commit
`07a21e32d19e99cf3845af9ad1658583bd1069f6` on July 23, 2026. The audit began
at `1a160a46`; the Q0Q1 packet is separately committed as `972d6b1c`.
Neither original script nor JSON was edited. Preserve these files as legacy
evidence, including misleading historical labels, rather than silently
replacing their outputs with corrected calculations.

The current abstract treats the formalisms as interpretive analogies;
Section 4.2 and related holonomy wording withdraw their confirmation.
General Section 4.1 arguments and unrelated empirical claims were not
audited here. This disposition does not validate the entire paper.

## Sheaf calculation: reproducibility and transport error

`scripts/run_sheaf_cocycle.py` reads
`outputs/non_circular_suite/exp1_unstructured/observer_42.pt`, takes its
`T1_embeddings` array of shape `(80, 24)` and dtype float32, and splits the
columns into eight three-dimensional blocks. Independent auditor Maxwell
and the coordinator reproduced the saved residual summaries:

| Quantity | Reproduced legacy value |
| --- | --- |
| Mean | 1.1225010830175581 |
| Standard deviation | 0.429060063752489 |
| Minimum | 0.09485056777713115 |
| Maximum | 1.9899898434909782 |

The checkpoint `T1_embeddings_raw.npy` equals the stored torch tensor.
This agreement verifies the local numerical input, not its interpretation
as eight distinct raw observer outputs.

For centered row-vector data, the script computes `R = U @ Vt` from the
SVD of `X.T @ Y`, so `X @ R` aligns with `Y`. Successive maps therefore
compose as `R_ij @ R_jk`. The script instead compares `R_ik` with
`R_jk @ R_ij`. The convention agrees with the objective documented by
[SciPy 1.16.2 orthogonal_procrustes](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.linalg.orthogonal_procrustes.html),
which minimizes the difference between `A @ R` and `B`.

The previously completed in-memory diagnostic, changing only composition
order, gave mean residual `0.885711577748725`. This is a diagnostic of the
legacy calculation, not a new finding, experiment, or replacement result.
Its nonzero value does not rescue the curvature interpretation.

An exact flat fixture isolates the error without corpus assumptions. Let
`Z = vstack([eye(3), -eye(3)])` and define

```text
Rx = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
Ry = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
views = [Z, Z @ Rx, Z @ Ry]
```

These are exactly rotated copies of one centered, full-rank point cloud.
Across six ordered triples, the legacy normalized residual has mean
`sqrt(2) = 1.414213562373095`; the correctly composed residual is zero.
This false positive alone defeats inference of curvature from the legacy
statistic. The fixture is an algebraic diagnostic already completed during
the audit, not a new empirical experiment.

The 336 triples are `8 * 7 * 6` overlapping ordered triples, representing
only 56 unordered triangles. They are not 336 independent observations;
the reported standard deviation describes this dependent enumeration.
There is no null, permutation analysis, or holdout, and no documented
calibration of the `0.15` cutoff. The three-dimensional blocks are full
rank: neither an `N << d` argument nor exact rank deficiency explains this
failure.

## Sheaf input provenance

The experiment-1 runner in `scripts/run_non_circular_suite.py` selects
`data/control_unstructured.jsonl`, enhanced mode, seed 42. Thus the legacy
input is one synthetic corpus artifact, not evidence across real and
synthetic corpora. Saved metadata records `use_gru=True` and channel
`logits`. However, `core/complete_pipeline.py` near line 2935 stores
`final_features` as `T1_embeddings`; near line 2451 it applies the temporal
GRU to the whole feature vector. `core/temporal_gru.py` near line 63 defines
the GRU over that feature dimension. Consequently, contiguous slices of
the saved representation are unverified as separate raw observer logits.

Upstream metadata records `git_dirty=True` and base commit
`587ec40c0695`. Upstream generation was not rerun. The runner and metadata
support this provenance description but do not reconstruct every local
change that produced the artifact.

## HoTT calculation: exact replay and inference limits

On September 5 the coordinator exactly replayed every value in
`data/hott_metrics.json` by invoking the original module's `main`, with
its sole output-file write intercepted into `io.StringIO`. Before/after
hashes checked the source, input, and saved output. The replay environment
was Python 3.13.2, NumPy 2.3.3, SciPy 1.16.2, and torch 2.10.0+cu128,
executing on CPU. This establishes reproducibility from the current local
artifact, not independently authenticated corpus generation.

`outputs/observer_43.pt` supplies `bot_rkhs` with shape `(50, 8, 2048)`.
Metadata calls the corpus real and records seed 43, but authenticated
article IDs and source-corpus generation were not verified. The analysis
itself fixes random seed 42. It uses 50 Dirichlet draws for each alpha in
`[0.1, 0.5, 1, 2, 5, 10, 20]`. A crack means that the maximum, over alpha,
of cosine-similarity variance exceeds `0.003`; a bond means mean similarity
exceeds `0.45`. Friction is variance of cosine distances across the eight
observers, using the same underlying features.

There are 1225 dyads: 1166 cracks and 59 others, with 1223 bonds. Bond and
crack labels are separate thresholds, not mutually exclusive categories.
The reproduced Welch statistic is `t = 54.88058415340992`, with stored
`p = 0.0`; mean friction is `0.05799954815662702` for cracks and
`0.0028699586704493386` otherwise. Crack/friction Pearson
`r = 0.3412425952403506` and Spearman `rho = 0.36958999936102155` are
distinct from the t statistic. Numerical zero is the software's output,
not a literal probability of zero. The complete remaining values stay in
the unchanged JSON.

Pairs share articles, so removing the diagonal and duplicate orientations
does not create independent observations. The installed
`scipy.stats.ttest_ind.__doc__` explicitly specifies independent samples;
Welch's unequal-variance option does not remove dyadic dependence.
Accordingly, the saved test does not support independent population
inference, and its small numerical p value cannot confirm a topological
prediction. Shared feature construction also limits interpretation, but
is neither an exact identity between crack and friction nor proof that
the entire association is artificial. There is no permutation null or
measured homotopy invariant. No new p value or rescue experiment is offered.

The initial independent HoTT auditor reached a usage limit before reporting,
so the coordinator completed that replay. On September 5 a separate final
reviewer executed both reproduction blocks below, confirmed exact HoTT
JSON equality and the sheaf diagnostics, checked all seven fingerprints
and the stated input limitations, and approved this bounded correction.
That approval excludes Section 4.1, unrelated claims, and whole-paper
empirical validation.

A clean Tectonic build produced the 28-page PDF. The coordinator inspected
the rendered abstract and discussion pages (1 and 21--24); the final reviewer
also inspected pages 1 and 23. No new compiler warnings or missing glyphs
were introduced; existing unrelated layout and Fontconfig warnings remain.
The legacy analysis scripts remain unchanged, including their documented
defects. This change corrects the manuscript and preserves the audit trail;
it does not deliver repaired production analyses.

## Evidence fingerprints and next requirements

These SHA256 values were recomputed with PowerShell `Get-FileHash` on
September 5, rather than transcribed from supplied hashes. Paths are
relative to the repository root.

```text
scripts/run_sheaf_cocycle.py
80be5ec8c9a74b784dc3f62bb45af84591ab1ae94a3b4908817639e166bd9c42
scripts/run_hott_predictions.py
1869c661048bf28b45d78596939432c3243446aaf5ca19f6118f4379132e2069
data/sheaf_metrics.json
1100791cbb2b420cfaff713fa259830a1f3bc7ae64906aed39cf252c02c5e76b
data/hott_metrics.json
4fc2442e489e38dae53a890089726884fe8807de40c8b9016ed116f758644246
outputs/non_circular_suite/exp1_unstructured/observer_42.pt
d903f694898a23ae9666ee0e6e4d7f705fb328ce01c36646999e03522a110788
outputs/non_circular_suite/exp1_unstructured/checkpoints/batch/T1_embeddings_raw.npy
10850befdba30d0b8afef57e9a91d41ca8b93f064b2d535f3de9a3381b30facc
outputs/observer_43.pt
0be3ce0de2d36a099fb07410c4922199f1a6c918b96863e20a1bfd77c79598db
```

Future evaluation requires corrected and tested transport, verified input
observer identity, a justified null and control design, an appropriate
independent unit, and a frozen protocol before examining outcomes. These
requirements describe future work; none of those new experiments was done.

## Reproducibility appendix

From the repository root, supply each Python block separately to
`python -B -` through stdin (in PowerShell, a single-quoted here-string
piped to that command). No files are written. These snippets document the
audit diagnostics and the coordinator's guarded replay method.

```python
from itertools import permutations
from pathlib import Path
import numpy as np
from scripts.run_sheaf_cocycle import compute_procrustes_rotation as fit

def residuals(views, order, reverse):
    R = {(i, j): fit(views[i], views[j]) for i in order for j in order}
    return np.array([
        np.linalg.norm(R[i, k] - (
            R[j, k] @ R[i, j] if reverse else R[i, j] @ R[j, k]
        ), ord="fro") / np.sqrt(3.0)
        for i, j, k in permutations(order, 3)
    ])

base = Path("outputs/non_circular_suite/exp1_unstructured")
X = np.load(base / "checkpoints/batch/T1_embeddings_raw.npy")
views = [X[:, 3*i:3*i+3] for i in range(8)]
Z = np.vstack([np.eye(3), -np.eye(3)])
Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
for label, reverse in [("legacy", True), ("correct", False)]:
    r = residuals(views, range(8), reverse)
    print(label, "actual mean,std,min,max:",
          r.mean(), r.std(), r.min(), r.max())
    f = residuals([Z, Z @ Rx, Z @ Ry], range(3), reverse)
    print(label, "fixture mean:", f.mean())
```

```python
import builtins
import hashlib
import io
import json
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch
import scripts.run_hott_predictions as legacy

output = (legacy.DATA_DIR / "hott_metrics.json").resolve()
saved_bytes = output.read_bytes()  # Before patching open.
expected = json.loads(saved_bytes)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

input_before = sha(legacy.OBSERVER_FILE)
source_before = sha(legacy.__file__)
real_open = builtins.open
buffer = io.StringIO()
stdout = io.StringIO()

@contextmanager
def output_sink():
    yield buffer

def guarded_open(file, mode="r", *args, **kwargs):
    if mode in ("r", "rb", "rt"):
        return real_open(file, mode, *args, **kwargs)
    elif mode == "w" and Path(file).resolve() == output:
        return output_sink()
    else:
        raise RuntimeError(f"Rejected open: {file!r}, {mode!r}")

with patch("builtins.open", guarded_open), redirect_stdout(stdout):
    legacy.main()
actual = json.loads(buffer.getvalue())
assert sha(legacy.OBSERVER_FILE) == input_before
assert sha(legacy.__file__) == source_before
assert output.read_bytes() == saved_bytes
print("JSON equality:", actual == expected)
print("t:", actual["t_test_cracked_vs_non_cracked"]["t_statistic"])
assert actual == expected
```
