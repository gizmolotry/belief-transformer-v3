# Full SPD Observer Tensor Necessity Preregistration

Status: preregistration only. This document reports or implies no experiment result.

This is a one-shot, bounded test of a narrow claim: whether the directional and off-diagonal contribution of the full rank-one SPD observer tensor improves cross-bank stability of an article-level construct beyond prespecified simpler geometries. It is not a test of publisher-label accuracy, MBFC/AllSides alignment, ideological truth, or a continuous-geodesic interpretation.

## 1. Frozen Cohort, Prompt Banks, and Artifact Binding

The real endpoint may run only after the following are committed and hash-bound:

- A fixed cohort of exactly 320 articles: 16 publishers with 20 articles each, identified by an ordered article-ID manifest and article-to-publisher mapping.
- Three ordered, prespecified prompt banks named `Q0`, `Q1`, and `Q2`, each containing eight contrastive hypothesis pairs. The existing Q1 and Q2 files are direct paraphrases of Q0, were added with Q0, and make no independent-authorship claim. They are valid only for secondary stability-under-paraphrase analysis and cannot authorize the real primary endpoint. Before that endpoint, Q1 and Q2 must be replaced by independently authored operationalizations created without exposure to Q0's wording. Their committed prompt sources, exact raw-file SHA-256 hashes, parsed ordered text hashes, and pair ordering are recorded.
- For every bank `q`, a distinct prompt-hash-bound extraction artifact set: article matrix `X_q`, NLI substrate `T0_q`, and bank-derived representation artifact `T1_q`. The manifest records the bank hash, ordered cohort hash, raw-byte hashes, shapes, dtypes, endianness, extraction configuration, and source commit for every artifact.
- A deterministic five-fold publisher-grouped split, committed as an ordered fold-assignment manifest.
- Numeric values for the practical margin and every validity threshold in Section 8, calibrated and committed before any real primary endpoint calculation.

`X_Q`, `T0_Q`, and `T1_Q` must be generated separately for each prompt bank. No arm, including `identity`, may reuse one embedding matrix as if it represented all three banks. A missing bank artifact, prompt-hash mismatch, cohort mismatch, or artifact without its bank provenance makes the run `invalid`.

No real primary endpoint may be inspected before these items are committed. Synthetic studies may validate the implementation and calibrate gates, but may not rescue a real endpoint.

### Independent-framing hard blocker

Every bank manifest records explicit `framing_provenance`. Q0 must declare `relation_to_Q0=anchor`. Q1 and Q2 must each declare `relation_to_Q0=independent_operationalization`, `independently_authored_without_q0_exposure=true`, and nonempty, mutually distinct `author` and `source_process` values. The corresponding prompt-source files must be tracked, committed, available at the recorded paths, and byte-identical to their manifest hashes. Real preflight and real authorization both fail closed otherwise.

This provenance is a human-audited declaration, not computational proof of independence. Its purpose is to make the authorship and exposure claim explicit, attributable, immutable, and reviewable. The current declarations (`Q1/Q2: relation_to_Q0=paraphrase_of_Q0`, `independently_authored_without_q0_exposure=false`) therefore preserve those banks for secondary analysis while hard-blocking the primary run.

## 2. Folds, Information Boundary, and Observer Target

All 20 articles from a publisher are assigned to one of five deterministic publisher-grouped outer folds. For every fold `f`, bank `q`, and seed `sigma`, the potential is trained only on the four training publisher groups and evaluated only on the held-out publisher group. There is no cross-fold tensor stitching.

The only training target is observer-derived. For article `i`, observer `o`, and bank `q`, the stored `T0_q` logits must be transformed by softmax using the manifest-recorded and hash-bound class-axis order. Any permutation of exactly the three canonical labels `{entail, neutral, contradict}` is allowed; the frozen DeBERTa-v2-xlarge-MNLI order is `[contradict, neutral, entail]`. Define

`s_ioq = P_ioq(entail) - P_ioq(contradict)`

and freeze the article target as the population standard deviation across the eight observers:

`y_iq = sqrt((1 / 8) * sum_o (s_ioq - mean_o(s_ioq))^2)`.

The manifest must record and hash-bind the unambiguous class-axis labels and their order before softmax. A missing or ambiguous order, a label set other than exactly the three canonical labels, an incorrect mapping between output indices and labels, a non-logit interpretation, or a target formula change makes the run `invalid`.

No public publisher label, human label, LLM label, external ideology scale, held-out outcome, or real-endpoint result may enter target construction, training, arm selection, threshold selection, or primary inference.

## 3. Potential Model and Training

For each `(f, q, sigma)`, fit exactly one `IdeologicalPotential` with:

- `RiemannianConfig(potential_scale=1.0)`;
- `hidden_dim=D`, the frozen width of `X_q`;
- `n_ideologies=8` and `ideology_idx=None`;
- full-batch `AdamW`, learning rate `1e-3`, weight decay `1e-4`, and exactly 300 epochs;
- loss `MSE + 1e-4 * mean(||grad Phi||^2)`;
- seeds `42`, `420`, and `4200`.

Standardize `y_iq` using only the training-fold mean and standard deviation. A zero training standard deviation is `invalid`. Held-out targets use only those training-fold parameters.

The full, scalar-trace, and diagonal arms use the same fitted potential and the same node gradients `v_i = grad Phi(X_iq)` in every cell. The full arm has no extra fitted parameters relative to those two arms. Its additional directional information is an analytic use of the shared gradient, so anti-overfitting checks assess potential generalization and geometric amplification, not presumed extra fitting capacity.

The implementation uses `torch.random.fork_rng` and deterministic operations where supported, and must not mutate the caller's global PyTorch RNG state. Training on held-out publishers or any prohibited target is `invalid`.

## 4. Frozen Support Graph and Metric Arms

Within each held-out `(f, q)` cell, construct one Euclidean union-symmetrized k-nearest-neighbor support graph from the bank-specific held-out `X_q`, with `k=10`. Neighbor selection resolves equal Euclidean distances by `(distance, canonical article ID)` only to construct edges. The resulting adjacency is shared by every arm in that cell. It must be connected; otherwise the primary result is `inconclusive`.

For edge displacement `d_ij = X_jq - X_iq`, all lengths are nonnegative square roots:

- `identity`: `L_id = ||d_ij||`.
- `oof_conformal`: use only an out-of-fold scalar prediction with the same information boundary; `rho_i = 1 / (1 + abs(yhat_i))`, `L_conf = ||d_ij|| / sqrt(rho_i rho_j)`.
- `scalar_trace`: with `a_i = 1 + ||v_i||^2 / D`, `L_trace^2 = 0.5 * (a_i + a_j) * ||d_ij||^2`.
- `diagonal`: `L_diag^2 = sum_r d_ijr^2 * [1 + 0.5 * (v_ir^2 + v_jr^2)]`.
- `full`: `L_full^2 = ||d_ij||^2 + 0.5 * [(d_ij dot v_i)^2 + (d_ij dot v_j)^2]`.
- `article_shuffled`: full-arm gradients are deterministically shuffled among held-out articles within publisher.
- `observer_shuffled`: observer targets are deterministically shuffled within publisher in the training fold before fitting the potential.
- `spectrum_matched_random`: replace each `v_i` by `||v_i|| u_i`, where `u_i` is a deterministic random unit vector.

The full formula is the trapezoid endpoint integration of `g_i = I + v_i v_i^T`; no dense `D x D` tensor allocation is required. The FMM/Track4 trace-reduced path is not evidence for this claim because it removes tensor orientation.

Sensitivity analyses use `k=5` and `k=15`; a union-arm support analysis is secondary. A complete graph is permitted only for synthetic fixtures with `N <= 100`.

## 5. Outcome, Midranks, and Cross-Bank Stability

For every held-out article, arm, bank, fold, and seed, calculate eccentricity as the mean finite shortest-path distance to all other held-out articles in that fold:

`e_iAfqsigma = (1 / (n_f - 1)) * sum_(j != i) D_Afqsigma(i, j)`.

Average eccentricities over the three prespecified seeds before calculating the primary statistic. Convert the seed-averaged eccentricities to midranks independently within each `(f, q, arm)`. Canonical article IDs may serialize files and manifests but must never break inferential ties.

For `m=3` banks, let `r_iqA` be the midrank, `R_iA = sum_q r_iqA`, and let `t_qg` be the size of tie group `g` in bank `q`. Compute tie-corrected Kendall's W:

`W_fA = [12 * sum_i (R_iA - mean_i(R_iA))^2] / [m^2 * (n_f^3 - n_f) - m * sum_q sum_g (t_qg^3 - t_qg)]`.

Then compute `T_A = sum_f (n_f / N) * W_fA`, and define

`Delta = T_full - max(T_identity, T_oof_conformal, T_scalar_trace, T_diagonal)`.

Because every bank has its own prompt-bound `X_q`, `identity` is bank-dependent and remains a required comparator. The strongest comparator is mechanically reselected in every observed statistic, permutation replicate, and bootstrap replicate.

## 6. Required-Arm Validity and Decision Rule

The required comparators are `identity`, `oof_conformal`, `scalar_trace`, and `diagonal`. They are never dropped or replaced after inspection. A failure of any required arm's connectivity, provenance, collapse, inflation, or potential-generalization gate makes the entire primary result `inconclusive`.

The full arm is also subject to every validity gate. A provenance, leakage, frozen-setting, or information-boundary violation makes the result `invalid`.

The primary pass rule requires every condition below:

- all validity and provenance gates pass;
- `Delta >= Delta_min`;
- one-sided permutation `p_Delta < .05`;
- the preregistered publisher-cluster bootstrap lower confidence bound for `Delta` is `> 0`;
- `T_full > T_scalar_trace` and `T_full > T_diagonal`;
- for each Q-bank pair, full's seed-averaged Spearman stability exceeds that of the strongest comparator;
- at least 80% of prespecified seed-by-Q-pair full-minus-strongest-comparator stability cells are positive;
- all negative-control requirements in Section 10 pass.

If all validity gates pass but any pass condition fails, the result is `fail`. A fail means the full tensor did not establish necessity for this preregistered stability claim; it does not establish that the rest of the system fails.

## 7. Permutation, Bootstrap, and Practical Margin

Use 10,000 primary permutations. Within each held-out fold and publisher, independently permute the Q1 and Q2 article correspondences relative to Q0; apply a given mapping to all arms; and reselect the strongest comparator in every replicate. The one-sided p-value is:

`p_Delta = (1 + count_b[Delta_b >= Delta_observed]) / 10001`.

Use 10,000 publisher-cluster bootstrap replicates within the original folds. Preserve the multiplicity of duplicate sampled publishers and their articles. Recompute midranks, tie correction, `W`, `T`, comparator selection, and `Delta` for every replicate.

Before the real endpoint, choose `Delta_min` from a fixed 0.01 grid by running the complete pipeline on frozen no-effect and planted-anisotropy simulations. Commit the smallest value with type-I error `<= .05` and power `>= .80`, rounded upward to the next 0.01. If no value meets both conditions, this design is rejected and the real endpoint must not run.

## 8. Calibrated Collapse, Generalization, and Inflation Gates

Before the real endpoint, freeze numeric values for `tau_disp`, `tau_tie`, `tau_gen`, `tau_gap`, and `tau_amp` using no-effect, planted-anisotropy, manufactured-collapse, and manufactured-inflation simulations.

- `tau_disp`: minimum allowed coefficient of variation of held-out eccentricities for each required arm and cell.
- `tau_tie`: maximum allowed near-tie fraction among held-out eccentricities for each required arm and cell. For each arm and cell, the near-tie tolerance is exactly `1e-8 * max(median(abs(eccentricities)), np.finfo(np.float64).tiny)`; exact ties are handled with midranks.
- `tau_gen`: minimum allowed held-out potential generalization, where `G_hold = 1 - MSE_hold(Phi) / MSE_hold(mean(y_train))`.
- `tau_gap`: maximum allowed generalization gap, where `G_gap = G_train - G_hold` and `G_train` uses the training-fold baseline and predictions.
- `tau_amp`: maximum allowed full-arm edge amplification, where `A_0.95 = Q_0.95({L_full / L_id : (i,j) is an edge in E})`.

The selected threshold set must flag at least 95% of manufactured collapse or inflation simulations and retain at least 95% of planted-anisotropy simulations. If no threshold set satisfies both conditions, the design is rejected and the real endpoint must not run.

Potential generalization and non-inflation are prerequisites and diagnostics. Their failure is `inconclusive`, not evidence that directional information is unnecessary and not evidence of additional full-arm fitted parameters.

## 9. Secondary Robustness and Corpus Controls

Report, without allowing them to rescue a failed primary endpoint:

- Q0--Q1, Q0--Q2, and Q1--Q2 pairwise results;
- each seed separately and each primary comparator separately;
- `k=5`, `k=10`, `k=15`, and secondary union-arm support;
- real, constant, shuffled, and random corpus controls.

Control randomization streams are derived independently for each `(control_family, Q_bank, outer_fold, seed)` tuple. A primary pass on any corpus control is invalidating.

## 10. Negative Direct Tests and Holm Family

The negative-control family contains exactly `article_shuffled`, `observer_shuffled`, and `spectrum_matched_random`. For each control `C`, compute `S_C = T_full - T_C` with paired permutations and publisher-cluster bootstrap. Apply Holm correction to the three permutation p-values. Every comparison must have Holm-adjusted `p < .05` and bootstrap lower bound `> 0`.

Also substitute each negative control as the candidate arm and rerun the complete primary rule, including strongest-comparator reselection inside every observed, permutation, and bootstrap calculation. A negative control passing the primary rule is invalidating.

`scalar_trace` and `diagonal` remain required scientific comparators and are not members of this Holm family.

## 11. Status Language

- `pass`: all frozen gates and all pass conditions hold. This supports only the narrow directional/off-diagonal stability claim stated above.
- `fail`: all validity gates hold, but one or more pass conditions do not.
- `inconclusive`: a required arm collapses, inflates, is disconnected, lacks required potential generalization, or otherwise fails a required-arm validity gate.
- `invalid`: a provenance/hash mismatch, leakage, uncommitted setting or threshold, information-boundary breach, corpus-control pass, or negative-control invalidation occurs.

No result may be described as proving continuous geodesics, ideological truth, external-label accuracy, or a full continuous Riemannian manifold.

## 12. Provenance and Commit Discipline

Every run writes a fail-closed manifest containing: git SHA and dirty status; entry point and arguments; runtime versions; corpus, cohort, ordered article-ID, publisher-ID, and fold hashes; exact Q-bank raw and parsed-order hashes; `X_q`, `T0_q`, and `T1_q` raw-byte hashes/shapes/dtypes/endianness and their bank binding; exact T0 class labels/order; target formula; training normalization; potential config, state, and loss-trace hashes; graph adjacency and edge-weight hashes; all distance-matrix hashes; control, permutation, and bootstrap seeds; every threshold and calibration manifest; and all statistic outputs.

The runner fails closed on missing or mismatched provenance, mislabeled tensor family, train/held-out overlap, disconnected required support, missing required artifact, or absent gate value.

Infrastructure, calibration, and real-result commits are separate. Preregistration and calibration changes after real-endpoint inspection cannot be mixed with result artifacts. This file remains an infrastructure artifact and contains no results.
