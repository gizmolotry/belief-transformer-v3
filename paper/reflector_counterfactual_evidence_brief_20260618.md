# Reflector Counterfactual Evidence Brief - 2026-06-18

## Core Finding

The counterfactual packet supports a post-capstone correction: observer/framing signal is more stable when modeled as a residualized force field over content topology than when early-fused as another coordinate. Early coordinate fusion still fails the specificity gate, while Reflector-style observer force passes both deterministic Track 4 least-action traversal and bounded stochastic traversal.

In short: content defines the map; framing changes the work required to move through it.

## Evidence Artifacts

- Full 3x3 stochastic packet: `outputs/counterfactual_intervention_suite/reflector_full_stochastic_20260618/counterfactual_intervention_summary.json`
- Full 3x3 rows: `outputs/counterfactual_intervention_suite/reflector_full_stochastic_20260618/counterfactual_rows.csv`
- Optimized scaled 3x3 packet: `outputs/counterfactual_intervention_suite/reflector_scaled_3x_optimized_20260618/counterfactual_intervention_summary.json`
- Optimized scaled 3x3 rows: `outputs/counterfactual_intervention_suite/reflector_scaled_3x_optimized_20260618/counterfactual_rows.csv`
- Scaled 3x graph-fix ablation packet: `outputs/reflector_graph_fix_ablation/scaled_3x_all_fixes_20260618/reflector_graph_fix_ablation.json`
- Scaled 3x graph-fix ablation rows: `outputs/reflector_graph_fix_ablation/scaled_3x_all_fixes_20260618/reflector_graph_fix_ablation.csv`
- Scaled 3x graph-fix claim matrix: `outputs/reflector_graph_fix_ablation/scaled_3x_all_fixes_20260618/claim_matrix.json`
- Scaled 3x transport ablation packet: `outputs/reflector_transport_ablation/scaled_3x_anchor_transport_20260618/reflector_transport_ablation.json`
- Scaled 3x transport ablation rows: `outputs/reflector_transport_ablation/scaled_3x_anchor_transport_20260618/reflector_transport_ablation.csv`
- Scaled 3x transport claim matrix: `outputs/reflector_transport_ablation/scaled_3x_anchor_transport_20260618/claim_matrix.json`
- Scaled 3x adversarial transport packet: `outputs/reflector_adversarial_transport_suite/scaled_3x_adversarial_20260619/reflector_adversarial_transport_suite.json`
- Scaled 3x adversarial transport rows: `outputs/reflector_adversarial_transport_suite/scaled_3x_adversarial_20260619/reflector_adversarial_transport_suite.csv`
- Scaled 3x adversarial transport claim matrix: `outputs/reflector_adversarial_transport_suite/scaled_3x_adversarial_20260619/claim_matrix.json`
- Scaled 3x semantic-use gate packet: `outputs/reflector_semantic_gate_ablation/scaled_3x_action_gates_20260619/reflector_semantic_gate_ablation.json`
- Scaled 3x semantic-use gate rows: `outputs/reflector_semantic_gate_ablation/scaled_3x_action_gates_20260619/reflector_semantic_gate_ablation.csv`
- Scaled 3x semantic-use gate claim matrix: `outputs/reflector_semantic_gate_ablation/scaled_3x_action_gates_20260619/claim_matrix.json`
- Scaled 2x canonical packet: `outputs/counterfactual_intervention_suite/reflector_scaled_2x_canonical_20260618/counterfactual_intervention_summary.json`
- Scaled 2x rows: `outputs/counterfactual_intervention_suite/reflector_scaled_2x_canonical_20260618/counterfactual_rows.csv`
- Implementation: `scripts/run_counterfactual_intervention_suite.py`
- Graph-fix ablation implementation: `scripts/run_reflector_graph_fix_ablation.py`
- Transport ablation implementation: `scripts/run_reflector_transport_ablation.py`
- Adversarial transport implementation: `scripts/run_reflector_adversarial_transport_suite.py`
- Semantic-use gate ablation implementation: `scripts/run_reflector_semantic_gate_ablation.py`
- Regression tests: `test_counterfactual_reflector_suite.py`
- Graph-fix ablation tests: `test_reflector_graph_fix_ablation.py`
- Transport ablation tests: `test_reflector_transport_ablation.py`
- Adversarial transport tests: `test_reflector_adversarial_transport_suite.py`
- Semantic-use gate tests: `test_reflector_semantic_gate_ablation.py`

## Packet Shape

Full packet:

| Field | Value |
|---|---:|
| Events | 12 |
| Controlled rows | 108 |
| Prompt banks | canonical, terse, journalistic |
| Seeds | 42, 420, 4200 |
| Grid cells | 9 |
| Stochastic walks per pair | 8 |

Scaled packet:

| Field | Value |
|---|---:|
| Events | 24 |
| Controlled rows | 216 |
| Prompt banks | canonical |
| Seeds | 42 |
| Stochastic walks per pair | 8 |

Optimized scaled 3x3 packet:

| Field | Value |
|---|---:|
| Events | 36 |
| Controlled rows | 324 |
| Prompt banks | canonical, terse, journalistic |
| Seeds | 42, 420, 4200 |
| Grid cells | 9 |
| Stochastic walks per pair | 8 |

The first scaled 3x3 attempt timed out because least-action traversal recomputed Dijkstra paths separately for every target and the stochastic walker recomputed edge actions inside every sampled step. The optimized implementation runs one multi-target Dijkstra search per neutral source and caches stochastic transition actions. The optimized scaled 3x3 packet completed, but it exposed a scale boundary: traversal separation did not survive the larger graph.

## Early Fusion Failure

The original early-fusion architecture still fails the important specificity test.

| Test | Result |
|---|---:|
| Early fusion specificity | 0/9 |
| Old robustness gate | 0/9 |
| Baseline comparison | 4/9 |
| Controlled counterfactual recovery | 9/9 |

Mean early-concat specificity ratio was `1.409`, below the `1.50` threshold. This is useful rather than embarrassing: it shows why the architecture should not treat observer/framing as just another coordinate.

## Reflector Correction

Reflector separates topology from dynamics:

- Content topology is built from text/content features.
- Observer signal is residualized against surface nuisance and event-locked content.
- Event-centered observer deltas are treated as stress/shear over the content graph.
- Prompt banks are reliability weighted rather than democratically averaged.

Full 3x3 packet:

| Metric | Mean |
|---|---:|
| Anchor-only specificity ratio | 13.739 |
| Reflector delta specificity ratio | 1.686 |
| Late graph observer-lift ratio | 1.686 |
| Early-concat specificity ratio | 1.409 |
| Weighted prompt ensemble | 3/3 |

Individual prompt banks remain visible: canonical is strongest, terse is mixed, and journalistic under-separates controls. The weighted ensemble passes because prompt banks earn weight based on reliability instead of being averaged equally.

## Full Traversal

The full traversal test uses Track 4 least-action graph traversal, not just local edge scoring. For each event, the system finds paths from the neutral node to frame/control variants over the content graph while observer deltas act as stress/shear/transport fields.

Full 3x3 packet:

| Test | Result |
|---|---:|
| Reflector full traversal | 9/9 |
| Mean frame/control action ratio | 1.613 |
| Mean observer-transport ratio | 2.289 |
| Weighted ensemble full-traversal ratio | about 1.63-1.64 |

Interpretation: frame edits are not merely farther as text; they require more work through the observer-conditioned graph.

## Stochastic Walker

The stochastic check is a bounded Boltzmann action walker over the same Reflector graph. It samples transitions with probability proportional to lower action and lower target distance, then compares successful sampled traversal cost for frame edits versus specificity controls.

Full 3x3 packet:

| Test | Result |
|---|---:|
| Reflector stochastic traversal | 9/9 |
| Mean stochastic successful-action ratio | 1.479 |
| Min stochastic successful-action ratio | 1.241 |
| Mean control-minus-frame success rate | 0.285 |

Scaled 2x canonical packet:

| Test | Result |
|---|---:|
| Reflector stochastic traversal | 1/1 |
| Stochastic successful-action ratio | 1.440 |
| Full traversal action ratio | 1.469 |
| Delta specificity ratio | 1.896 |

Optimized scaled 3x3 packet:

| Test | Result |
|---|---:|
| Beats early concat | 9/9 |
| Reflector delta specificity | 3/9 |
| Reflector full traversal | 0/9 |
| Reflector stochastic traversal | 0/9 |
| Weighted prompt ensemble | 0/3 |
| Mean full-traversal action ratio | 1.022 |
| Mean stochastic successful-action ratio | 0.944 |

Interpretation: the 12-event packet supports the Reflector traversal claim, and the 2x canonical scale check supports the same direction. The full 3x3 scale test does not support a broad scale-robust traversal claim yet. This likely reflects graph-topology dilution: as more near-neighbor event variants enter the graph, least-action and stochastic walkers find alternate low-cost routes that flatten frame/control traversal differences.

This is stochastic path sampling, not a claim that the legacy MCMC/cyclic walker has been fully validated.

## Graph-Fix Ablation

The scaled traversal failure was tested against five concrete engineering fixes using the same ledger style as the counterfactual suite:

- Event-stratified graph support
- Observer-resistance floor
- Frame-aware shortcut penalty
- Mutual-kNN support
- Per-event transition normalization

A combined-all-fixes condition was also tested. The goal was deliberately strict: restore scaled frame-over-control traversal separation on the 36-event, 324-row, 3-seed, 3-prompt-bank packet.

Scaled 3x graph-fix packet:

| Variant | Passes | Mean action ratio | Mean reached rate |
|---|---:|---:|---:|
| Baseline global union | 0/9 | 1.027 | 1.000 |
| Event-stratified graph | 0/9 | 1.035 | 0.988 |
| Observer-resistance floor | 1/9 | 1.084 | 1.000 |
| Frame-aware shortcut penalty | 0/9 | 1.018 | 1.000 |
| Mutual-kNN graph | 0/9 | 0.791 | 0.833 |
| Per-event normalized | 0/9 | 1.039 | 1.000 |
| Combined all fixes | 0/9 | 0.580 | 0.278 |

Interpretation: none of the five simple graph-level fixes restored the scaled traversal claim. The observer-resistance floor was the only partial improvement, suggesting the failure is genuinely about low-cost observer shortcuts rather than pure runtime or pure kNN support. Mutual-kNN and combined constraints hurt reachability, which means the system cannot simply prune harder without losing traversal coverage.

Engineering reading: the next architecture should not just add more penalties to the current global graph. It likely needs a stronger separation between event-family topology and observer transport: content builds local event neighborhoods, observer deltas define directed edge resistance inside or between those neighborhoods, and cross-event bridges need calibrated transport gates rather than universal nearest-neighbor shortcuts.

## Transport Ablation

The next ablation tested that stronger architecture directly. Instead of treating observer signal as another coordinate or as a simple penalty on every global edge, the transport suite separates three jobs:

- Content/event topology defines where local movement is possible.
- Residualized observer deltas define radial transport potential away from the event-neutral baseline.
- Raw anchor-salience displacement acts as an instrument gate when residualization weakens the prompt signal.

Scaled 3x transport packet:

| Variant | Passes | Mean action ratio | Min action ratio | Mean reached rate |
|---|---:|---:|---:|---:|
| Baseline global union | 0/9 | 1.027 | 0.967 | 1.000 |
| Radial transport potential | 3/9 | 1.315 | 1.043 | 1.000 |
| Event-family topology | 6/9 | 1.336 | 1.123 | 1.000 |
| Bridge-gated transport | 6/9 | 1.433 | 1.151 | 1.000 |
| Anchor-salience transport | 9/9 | 1.894 | 1.648 | 1.000 |
| Calibrated transport combo | 9/9 | 1.802 | 1.457 | 1.000 |
| Oracle role upper bound | 9/9 | 1.859 | 1.604 | 1.000 |

Interpretation: the scale failure was not simply caused by too many nodes or a bad shortest-path implementation. The information needed to recover framing transitions survives at 3x scale, but it is not reliably preserved by residual observer deltas alone. Anchor-salience transport restores separation across all prompt banks and seeds, including the journalistic prompt bank that previously failed at about `1.23`.

Engineering reading: the system should treat observer/framing as a force field over content topology, but the force field needs an instrument channel. Residualized deltas are conservative and useful for reducing surface leakage; raw anchor salience is less conservative but better at preserving weak prompt signal. The calibrated combo is therefore the current best candidate: use residualized deltas for topology-aware transport, and use calibrated raw anchor salience as a bounded gate rather than as an unconstrained coordinate.

Caveat: anchor-salience transport is not the same claim as real-world ideology classification. It proves a controlled counterfactual mechanism: planted frame edits require more observer-transport work than paraphrase/entity controls in the scaled synthetic packet. Before using it as a publication-strength method, it needs adversarial prompt-artifact tests and non-synthetic validation.

## Adversarial Transport Validation

The adversarial suite tests whether the transport recovery survives five fake-signal traps:

- Sentiment-only language
- Salient entity swaps
- Length padding
- Lexical overlap with anchor words
- Prompt-paraphrase/meta language

The strict pass condition is that frame edits must cost at least `1.25x` more than every adversarial trap family in every seed/prompt-bank cell.

Scaled 3x adversarial packet:

| Variant | Passes | Mean frame/control ratio | Mean minimum frame/trap ratio | Main failure |
|---|---:|---:|---:|---|
| Baseline global union | 0/12 | 0.985 | 0.809 | lexical overlap, prompt-meta |
| Anchor-salience transport | 0/12 | 1.501 | 0.485 | lexical overlap, prompt-meta |
| Calibrated transport combo | 0/12 | 1.857 | 0.657 | lexical overlap, prompt-meta |
| Oracle role upper bound | 6/12 | 1.989 | 1.151 | prompt-meta |

Trap family pass rates:

| Variant | Sentiment | Entity | Length | Lexical overlap | Prompt-meta |
|---|---:|---:|---:|---:|---:|
| Anchor-salience transport | 12/12 | 12/12 | 9/12 | 0/12 | 0/12 |
| Calibrated transport combo | 12/12 | 12/12 | 12/12 | 0/12 | 0/12 |
| Oracle role upper bound | 12/12 | 12/12 | 12/12 | 12/12 | 6/12 |

Interpretation: anchor salience is a real recovery mechanism for scaled traversal, but it is not yet robust to lexical prompt artifacts. It distinguishes frame edits from sentiment-only, entity-swap, and mostly length-padding controls. It fails when a control text explicitly contains the same conceptual vocabulary used by the observer probes without actually adopting the frame. This is exactly the failure mode a reviewer would worry about.

Engineering reading: the next architecture must add semantic-use gating. Anchor salience cannot be used as a raw keyword-sensitive potential. It needs a decoy detector or entailment/use filter that distinguishes "this text frames civilians as victims" from "this text mentions that the word victim appears in public debate." Until then, the transport result is a promising controlled recovery, not a fully artifact-robust method.

## Semantic-Use Gate Ablation

The semantic-use gate ablation tested eleven candidate gates against the same scaled 3x adversarial packet. The key distinction was whether a gate merely filters anchor salience, or whether it also discounts the observer-action term when a target appears to mention a frame without adopting it.

Scaled 3x semantic-use gate packet:

| Variant | Passes | Mean minimum frame/trap ratio | Global minimum frame/trap ratio | Reading |
|---|---:|---:|---:|---|
| No gate anchor salience | 0/12 | 0.485 | 0.308 | Raw salience is artifact-sensitive |
| Frame evidence gate | 3/12 | 0.950 | 0.652 | Too brittle across prompt banks |
| Meta decoy suppression | 6/12 | 1.211 | 0.860 | Helps, but cannot handle all prompt banks |
| Use-minus-mention margin | 6/12 | 1.138 | 0.804 | Helps, but still leaks prompt artifacts |
| Fuzzy semantic-use gate | 6/12 | 1.222 | 0.849 | Helps, but still leaks prompt artifacts |
| Hybrid meta/evidence gate | 9/12 | 1.449 | 0.969 | Salience filtering alone almost works |
| Meta action-discount gate | 6/12 | 1.844 | 0.961 | Discounts action, but misses non-meta traps |
| Margin action-discount gate | 9/12 | 1.898 | 1.068 | Stronger, but still fails formal paraphrase |
| Two-channel use gate | 9/12 | 2.147 | 1.160 | Strong, but still misses the strict threshold |
| Oracle trap-filter upper bound | 9/12 | 1.860 | 1.087 | Even salience-only oracle filtering is insufficient |
| Hybrid action-discount gate | 12/12 | 2.942 | 1.699 | Current controlled-adversarial-safe candidate |

Interpretation: the remaining artifact problem was not solved by better salience filtering alone. Even the oracle trap filter failed the formal-paraphrase prompt-meta cells because the path action still treated meta-language as observer-use movement. The passing variant fixes this by applying the semantic-use gate to the observer-action term itself. In plain terms: a text that merely talks about a frame should not bend the traversal field as much as a text that actually uses that frame.

Engineering reading: the next Reflector architecture should represent semantic use as a gate on observer dynamics, not merely as a scalar cleanup on anchor features. The safe implementation pattern is: content builds the traversable graph, anchor salience proposes observer force, and semantic-use gating decides how much of that force is allowed to enter the edge-action calculation.

Caveat: this is still a controlled synthetic adversarial packet. It supports an engineering claim about artifact resistance under planted traps, not real-world ideology detection.

## Real-Corpus Claim Boundary

The real corpus remains exploratory. The full packet reports real-corpus frame-axis spread only. It does not establish ground-truth ideology accuracy because there are no independent human labels.

Safe real-corpus language:

> On unlabeled real articles, the system can report exploratory frame-axis spread and provenance-aware geometry. It cannot claim correct ideology classification without independent validation labels.

Unsafe real-corpus language:

> Reflector proves real-world ideology or bias detection accuracy.

## Paper-Safe Claim

Use:

> In controlled counterfactual experiments, early coordinate fusion failed to separate frame edits from specificity controls. A Reflector-style architecture, which treats framing as a residualized observer-force field over a content graph, recovered the planted framing interventions more reliably. The strongest evidence is traversal-based: both least-action Track 4 traversal and bounded stochastic graph walking assigned higher movement cost to framing edits than to paraphrase/entity controls across the 3-bank, 3-seed packet.

Scale-qualified version:

> The 12-event controlled packet supports the Reflector traversal hypothesis, and a 2x canonical scale check preserves it. A full 3x3 scaled graph removes traversal separation under the naive global graph, but the typed transport ablation restores it when anchor-salience transport is used as a calibrated instrument gate. However, adversarial validation shows that this recovery is vulnerable to lexical-overlap and prompt-meta traps. Large-graph traversal is therefore recoverable in controlled settings, but not yet artifact-robust enough for a settled publication claim.

Updated semantic-use qualified version:

> In the scaled synthetic adversarial packet, semantic-use gating restored artifact robustness only when the gate modified the observer-action term itself. The hybrid action-discount gate passed all 12 seed/prompt-bank cells, while salience-only gates, including an oracle trap filter, failed at least one prompt-bank family. This supports the architectural claim that observer force must be conditioned on semantic use rather than lexical mention alone.

## Unsafe Claim

Avoid:

> The Semantic Interferometer proves real-world ideology detection or objective bias measurement without human labels.

## Engineering Boundary

The timeout is fixed, and the first graph-fix ablation showed that simple graph pruning or penalties are insufficient. The transport ablation identifies a viable recovery direction: preserve observer-force contrast with typed transport and calibrated anchor salience rather than treating residualized observer deltas as the only force channel. The adversarial suite then narrowed the artifact boundary: raw anchor salience lets lexical mentions of observer concepts masquerade as adopted frames. The semantic-use gate ablation resolves that boundary in the controlled synthetic packet by discounting observer action when a target mentions a frame without using it.
