# Original User Request

## Initial Request — 2026-07-21T14:24:07-07:00

<USER_REQUEST>
Implement three major architectural upgrades to the Semantic Interferometer pipeline to automate hypothesis generation, introduce a learnable Riemannian metric tensor, and establish a closed-loop observer feedback mechanism.

Working directory: d:/belief-transformer/V3
Integrity mode: development

## Requirements

### R1. Automated Hypothesis Generation ("Auto-Observers")
- Extract the document embeddings for a target corpus and apply Singular Value Decomposition (SVD) to identify the primary axes of semantic variance.
- For each of the top SVD axes, extract the articles located at the positive and negative extremes of the axis.
- Query a local LLM (Qwen2.5-Coder-14B) with these extreme article texts, prompting it to generate a contrastive hypothesis pair (Pair A and Pair B) that characterizes the ideological or thematic tension defining that axis.
- Integrate the generated hypotheses dynamically into the NLI observer forward pass, replacing the manual authoring bottleneck.

### R2. Learnable Metric Tensor & Continuous Geodesic Solver
- Implement a localized, learnable metric tensor $g_{\mu\nu}(x)$ parameterized by a shallow neural network (MLP). The network takes local density $\rho(x)$ and local semantic gradients $\nabla\Phi(x)$ as input and outputs a localized coordinate warp.
- Train this warp network using a self-supervised contrastive loss designed to maximize distance between topologically distant points while preserving local neighborhood consistency, without using publisher editorial labels.
- Replace the discrete Dijkstra $k$-NN solver with a continuous geodesic solver based on the Fast Marching Method (FMM) to solve shortest paths on the warped manifold.

### R3. Dynamic Closed-Loop Observer Tuning
- Implement a closed-loop feedback mechanism that monitors Track 3 topological crack density (Dirichlet fractures).
- For articles situated in regions of high topological crack density, dynamically scale the Dirichlet concentration parameter $\alpha$ (adjusting the mixture weights) to focus NLI pressure on high-conflict boundaries.

---

## Verification Plan

### Automated Verification
- **Hypothesis Generation Check:** A verification script must load a corpus, trigger the Auto-Observer pipeline, output $K$ generated hypothesis pairs, and verify they produce valid NLI logit structures when run through DeBERTa.
- **Learnable Metric Evaluation:** A benchmark script must train the local warp network, compute geodesic distances via FMM, and verify that the learnable metric achieves a statistically significant improvement in recovering publisher editorial boundaries (NMI/ARI) over the flat Euclidean baseline.
- **Feedback Loop Test:** A simulation script must run the closed-loop pipeline and verify that $\alpha$ scales proportionally to Track 3 crack density, concentrating the observer mixture weights in high-friction regions.

---

## Acceptance Criteria

### Auto-Observer Generation
- [ ] The pipeline autonomously generates at least 8 contrastive hypothesis pairs from SVD axes without manual input.
- [ ] The generated hypotheses produce valid 3-class NLI outputs on DeBERTa forward passes.

### Learnable Metric & Geodesic Solver
- [ ] The learnable metric tensor network is trained using self-supervised contrastive loss (zero-shot with respect to publisher labels).
- [ ] Geodesic distances are successfully solved using Fast Marching Method (FMM).
- [ ] The learnable metric achieves higher publisher boundary recovery metrics (NMI and ARI) than both the Euclidean baseline and the static Riemannian baseline on the real corpus.

### Closed-Loop Feedback
- [ ] The Dirichlet concentration parameter $\alpha$ dynamically scales based on Track 3 topological crack density.
- [ ] The feedback loop concentrates the observer weights in high-friction regions.
</USER_REQUEST>
