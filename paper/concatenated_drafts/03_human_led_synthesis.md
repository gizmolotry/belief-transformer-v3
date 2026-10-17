

# Introduction and Literature Review

This paper outlines a zero-shot, observer-indexed NLI approach to mapping semantic geometry. We examine the classification frameworks driving contemporary news aggregation, using Ground News as our principal case study. We critique the "geometric fiction of objective neutrality"---the reification of a frame-dependent coordinate as an unmarked center of political discourse. Drawing on Kuhn and Goodman's work on the non-neutrality of classification, and Bowker \& Star on classificatory systems, we treat observer disagreement as a structured signal rather than noise. While category-theoretic and homotopy formalisms are introduced as theoretical analogies, we show how their structural predictions can be converted into testable geometric properties. Our methodology outlines the vertical tracks used to estimate perspectival shear, Dijkstra-solved geodesics, and Dirichlet annealing, testing these predictions empirically on a corpus of reporting on the Israel-Palestine conflict.

## Ground News and Horizontal Expansion

Ground News[^1] is a private news aggregator designed to reduce "bias," which it defines as a negative deviation from neutrality. The platform organizes articles on political subjects by mapping them onto a left-right spectrum relative to an assumed "middle" consensus. While marketed as a tool for critical thinking and pluralism, its sorting relies on publisher-level left-right labeling sourced from third parties. Ground News functions as a horizontal system: at each processing stage, it appends external metadata rather than analyzing content. For example, when processing a new article, the platform classifies it using a preconceived bias score assigned to the publisher. In contrast, endogenous classification relies entirely on features derived directly from the text.

Because it relies on publisher-level ratings from third-party outlets (such as AllSides, Ad Fontes, and Media Bias Fact Check), Ground News does not perform direct semantic analysis on individual articles. Instead, it aggregates publisher metadata, connecting articles through external global labels. This case study signals a broader methodological trend: systems that claim to expose bias often rely on prior classification frameworks external to the content they organize. This raises questions about whether these frameworks merely describe political reality or actively construct it by enforcing a predefined "middle" perspective [bowkerstar1999].

Formally, the horizontal pipeline can be described as a mapping $H(a) = (a,\, m(p(a)))$ where $a$ is an article, $p(\cdot)$ returns the publisher, and $m(\cdot)$ returns external metadata. The bias score becomes $b_{\text{hor}}(a) = \psi(m(p(a)))$ for some projection $\psi$---a function of publisher identity, never of content. The vertical pipeline proposed in this paper maps $x \mapsto \{z_o(x)\}_{o \in \mathcal{O}}$ where each $z_o(x) = E_\theta(x; h_o)$ is an observer-conditioned representation derived directly from the article's content under hypothesis $h_o$.

## The Philosophical Foundations of Observer-Dependent Semantics

Debates on journalistic objectivity directly inform semantic classification. Thomas Kuhn (kuhn1962) argued that observation is always theory-laden; data cannot be evaluated outside of an interpretive framework. Nelson Goodman (goodman1978) extended this, showing that measurement systems do not merely describe structure but construct the categories they map. To address this, interperspectival NLP models probabilistic distributions over interpretations rather than seeking a single "correct" label. This avoids the limitations of consensus aggregation, which suppresses minority viewpoints to construct a artificial consensus.

We will now consider the academic discourse that exists on exploring disagreement as a signal in and of itself rather than treating it as noise, particularly in relation to natural language inference and labeling tasks. frenda2024 explicitly frame perspectivist NLP as an alternative to collapsing annotator identities and disagreement into one canonical target, arguing instead that interpretive diversity can itself be methodologically informative for subjective phenomena. xujurgens2026 extend this in their work by treating disagreement as a widespread feature of NLP tasks that require its own separate modeling and evaluation framework.

Empirical studies of natural language inference (e.g., {pavlick2019}) show that even under strict guidelines, human semantic judgments remain unstable. Attempting to force agreement by over-specifying instructions can raise agreement scores, but often merely introduces the researchers' biases---a process aroyo2015 describe as "perfuming the agreement scores." This instability indicates that linguistic meaning is underdetermined; the same text can yield systematically distinct, internally coherent interpretations. Consequently, modern NLP increasingly rejects the assumption of a single ground-truth semantic coordinate.

The CrowdTruth method [aroyo2015] provides a foundation for this paper by demonstrating that disagreement is a valuable semantic signal. However, CrowdTruth maintains an epistemological divide between objective consensus and subjective disagreement, treating unresolved variation as noise. Natural language inference provides a way to bridge this divide by measuring semantic relations directly as interactions between text inputs, structured observer hypotheses, and model weights. While CrowdTruth exposes the flaws of consensus-seeking statistical reductions, our framework models these perspectival interactions directly, rather than filtering them through annotator sorting heuristics.

## Civility, Pluralism, and the Democratic Imperative

While journalistic ethics rejects complete epistemological relativism by encoding a "respect for facts" [ifj2019], consensus aggregators adopt moralistic premises that contradict the principles of liberal pluralism. As reiheld2013 points out, enforcing "exceptionless civility" suppresses the fundamental moral conflicts necessary for pluralism. Drawing on Isaiah Berlin's (berlin1998) value pluralism, Reiheld argues that civility projects often prioritize social management ("making society work") over respect for individuals. By sanitizing moral friction, consensus-oriented frameworks establish an artificial, civil center that excludes legitimate political antagonism.

Mapping political discourse onto a simple left-right spectrum introduces severe axial bias and methodological opacity. More fundamentally, consensus-seeking aggregators neutralize political antagonism, which weakens the conflict-driven processes essential to democracy [mouffe2000]. Real-world political debate is not a deviation from a neutral middle, but a series of irreducible conflicts that require representative containers rather than statistical erasure [swanstromweissmann2005].

## Representation, Projection, and the Limits of Classification

When analyzing semantic spaces, high-dimensional geometries must be projected onto lower-dimensional coordinates. Principal Component Analysis projects data onto its axes of maximum variance, whereas news aggregators heuristically compress political semantics into a single dimension using publisher metadata. The problem arises when this single dimension is reified as an objective, global representation of political bias. On a metaphysical level, Ludwig Wittgenstein argued that the perspective from which the world is described cannot be placed inside the description itself: "the subject does not belong to the world but it is a limit of the world" [wittgenstein1922].

In practice, the survey participants who supply AllSides and Ground News with bias ratings each project their own perspective, creating a local, observer-dependent shadow. Consensus aggregators inherit the intuition that multiple perspectives are valuable, but collapse them first across observers (via averaging) and then across dimensions (via a 1D scale). When combining Wittgenstein's (wittgenstein1922) point on the limits of representation, Bowker and Star's (bowkerstar1999) analysis of classification systems, and Kuhn's (kuhn1962) rejection of framework neutrality, these premises compose mereologically to yield a concatenatable inference: middle positions are not neutral baselines, but artifacts of a specific representational frame.

This inference admits algebraic demonstration. Define the Fréchet mean of a finite set of observer-conditioned representations $\{z_{o_1}(x), \ldots, z_{o_K}(x)\}$ as $\bar{z}(x) = \arg\min_z \sum_{k} d^2(z, z_{o_k}(x))$. The pairwise dispersion identity shows that $\sum_k d^2(\bar{z}, z_{o_k}) = \frac{1}{2K}\sum_{j \neq k} d^2(z_{o_j}, z_{o_k})$: the variance about the consensus centre is determined by the inter-observer disagreement. Any subsequent one-dimensional projection $\pi\colon \mathcal{Z} \to \mathbb{R}$ satisfies $\mathrm{Var}(\pi(\bar{z})) \leq \mathrm{Var}(\pi(z_{o_k}))$ for at most one $k$. A system that (i) averages across observer frames and (ii) projects to a single axis thereby executes a double variance collapse: first across observers, then across dimensions. The resulting scalar inherits no formal guarantee that it preserves the inter-observer distances that constituted the bias signal.

To test this algebraic proof empirically, we measured the variance collapse on the high-dimensional Qwen synthetic corpus ($N=60$ articles, $B=8$ observers, $H=1536$ dimensions). Averaging across the 8 observer representations to compute the consensus Fréchet mean erases $92.65\%$ of the original representation variance. Subsequently, projecting these consensus centers to a single dimension (the first Principal Component) collapses another $18.96\%$ of the remaining variance. In total, the cumulative double collapse erases $94.04\%$ of the original perspectival variance, leaving only $5.96\%$ of the semantic information intact. This demonstrates that one-dimensional aggregation maps do not merely simplify bias metrics, but fundamentally extinguish the primary information signals that constitute narrative disagreement.

This presents a crucial mismatch of units of analysis for systems like Ground News. In the following methodology we will discuss how our system attempts to account for this ontological discrepancy as well as the perspectival opacity and concatenation that hide the values which constitute discourse.

# Methodology

Our vertical pipeline is endogenous from its second layer: the first layer consists of raw text evaluated against predefined premise hypotheses using a pre-trained natural language inference model (DeBERTa-v2-xlarge-mnli). We use contrastive adversarial framing to model the Overton Window (the range of politically acceptable ideas) by mapping structural invariants and perspectival variations. By treating narrative bias as a geometric shape, we perform transformations to infer structural properties of the semantic space.

To map the Overton Window, we evaluate how semantic structure responds to observer-conditioned geometric warps. Rather than averaging embeddings into a single consensus vector, we construct independent manifolds from each observer's perspective. We establish a "Global Mean Manifold" as a baseline coordinate space. Instead of suppressing disagreement, we construct a joint manifold where perspectival divergence acts as the primary coordinate. Articles occupying similar topical coordinates in the base space are subsequently separated in the third dimension by the opposing forces of observer-indexed NLI responses.

By mapping independent manifolds and comparing them using Orthogonal Procrustes analysis, we quantify the structural divergence between different observer perspectives. For example, we can compare the same set of articles mapped under a pro-Israel frame versus a pro-Palestine frame. Procrustes analysis aligns these high-dimensional shapes through rigid rotation and translation without deforming the underlying manifolds, measuring the residual distance (the alignment error) as a metric of structural disagreement.

We deform the semantic manifold based on individual observer potentials before any consensus mixing occurs. Where an observer's perspective introduces ideological friction, the metric space between articles is stretched. Once these warped coordinate systems are constructed, we evaluate their alignments. Nodes that resist this perspectival shear are bound by shared semantic coherence, which we validate against shuffled, random, and constant controls.

## Adversarial Framing and the V-Observers

The proposed system takes one frozen NLI bot and gives them 8 different adversarial or contrastive hypotheses and prompts them to analyze any given number of articles. In this paper we refer to these different conditioned perspectives as `V-Observers'. In our implementation we used both real and synthetic articles relating to the Palestinian genocide. First, our real articles were scraped using a mix of curated RSS feeds and GDELT in parallel, filtering them for Palestinian related keywords, deduplicated by URL, extracted full article text, and then wrote the results to a JSON ledger. For our synthetic articles we attempted to generate a balanced corpus by shifting through 8 ideological clusters using a Qwen2.5-Coder-14B bot. Using this bot we prompted the local model to generate a number of articles constrained by perspectives, publication bias, and tone to the best of its ability. **Synthetic Generation and Caricature Bias:** A critical constraint of this synthetic corpus is that it relies on LLM-generated texts designed to fit specific ideological profiles. Consequently, the clear separations and high recovery metrics observed on synthetic data likely capture the LLM's stereotyped caricature of these political stances rather than the complex, subtle, and natural variations found in genuine political discourse. The synthetic results should therefore be interpreted as a mathematical validation of the pipeline's recovery capacity under idealized, caricature-distinct conditions, rather than a direct reflection of real-world editorial landscapes.

The mechanism of adversarial framing allows our approach to dialectically process each article under paired framing hypotheses rather than the raw average embeddings. Furthermore, the system as a whole is based on the idea that bias is not a static property but the measure of perspectival elasticity. For every article we query a series of hypothesis pairs designed to stress test the narrative structure of reporting on Palestine rather than for a particular ideological position.

    - **Pair 1: Israeli actions framing**:
    A: "This article portrays Israeli military actions as legitimate self-defense against terrorism." 

    B: "This article portrays Israeli military actions as disproportionate aggression against civilians."
    - **Pair 2: Palestinian framing**:
    A: "This article portrays Palestinians as innocent victims of occupation and violence." 

    B: "This article portrays Palestinians as violent aggressors and terrorists."
    - **Pair 3: Proportionality framing**:
    A: "This article describes Israeli responses as proportionate and measured." 

    B: "This article describes Israeli responses as excessive collective punishment."
    - **Pair 4: Resistance vs terrorism framing**:
    A: "This article frames Palestinian armed groups as legitimate resistance movements." 

    B: "This article frames Palestinian armed groups as terrorist organizations."
    - **Pair 5: Occupation framing**:
    A: "This article frames the situation as Israeli occupation of Palestinian territory." 

    B: "This article frames the situation as Israeli security measures against threats."
    - **Pair 6: Civilian casualties framing**:
    A: "This article suggests Israeli forces deliberately target civilians." 

    B: "This article suggests civilian casualties are unintended collateral damage."
    - **Pair 7: Historical context framing**:
    A: "This article frames Israel as a colonial settler state." 

    B: "This article frames Israel as the indigenous homeland of the Jewish people."
    - **Pair 8: International law framing**:
    A: "This article emphasizes Israeli violations of international law." 

    B: "This article emphasizes Israeli compliance with international law and right to self-defense."

By forcing the text to pass through these specific dialectical gates we can attempt to measure friction generated when a narrative is stretched between two opposing coordinated frameworks of reality.

Formally, each article $x$ passes through 16 hypothesis-conditioned forward passes (two per pair), each producing a three-class NLI verdict (entailment, neutral, contradiction). This yields a raw 48-dimensional logit tensor per article. The contrastive channel reduces to $8 \times 3 = 24$ values when differences are taken within each pair. In the notation established in the introduction, each hypothesis $h_o$ produces an observer-conditioned embedding $z_o(x) = E_\theta(x; h_o) \in \mathbb{R}^d$, and the vertical record for article $x$ is the family $V(x) = \{z_o(x), r_o(x, h_o)\}_{o \in \mathcal{O}}$.

Our architecture relies on cross-encoder NLI models (specifically DeBERTa-v2-xlarge-mnli) rather than traditional bi-encoders. Bi-encoders map texts to separate vectors independently, yielding geometrically shallow representations. DeBERTa processes the article and hypothesis simultaneously through joint attention layers. By decoupling content and position representations, DeBERTa preserves syntactic relationships (e.g., distinguishing "A attacked B" from "B attacked A"). Standard classifiers, such as a BERT sequence classifier, require fine-tuning on domain labels, which introduces prior classification biases. In contrast, NLI models evaluate general logical relationships (does hypothesis $H$ follow from text $T$?), enabling zero-shot, parameter-free semantic assessment without defining "bias" a priori.

Generative LLMs (e.g., GPT-4) are less suited for this pipeline. RLHF alignment makes their outputs structurally biased toward consensus, and their autoregressive nature introduces stochasticity. Interrogating semantic structures requires a deterministic representation space. Furthermore, generative models are trained to mimic patterns, whereas discriminative NLI models are optimized to categorize logical relations, making them more effective at detecting subtle semantic differences without collapsing them into expected textual continuations.

We treat the model's weights not as a source of truth, but as a frozen semantic substrate. By measuring how a text interacts with different prompted hypotheses, we isolate the variance induced by the observer frames. In this setup, DeBERTa's pre-trained weights act as a prior distribution, and the resulting coordinate shifts represent the structural alignment or resistance between the source text and the observer's worldview.

## The Anisotropy Problem

Transformer representations suffer from anisotropy: embeddings cluster in a narrow high-dimensional cone, rather than distributing uniformly across orthogonal dimensions [gabaix1999]. This representation degeneration compresses cosine similarities, making flat Euclidean metrics unreliable. Simply applying standard normalization techniques (e.g., centering or whitening) does not resolve the underlying issue, as the semantic relations remain embedded in a curved space that cannot be flattened without deforming the local neighborhood structures.

Simply scaling or dispersing these clustered vectors (like dispersing a dense cloud of points) does not alter the underlying directional concentration of the representations. The linear operations inside the attention heads map onto a curved semantic manifold, meaning that local distances are warped relative to the overall coordinate axes.

Standard models assume Euclidean spaces where the shortest path between any two concepts is a straight line. However, semantic relations are non-Euclidean and observer-dependent. Consider the relationship between "property" and "theft." Under an anarchist perspective, these concepts share a direct semantic link, while under a liberal perspective they are separated by structural barriers. Raw cosine similarity can measure topical orientation, but it fails to capture this observer-dependent traversability. To measure semantic paths, we must evaluate the topology under different observer-induced warps.

We avoid standard non-linear dimensionality reduction techniques like UMAP for diagnosing endogenous bias [mcinnes2018]. First, UMAP's neighborhood-preservation objective tends to bridge topological breaks, smoothing out the structural separations that define observer disagreement. Second, UMAP compresses the distance between opposing clusters to render the layout legible. In our context, this behavior is counterproductive: it collapses the separation between adversarial interpretations (e.g., framing an event as "resistance" versus "terrorism") into a single, averaged neighborhood. Rather than relying on non-linear visualization heuristics, we treat the observer-conditioned spaces as a manifold sheaf, preserving the local curvature and coordinate differences.

Conceptually the system is based on a Vertical Waterfall architecture. In line with aforementioned constraints, the system is organized into sequential tracks where the output of one layer may be extracted and analyzed while also serving as the input to the next layer. We will describe each layer as a `track'.

## Track 1: Logits

The first step of the pipeline is ingestion. We take a target article and run it through the eight adversarial NLI gates. As the article passes through each gate, the model measures the friction between the text and the prompted hypothesis. It outputs a three-dimensional score: entailment, neutral, and contradiction. In implementation, this becomes a raw 24-dimensional verdict channel, corresponding to eight NLI views multiplied by three NLI labels. These logits function strictly as the baseline diagnostic of the model's forward-pass verdict behavior. They are not treated as the main geometric object of the system and are kept separate from the CLS-derived observer geometry.

This tensor of scores provides low-resolution information but represents the final classification product of the NLI forward pass. It is useful precisely because it is comparatively simple: it tells us how the model distributes entailment pressure across the hypothesis gates before any of the later geometric operations. Other than projecting logits for comparison and diagnostic visualization, we do not treat Track 1 as the primary manifold. Track 1 is therefore best understood as the constitutional floor of the system: the raw verdict substrate against which later geometric transformations can be compared.

## Track 1.5: Spectral Polarity Extraction

The logits, as mentioned, do not give us very many specifics as to why the model made a decision for that particular article and hypothesis. A score of high entailment, for example, could reflect meaningful semantic alignment, or it could reflect the model reacting to a small set of vocabulary cues. To get a better picture, Track 1.5 measures the article's weighted resistance to reinterpretation by extracting the semantic shift induced by contrastive observer frames.

Earlier versions of the architecture considered a more computationally expensive backward-pass gradient probe, in which the model weights were frozen and the loss was differentiated with respect to the input representation. The current architecture uses a forward-pass approximation instead. For each article, the system extracts the CLS or mean-pooled observer representations generated by the paired hypotheses and computes the A-minus-B polarity deltas across the eight contrastive pairs. These deltas form an eight-way aggregate field of local semantic shear.

Formally, for contrastive pair $k \in \{1,\ldots,8\}$, let $\phi_A^{(k)}(x)$ and $\phi_B^{(k)}(x)$ denote the CLS-pooled representations under hypotheses $h_A^{(k)}$ and $h_B^{(k)}$ respectively. The polarity delta for pair $k$ is:

$$
\Delta_k(x) = \phi_A^{(k)}(x) - \phi_B^{(k)}(x)
$$

The matrix $[\Delta_1(x) \mid \cdots \mid \Delta_8(x)]$ encodes the full contrastive shear field for article $x$.

More simply, the "Polarity Delta" measures how the article's semantic position shifts when it is contrasted against opposing observer frames. Where the field is steep, the article is resistant to reinterpretation; where it is flatter, the article has more leeway of interpretation. This gives us two pieces of information: the direction of semantic shift, and the magnitude of stress required to move the article through the observer field.

We already know why logits are not enough on their own, but even raw forward-pass geometry is misleading. Mapping the articles directly in vector space inherits the anisotropy of the transformer representation, causing data to cluster together in ways that can obscure the very disagreement the system is meant to inspect. Specifically, Track 1.5 performs a spectral decomposition over these discrete contrastive delta matrices. Instead of assuming a manually labeled binary conflict and subtracting magnitudes by hand, we treat the eight contrastive pairs as an unstructured point cloud of observer-induced shifts.

Since our system relies on contrastive hypotheses, our bots do not have static personas, such as a simple "Pro-Israel" or "Pro-Palestine" bot. Instead, the system creates an interferometric field from the composite polarity vector space. If we used static personas, that would introduce an unnecessary layer of exogenous information: we would be the ones orchestrating the poles. The current field represents the net geometric tension generated when different contrastive topical frames exert high-dimensional divergence on the embedding. We define this as an Observer Shear Field. It does not simulate a conversation. It performs a structural stress test within the positional geometry of the model's forward-pass representation.

By performing Singular Value Decomposition, equivalently PCA on the centered contrastive delta matrices, the system endogenously discovers the primary axes of semantic stress without human intuition dictating which side of the conflict must dominate. In this sense, Track 1.5 does not tell the model what the ideological poles are. It asks which direction of observer-conditioned variation carries the most stable stress signal. Track 1.5 is the system's compass, not its judge. It discovers the dominant direction of semantic shear but does not decide what that shear morally means.

## Track 2: Geometric Expansion

As our baseline experiments indicated, computationally speaking, a flat surface is not the best container for semantic thought. Given that we have established the Spectral Polarity Delta as a superior measurement of ideological resistance, a natural question arises: why retain the geometric map at all? Why not simply apply the polarity delta to logits, using resistance to add more information to a confidence score and discarding embedding space entirely?

The answer is that the goal is not simply to score the narrative, but to traverse it. If we applied the deltas only to logits, we would reduce the article to a scalar or low-dimensional verdict profile. Such a score is a static judgment. It cannot tell us which other articles are nearby, whether a transition path exists, or how observer stress changes as the article moves through a field of related texts. A logit can have no neighbors and cannot occupy a position in a manifold. Track 2 supplies the coordinate container necessary to position Track 1.5 shear in relation to a latent topology.

This is the latent embedding that represents a specific observer-conditioned understanding of an article. More specifically, the magnitude and stability of these embeddings contribute to the semantic mass of an observation: high-confidence or structurally stable observations act as denser anchors in the manifold, while low-confidence or unstable observations act as lighter, more volatile points.

Unlike logits, which can be mathematically compared in a shared 24-dimensional verdict channel, embeddings have no automatic shared currency. The solution is to enforce a shared spatial geometry, making the observer views born-aligned in the same coordinate system. To resolve this, we employ Random Kitchen Sinks [rahimi2008], not merely to expand dimensionality but to approximate a kernel feature space in which different observer-conditioned embeddings can be compared as objects in a common Hilbert space.

The RKS approximation constructs an explicit finite-dimensional feature map:

$$
\hat{\phi}(x) = \sqrt{\frac{2}{D}}\left[\cos(\omega_1^\top x + b_1),\; \ldots,\; \cos(\omega_D^\top x + b_D)\right]
$$

where $\omega_j$ are sampled from the spectral distribution of the chosen kernel and $b_j \sim \mathrm{Uniform}[0, 2\pi]$. By Bochner's theorem [bochner1933], the inner product $\hat{\phi}(x)^\top \hat{\phi}(x')$ converges to the kernel evaluation $k(x, x')$ as $D \to \infty$.

This approach offers two advantages. First, standard kernel methods can imply a theoretical manifold without producing a directly navigable shape for our downstream system. RKS provides an explicit finite-dimensional approximation of that kernel space. Second, by using a shared basis across observer views, the system prevents each observer from inventing a private coordinate system. All observer projections pass through the same mathematical lens.

### Dimensional Adequacy and the Johnson--Lindenstrauss Lemma

The question of how much dimension is required to retain the relations of interest admits a standard answer. The Johnson--Lindenstrauss (JL) lemma states that for $0 < \varepsilon < 1$ and any finite set $X = \{x_1, \ldots, x_m\} \subset \mathbb{R}^d$, when

$$
k \geq 4\left(\frac{\varepsilon^2}{2} - \frac{\varepsilon^3}{3}\right)^{-1}\log m,
$$

there exists a map $f\colon \mathbb{R}^d \to \mathbb{R}^k$ such that for all $i, j$,

$$
(1 - \varepsilon)\|x_i - x_j\|_2^2 \leq \|f(x_i) - f(x_j)\|_2^2 \leq (1 + \varepsilon)\|x_i - x_j\|_2^2.
$$

Thus $k = O(\varepsilon^{-2} \log m)$ dimensions suffice to preserve all pairwise squared distances within controlled multiplicative distortion [johnsonlindenstrauss1984, dasguptagupta2003]. Lower-bound work shows this dependence is essentially optimal for linear maps [larsennelson2016]. The JL lemma is significant here as a *distortion budget*: a generic one-dimensional centre offers no comparable promise for many observer-conditioned semantic states. High-dimensional embeddings are methodologically necessary in the practical sense that they supply room for potentially independent semantic relations to remain distinguishable until evidence warrants reduction.

We experimented with whitening methods such as ZCA, which attempt to inflate the anisotropic transformer cone while preserving the original rotation of the data. In practice, whitening was not always necessary and could be counterproductive, because it sometimes destroyed local disagreement texture. This is especially important because Track 1.5 already uses spectral analysis to identify semantically relevant directions of variation. Whitening can make the space cleaner while making the disagreement less legible.

The system supports several kernel families, including RBF, IMQ, and Mat\'{e}rn. The Mat\'{e}rn implementation samples from a Student-t spectral distribution, where the smoothness parameter controls the heaviness of the tails. Conceptually, this was motivated by the idea that heavy-tailed spectral sampling could preserve high-frequency semantic features and topological barriers that a Gaussian kernel might smooth over [rasmussen2006]. However, this should be treated as an ablation condition rather than a proven universal advantage. The current evidence supports testing kernels comparatively, not declaring one kernel as intrinsically superior.

We formally qualify the interaction between geometry and observer stress through the Observer Metric Tensor. While standard embeddings imply a Euclidean distance between concepts, the kernel-approximated manifold allows observer shear to warp the effective distance between articles. Conceptually, we can write this as:

$$
g_{\mu\nu}^{\text{total}}(x) = \frac{1}{\rho(x)}\delta_{\mu\nu} + \nabla_\mu \Phi\, \nabla_\nu \Phi
$$

Here, $\delta_{\mu\nu}$ represents the base content-to-content geometry supplied by Track 2, while $\nabla_\mu \Phi\, \nabla_\nu \Phi$ represents semantic shear from Track 1.5. Track 3 later supplies the conformal density term $\rho(x)$. Operationally, the production Track 5 path implements this through kernel fusion and conformal scaling, while the stricter Riemannian implementation exists as an ablation branch. This means the equation should be understood as the conceptual scaffold for the architecture rather than a claim that every visualization directly solves a continuous geodesic field.

The polarity delta is the force vector of observer shear stress. In regions of consensus, the delta may be flatter and the metric resembles a more ordinary Euclidean space. Near a structural singularity, the shear magnitude increases and the effective distance between neighboring articles can stretch. Though the V-Observers are born-aligned by way of RKS, the Observer Polarity Delta from Track 1.5 acts as a local warping force inside the shared geometric container.

This is where we unlock a defined physical manifestation of the aforementioned singularities. Consensus manifests as constructive wave interference, where different observer-conditioned waveforms align to create a higher-amplitude, lower-resistance region. The bridge is defined by the constructive alignment of article density in the geometric manifold and reduced observer shear. The topological vocabulary used throughout the system emerges from two orthogonal quantities:

*The Landscape*: this relates to existence. It asks whether a geometric path exists between articles in the shared manifold. Physically, it is derived from Track 2 content-to-content structure.

*The Polarity Delta*: this relates to resistance. It asks whether the path is actually traversable under observer stress. Physically, it is derived from Track 1.5 content-to-observer interference.

Together these produce four terrain modifiers:

    - **The Bridge**: (High Density, Low Gradient): constructive interference where the system detects many similar articles occupying one region, and the V-Observers are in relative perspectival agreement.
    - **The Swamp**: (High Density, High Gradient): phase incoherence. There is a high volume of related articles, but the V-Observers are out of phase. The base geometric path exists, but it is difficult to navigate cleanly because observer shear is high.
    - **The Tightrope**: (Low Density, Low Gradient): brittle resonance. The V-Observers have comparatively high perspective rapport, but the source-text landscape is sparse. If the path deviates even slightly, it risks falling into the void.
    - **The Void**: (Low Density, High Gradient): sparse geometry combined with incoherent observer stress. Here the coordinate system is not simply empty; it is weakly supported and high-resistance.

This completes the geometric manifold, Track 2. Defining a map, however, is not the same as making it navigable. We have the slope and the plane, but we still need an explicit density validator.

## Track 3: Dirichlet Sweep

If Track 2 provides a map, Track 3 provides a lens. Having established a topological floor for the manifold, we face another limitation: we cannot possibly account for every rhetorical frame in existence. A system with 10,000 bots is computationally unfeasible and would be incoherent. Because of this, early versions of Track 3 were motivated by the need to blur the lens and estimate missing dialectic structure between frames. This was not the final role of Track 3.

The original intuition that we needed to smooth the gaps between observers rested on a dangerous Euclidean assumption. Inside the model, there is no guaranteed continuous plane where the distance between every two points is meaningful. If we rely only on ordinary Euclidean metrics, we demand an indestructible field that hallucinates continuity where the model may know nothing. As a result, Track 3 becomes a density scanner: gaps are not smoothed away, but treated as evidence about the substance or fragility of semantic support.

Consider the dialectic of International Law. One pole emphasizes violations, while the other emphasizes compliance. Scenario A: an article discusses a border skirmish using language that strongly activates both frames, increasing tension. Scenario B: an article offers diplomatic platitudes without citing specific legal frameworks or rhetorical devices, causing both poles to activate weakly. In both cases, a simple midpoint location may look similar. But the first midpoint is the result of strong opposing forces, while the second is the result of weak semantic substance. A snapshot of the tug-of-war cannot tell us which case we are seeing. Track 3 asks how stable the article is under pressure.

Mechanically, Track 3 uses a dynamically optimized weight simplex over RKHS-projected observer views. Instead of treating the observer outputs as one softmax verdict, we treat the eight observer views as a simplex of possible mixtures. With high alpha, the system is forced toward consensus: probability mass is spread broadly across observer views. With low alpha, the system is allowed to fracture: probability mass can collapse into sharper observer-specific spikes.

This is the algorithmic source of the tug-of-war metaphor. The article begins near a high-entropy barycenter. Over sequential annealing stages, the system relaxes the constraint and records whether the observer mixture holds together or tears apart. Functionally, this acts as KL-regularized simplex-weight annealing over RKHS-projected bot perspectives. The KL term binds each stage to the previous stage, giving the system inertia. Without this term, each alpha interval would be an independent snapshot. With it, Track 3 measures the integrity of a path through observer-mixture space.

The implementation is best described as hot-to-cold annealing. The system begins in a high-alpha consensus state with strong prior regularization and weak separation pressure. It then cools toward lower alpha values, where prior constraint relaxes and structured observer separation can emerge. The variance term encourages observer mixtures to pull apart, while KL divergence binds the process to the previous stage. This treats the eight-component simplex of attention allocated to the observer gates as a dynamic state variable rather than a static hyperparameter.

Concretely, the annealing objective can be summarized as a competition between separation and inertia:

$$
L_t(w_t) = -\alpha_t^{-1}\,\mathrm{Var}\left(\sum_b w_{t,b}\phi_b(x)\right) + \lambda_{KL}\alpha_t\,\mathrm{KL}(w_t\|w_{t-1})
$$

where $w_t \in \Delta^7$ is the simplex-constrained observer weight vector at annealing stage $t$, $\phi_b(x)$ is the RKHS-projected semantic embedding contributed by observer $b$, and $\alpha_t$ controls the annealing schedule.

From this trajectory, the system derives two important quantities. The first is adaptive work:

$$
W_{\text{anneal}} = \sum_{t=0}^{T-1} \|w_{t+1} - w_t\|
$$

The second is the Track 3 conformal density:

$$
\rho_i = \frac{1}{1 + \tau \|b_i\|}
$$

where $b_i$ is the Track 3 fused variance for article $i$, and $\tau$ is the temperature.

Track 3 therefore does not add another raw coordinate block to the final Track 5 representation. It produces a conformal density field used to validate and scale the fused manifold. Articles that maintain their observer-mixture position under annealing produce denser, more stable local geometry. Articles that tear away from the consensus position produce lower-density, higher-uncertainty regions. Track 3 is not a third map; it is the pressure test that tells us whether the map is physically trustworthy.

### Homotopy Type Theory Interpretation

The Dirichlet sweep admits a natural reading through Homotopy Type Theory (HoTT). In HoTT, identity between two objects need not be a binary proposition; it can be witnessed by paths, and paths between paths form higher structure [hottbook2013]. Applied cautiously to Track 3, an article whose observer-mixture position remains stable under annealing corresponds to a semantic identity that admits a continuous path through observer-mixture space. An article that tears apart under cooling corresponds to a path that cannot be continuously deformed without breaking---a topological obstruction.

This interpretation is offered as a formal analogy and research program, not as a claim that the Dirichlet simplex implements genuine HoTT identity types. The computational substrate is finite-dimensional optimisation on a probability simplex; the HoTT language disciplines the questions we ask of the output: Are these two observer-conditioned representations connected by a path? Does that path survive deformation? Where do paths break?

## Track 4: Traversal Telemetry

In attempting to operationalize the information geometry of the fused manifold, Track 4 simulates particles across the article graph based on density, stress, and metric distance. Its purpose is to measure the work required to make semantic leaps between article nodes. Mechanically, Track 4 is best described as a thermodynamic stochastic walk over a metric-respecting article graph. It is Markovian because each transition depends on the current article, local neighbors, walker temperature, and metric resistance. It is MCMC-inspired, but it should not be described as a full posterior sampler, because the current implementation does not define a target posterior distribution and then prove asymptotic convergence through accept-reject sampling.

Take our dataset $E \in \mathbb{R}^{N \times H}$, where $N$ is the number of articles and $H$ is the embedding dimension of the fused representation. At step $t$, a walker is at article $i$. The system evaluates the local neighborhood of $k$ closest articles using the metric graph constructed from Track 2 geometry, Track 1.5 stress, and Track 3 density. For each possible neighbor $j$, it calculates a transition weight:

$$
w_{ij} = \exp\left(-\frac{d_g(x_i,x_j)}{\tau}\right)
$$

where $d_g$ is the discrete metric edge cost induced by the fused geometry.

We condition two types of walkers: Cold Walkers and Hot Walkers. Cold Walkers have lower temperature and are more rigidly attached to the boundaries of the manifold. Hot Walkers use a higher temperature multiplier, allowing them to tunnel through higher-stress regions. As the walker explores, it accumulates work:

$$
W_{\text{total}} = \sum_{t=0}^{T-1} d_g(x_t, x_{t+1})
$$

When accumulated work reaches the cognitive horizon, the walker flips into retreat mode and is penalized toward the anchor coordinate:

$$
D_{j \to A} = d_g(x_j, x_A)
$$

The resulting retreat transition weight is:

$$
w_{ij} = \exp\left(-\frac{d_g(x_i,x_j)}{\tau}\right) \cdot \exp\left(-\gamma\, d_g(x_j, x_A)\right)
$$

The key architectural boundary is that Track 4 does not assign final semantic verdicts. It exports mechanical telemetry: high-dimensional path coordinates, the work integral of each trajectory, a closed-loop survival boolean, and Markov diagnostics such as committor probability, mean first passage time, and reactive flux when boundary states are available. Whether a high-work or failed-loop path should be interpreted as semantically important belongs downstream in Track 5 or the evaluation layer. Track 4 tells us how the traversal behaved; it does not decide what the article is.

Current anchor selection is also terrain-aware. Rather than selecting only the highest-stress articles, the system selects a small set of polarity anchors using zone-constrained farthest-first logic, attempting to include Bridge and Void anchors when available and then a third distinct terrain region. For each anchor, the current swarm runs five walkers: two hot and three cold.

## Track 5: Synthesis

Track 5 is the synthesis layer of the architecture. Earlier iterations of the Semantic Interferometer relied on concatenating all tracks, simply treating all raw feature vectors as another set of coordinates. This failed because Track 2 geometry, Track 1.5 shear, and Track 3 density do not represent the same kind of object. Concatenation forces coordinate features, directional stress, and density/uncertainty into one flat vector space, which collapses the mathematical distinction the architecture is designed to preserve.

The final architecture relies on kernel and metric fusion rather than raw concatenation. In the Hadamard branch, Track 2 and Track 1.5 are converted into similarity kernels and multiplied element-wise:

$$
K_{\text{fused}} = K_{T2} \odot K_{T1.5}
$$

Track 3 is then used as a conformal density normalizer:

$$
D_{\text{conformal}}(i,j) = \frac{D_{\text{fused}}(i,j)}{\sqrt{\rho_i \rho_j}}
$$

where $\rho_i$ and $\rho_j$ are the Track 3 densities. Low-density articles stretch local distances, while high-density articles compress them.

The system also includes a stricter Riemannian ablation branch. In that branch, Track 2 acts as the base geometry container, Track 1.5 enters as the shear or anisotropy term, and Track 3 enters as the conformal density term. A pairwise metric-respecting distance matrix is built first, then embedded back into a vector space for downstream compatibility.

The crucial correction is that Track 3 is not multiplied onto final 2D or 3D visualization coordinates. Doing so would merely push points radially away from an arbitrary plotting origin and destroy local topology. Instead, Track 3 modifies pairwise distance or kernel structure before visualization. The visible manifold is therefore a rendered projection of a fused structure, not the mathematical object itself.

Track 5 is the interferometer proper. Track 2 provides where articles are, Track 1.5 provides how observer stress pulls them, and Track 3 provides how much local semantic substance exists. Track 5 is where those separate physical meanings become one testable manifold.

## Methodological Constraints and Interpretive Limits

Several distinctions must be maintained to prevent the formal language from overstating what the implementation computes. Random Kitchen Sinks approximate a kernel inner product; they do not by themselves guarantee the distance-preservation properties of the Johnson--Lindenstrauss lemma, which applies to a different class of random projections. HoTT identity paths are a formal analogy for the Dirichlet annealing trajectory; the simplex optimisation does not construct identity types in the type-theoretic sense. Procrustes alignment produces a discrete orthogonal rotation between two finite point clouds; it does not automatically induce a continuous path or a sheaf-theoretic transport map.

These distinctions matter because conflating them would manufacture false mathematical authority. The architecture computes kernel similarities, simplex weights, pairwise distances, and orthogonal alignments. The formal language of HoTT, topos theory, and Riemannian geometry organises the questions we ask of those computations---What is local? What is transportable? Which paths are distinct?---without substituting for the empirical answers.

Additionally, the entire system is conditioned on the eight hypothesis pairs specified in this section. The contrastive NLI entailment set is not derived from the data; it is authored. If these hypotheses were reformulated, the resulting manifold geometry would change. This is not a flaw---it is a consequence of verticality. The hypotheses are declared interventions, not transparent windows. Their influence is indexed and available for inspection rather than hidden inside a source-level metadata map.

# Results
If bias is the embedded signal we were hunting for, there were a number of noisy red
herrings that attempted to corrupt the system and lead to erroneous results. First, the
aforementioned anisotropy trap; models group things together if they talk about the same things,
which we solve in Track 1.5. Second is the Euclidean hallucinations that accompany methods
like UMAP, which we solved using the RKS kernel trick. The third is barycentric consensus, or
the 'Ground News effect' where the classifier is geared towards finding a 'golden middle'. This
we address in Track 3 by ripping apart polite consensus, defining the middle path as the noisy
compromise.
Using our Procrustes analysis we can compare real, random, constant,
and shuffled versions of our inputs to estimate whether the resulting geometry separates from
matched controls. In the submitted capstone run, the shuffled control yielded a distance of 1.313
against the real manifold, while the random control yielded a distance of 1.302. These distances
should be interpreted as evidence of geometric separation under that run configuration rather
than as a universal proof that the system has isolated semantic bias. The stronger claim is that
the manifold contains a non-trivial structure that can be compared against stochastic and
matched controls, and that the direction and magnitude of separation must be reported by kernel,
control family, seed, and metric basis.

## Procrustes and Gauge Alignment
Across the formal alignment evaluation, the Procrustes pass rate was $1.000$, with a mean ratio of $2.624$. The maximum observed rotation was $113.68^{\circ}$, quantifying the extent to which gauge alignment must accommodate changes in orientation while preserving the comparative geometry.

## Synthetic Ground-Truth Recovery
On synthetic data with known ground truth, recovery achieved a mean normalized mutual information (NMI) of $0.7433$ and a mean adjusted Rand index (ARI) of $0.4836$. These complementary measures report substantial recovery of the planted structure while retaining sensitivity to disagreement at the level of individual assignments.

All this said, a fundamental sensitivity here is the contrastive NLI entailment and
hypotheses set. Unavoidably, the entire system is defined by these inputs. If they could somehow

be auto-formalised on inference, the system might serve as a much more reliable Semantic
Interferometer.

## Observer, Metric, and Dynamic Tuning Upgrades
The auto-observers upgrade showed that endogenous pairs captured a higher variance of $2.928 \times 10^{-4}$ than the $2.922 \times 10^{-4}$ captured by static pairs. Although the difference is narrow, its direction indicates that observer pairs derived from the material itself retain more of the variance available to the measurement process.

The metric tensor upgrade produced a sharper distinction in traversal cost. The Riemannian fast marching method (FMM) walker required $3.313$ work, whereas Euclidean transport required only $1.214$ work. This $+172\%$ increase demonstrates that the learned space reflects ideological friction that is absent from a purely Euclidean account of transport.

Closed-loop scaling of the Dirichlet parameter $\alpha$ reduced mean crack density from $0.479$ to $0.393$, while increasing observer sparsity from $0.223$ to $0.277$. The simultaneous movement of these measures shows that dynamic tuning can concentrate observer structure while reducing discontinuity in the resulting manifold.

## Reflector Robustness
Early coordinate fusion failed the specificity tests, passing $0/9$. Semantic-Use Gating, however, achieved perfect artifact robustness across $12/12$ cells, successfully distinguishing between a text that {uses} a frame and one that merely {mentions} it. This distinction isolates semantic deployment from superficial reference and prevents frame vocabulary alone from determining the result.

Figure 1: This is our Global Mean observer manifold snapshot of a group of 30 synthetically
generated articles, displaying the foundational coordinate prior to targeted observer
displacement.

Figure 2: This visualisation illustrates the system's ability to use the 16 contrastive probes to
discover relevant axes. Rather than enforcing manual labels, the architecture uses Singular Value
Decomposition on the resulting matrices to discover the principal axes of the discourse. Notice
here that the nature of our dialectal prompting likely inspired a generally two sided conflict

where the main axis leads one direction.
Figure 3: This rendering demonstrates the path mechanics and constraints of the Semantic
Interferometer. This is a sequence of walker trajectories generated from a selected polarity
anchor on the far right of the image. The path color indicates high traversal work and instability
in the path telemetry, not a final Track 4 semantic verdict.
5 Polarity anchors represent selected high-structure or high-shear articles used to probe traversal behavior.

## Track 4 Work Barrier
Across the Track 4 comparison, the mean real-versus-control work gap was $284.28$. The magnitude of this barrier provides direct empirical evidence that traversal through the measured geometry requires substantially different work in the real condition than in its matched control.

The starting path requires disproportionately higher energy relative to its manifold placement. In
the language of the traversal layer, the walker's accumulated fatigue exceeded the thermal
threshold before it could close a stable loop.
Figure 4: Consider the following synthetic article labeled 'Liberal Zionist'. It sits on a ridge far
from the center on this set, indicating that both Track 2 kernel outputs and Track 1.5 Polarity
perceive the article as further outside the discourse, suggesting it has both perspectival shear and

a geometrically distinct topical embedding. Articles near the center on this given set are largely
consistent of Western Leftists and Islamic Resistance framings.
Figure 5: This demonstrates the system's observer-centered visualization mode. A selected
article can be treated as the local reference point for a centered manifold view, allowing us to
inspect how article neighborhoods, observer displacement, and local stress differ relative to that
anchor. The claim is not that the model literally adopts the article's opinions, but that the
geometry can be re-expressed around an article-conditioned reference frame. In this view, the
Mean Manifold and an Article Observer Manifold may cluster differently, revealing changes in
local alignment, displacement, and observer shear.
Figure 6: This snapshot records the system's waterfall architecture and how the NMI score of the
output fluctuates but remains stable throughout the inference despite the number of
transformations applied throughout the architecture.

# Discussion

Where guidelines constrain human annotation workers, hypotheses in NLI constrains bots internal activation pattern and attention allocation, shaping which semantic relations are made salient during inference. In this sense the NLI hypothesis is likely bound to similar methodological realities; the hypothesis can serve to safeguard the model from its own bias but would likely not improve the signal or enhance its `neutrality'. Where disagreement among workers can serve as a useful statistical signal, agreement in NLI amounts to consistent outputs across hypotheses. This all suggests, however, that consensus, whether among human annotators or across model outputs, is not an output with enough definition to indicate semantic clarity. In both cases the apparent structure may be the results of the probing framework rather than the structure of the text itself. Because of this, the critical variable here is not the presence or absence of agreement, but the conditions under which agreement is produced and maintained.

This is where the unique advantage of an observer based NLI emerges. Beyond scale, the system allows for controlled variation of interpretive frames within a shared representational space, allowing for a form of latent space tomography where it is possible to interrogate the nature of the represented signal of semantics itself rather than a labeled profile. The question then shifts, then, from a matter of more complicated statistical analysis of labeled profiles to direct analysis of the geometric of perspectival variation under controlled conditions.

The system in this paper attempts to begin precisely at this point, asking whether disagreement can be evaluated beyond non-linear profiling statistics as a series of transformations applied within a shared semantic space. In CrowdTruth, judgement happens upstream, based on questions such as: is this disagreement an artifact of ambiguity? Is this bad worker behavior? Task design? In Semantic Interferometer the statistical adjudication partially occurs further downstream, as all NLI observer instances are allowed to generate all their respective outputs without supervision. Only once those perspectives are injected into a shared representational space does the system emergently discover perspectival shear, quantifying the relevance of the different observer perspectives as mathematical derivations.

Where in CrowdTruth disagreement is portrayed as an artifact of the object itself, the architecture proposed in this paper attempts to acknowledge the inherent limits of measuring semantic shadows. Using the collection of several AI `mindspace' projects collided into a single object, the Semantic Interferometer attempts to represent disagreement as the repeated structure of the shadows statistically as they build a complex shape. Disagreement can become a question of coherence, stability, asymmetry, density, and path cost over a series of perspectival perturbations, rather than noise.

The vertical pipeline operationalizes this through five distinct physical interrogation types:

    - *Coherence*: Do observer-conditioned representations preserve neighborhood structure under transport? (Naturality test, Track 5)
    - *Stability*: Does the observer-mixture position survive Dirichlet annealing, or does it fracture? (Track 3 conformal density)
    - *Asymmetry*: Are the magnitudes of observer displacement symmetric or skewed? (Track 1.5 polarity delta)
    - *Density*: Is there enough semantic substance to support the geometric path, or is the region a void? (Track 3 $\rho_i$, terrain classification)
    - *Path cost*: How much thermodynamic work is required to traverse from one article to another? (Track 4 work integral)

Each of these is a computable quantity, not a philosophical posture. The distinction matters: disagreement becomes amenable to differential diagnosis rather than binary classification as signal or noise.

The Semantic Interferometer was designed to mathematically explore relativism in journalistic semantic analysis and demonstrate the failures of objective neutrality. Prolonged interaction with the foundational mechanism of natural language processing, particularly in relation to high dimensional embeddings and their inherent anisotropic properties, prompted a methodological and conceptual drift. With each dead end (from pre-labeled signed axis to raw embedding concatenation to GPA procrustes sequencing) the constraints of current NLP architectures repeatedly demonstrated that treating semantics as perfectly sociologically fluid did not align with the statistical reality of the learned representation space. Every attempt to use arbitration as a means to force the text into a certain framework while transposing that frame heuristically failed on an empirical basis. Crucially, these failures were not evidence of relativistic uncertainty but rather suggested structural resistance of the vector space in relation to its pairwise perspective. An assortment of approaches in creating a Semantic Interferometer failed largely because they relied on the infinite malleability of encoded text when the topology of the specific model's latent space demands to be respected even in a vacuum.

The issue is not that Ground News reduces the complexity of semantics into a one dimensional line, which is a forgivable inevitability; the system described in this paper works similarly, producing an annotated rank-1 projection, or a three dimensional object. Our system attempts to indicate how the very concept of the Overton Window, the unsaid bedrock on which Ground News sits, is massively underdefined and interrogated in such a way that inhibits a more causal understanding of observer dependent semantics. Through this lens Ground News goes beyond revealing bias and actively operationalizes a preconceived center by treating the axis as a static object, when it is not. The preceding results demonstrate how the Semantic Interferometer accounts for an inherently frame-dependent Overton Window through geometric, topological, and sheaf-theoretic diagnostics applied to both real and synthetic corpora.

## Observer Transport and the Impossibility of a Neutral Basepoint

The results reported in the preceding section crystallise the philosophical argument into mathematical form. The Procrustes alignment between any two observer frames $o$ and $o'$ produces an orthogonal rotation $R_{o \to o'}$ and a residual divergence $\delta(o, o')$. When $\delta$ is near zero, the two frames agree on local geometry; when it is large, they disagree structurally.

In the language of Homotopy Type Theory [hottbook2013], a transport map between observer frames is a function $T_{o \to o'}\colon \mathcal{Z}_o \to \mathcal{Z}_{o'}$ that carries semantic identifications in one frame to identifications in another. The equivariance requirement is that transport should commute with the neighbourhood operation:

$$
\mathrm{NN}_k(T_{o \to o'}(z_o(x))) = T_{o \to o'}(\mathrm{NN}_k(z_o(x)))
$$

The Naturality test (Section 3.5) measures exactly this: the Jaccard overlap between pre-transport and post-transport $k$-nearest-neighbour sets. A score of $\sim 0.99$ for the real corpus indicates that transport approximately commutes with local structure---but only between pairs of non-degenerate frames. The constant control's $\sim 0.54$ score shows that degenerate frames destroy the commutation entirely.

A "neutral" centre would require a distinguished frame $o^*$ such that $\delta(o^*, o) = 0$ for all $o \in \mathcal{O}$. But the Procrustes residuals are non-zero and vary by kernel and seed: the mean coordinate displacement is $0.174$ with a maximum rotation of $113.68°$. No frame simultaneously minimises residual divergence with all other frames. The "centre" is not a position but a frame-dependent artefact.

## Limitations and Research Programme

Several boundary conditions constrain the scope of the claims. First, the architecture is conditioned on eight manually authored hypothesis pairs. These pairs were designed to stress-test the Palestine discourse specifically; they are not automatically transferable to other domains without re-authoring.

Second, the formal language of HoTT, topos theory, and Riemannian geometry is no longer purely analogical. Three of the four formalisms invoked in this paper have been partially operationalized and empirically tested:

    - *Riemannian geometry*: The observer metric tensor $g_{\mu\nu}^{\text{total}}$ is discretized on the $k$-NN graph and solved via Dijkstra. Geodesic distances under observer shear ($\lambda > 0$) improve publisher clustering recovery over the flat Euclidean baseline, with statistical significance confirmed by permutation test ($p < 0.001$; Section~[sec:riemannian-solver]).
    - *Sheaf theory*: The cocycle consistency of Procrustes transport maps was evaluated over all 336 observer triples. The mean residual of $1.1225$ establishes path-dependent holonomy---observer transport does not satisfy the gluing axiom, confirming genuine local curvature in the observer bundle (Section~[sec:sheaf-cocycle]).
    - *Homotopy Type Theory*: Topological cracks under Dirichlet annealing correlate significantly with perspectival friction ($t = 54.88$, $p \approx 0$), validating the prediction that annealing-path obstructions correspond to regions of high observer conflict (Section~[sec:hott-verification]).

Topos-theoretic constructions (subobject classifiers, categorical composition of observer functors) remain aspirational and are deferred to the research programme.

Third, certain empirical results remain mixed. The Track~4 terrain traversal classification achieves only an $8.3\%$ pass rate, suggesting that mapping continuous metric paths onto discrete typologies requires supervised calibration. The Naturality metric does not discriminate real from shuffled corpora; it discriminates structured from degenerate embeddings. These partial results are reported transparently rather than suppressed or reframed. Additionally, the absolute effect sizes for the Riemannian geodesic clustering improvement are modest ($\Delta\text{NMI} \leq 0.018$, $\Delta\text{ARI} \leq 0.046$) and the corpus is limited to $N = 80$ articles from a single conflict domain. Validation on larger, multi-topic corpora is required before generalisability can be claimed.

The research programme that follows is fourfold: (i)~auto-formalisation of hypothesis pairs from corpus statistics, removing the manual authorship bottleneck; (ii)~scaling the evaluation to multi-topic corpora (e.g., world economics, climate policy) with larger sample sizes; (iii)~computing discrete Ricci curvature (Ollivier or Forman) on the metric graph to provide pointwise curvature diagnostics that complement the global holonomy signal; and (iv)~tighter integration of the category-theoretic and homotopy formalisms with the computational substrate, moving toward genuine categorical implementation of sheaf-theoretic semantic transport.

## Conclusion

The central thesis of this paper is not that Ground News is wrong, but that the geometric fiction of an objective centre---a frame-independent neutral position from which all bias can be measured---is mathematically untenable. The Semantic Interferometer does not discover the centre; it demonstrates that there is no centre to discover. What exists instead is a family of observer-conditioned manifolds whose structural invariants survive Procrustes alignment and whose perspectival variations encode the very disagreement that horizontal systems collapse.

Middle positions are not neutral points but the effects of a particular representational frame onto an object. The Semantic Interferometer attempts to make this frame-dependence inspectable rather than invisible.

## Footnotes

[^1]: Website link: \url{https://ground.news/}
