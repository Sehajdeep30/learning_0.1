# AI/ML Coverage Map

A dashboard, not a syllabus. Never work through it in order. Its only job is to make gaps visible so each learning cycle can turn a weak cell into a strong one.

## How to use this

**Levels (fill the Lvl column honestly, edit it in place):**

- **0**: never seen it
- **1**: can recognize it and say roughly what it is
- **2**: can explain it from memory, without notes, to a colleague
- **3**: can derive or defend it, and say when it fails and what the alternative is
- **4**: have used it in practice and been surprised by it at least once

**Tiers:**

- **★** core to your niches (efficient ML, evaluation) or to your past projects. Get these to level 3.
- **○** supporting. Level 2 is enough unless a question pulls you deeper.
- **◇** research hedge (interpretability, robustness). Slow burn, one cycle every quarter or two.

**Rules of the dashboard:**

1. Every cycle must move at least one cell up a level. Write the date next to the change.
2. When choosing the next question, prefer cells that are both low level and ★.
3. A cell is only level 3 if you did the check without looking. The "Can you..." column is the test.
4. Re-grade everything every 6-8 weeks. Levels drop when you forget, and that is useful information.
5. Anything you hit while reading that is not on this map gets added as a new row. The map grows with you.

**Fast start (30 minutes):** go through all rows once and put a number in Lvl. Do not look anything up. Your first honest grading is the most valuable data in this document. Then pick the five lowest-scoring ★ rows. Those seed your first question.

---

## A. Mathematical foundations

The reason this matters: every later topic (LoRA, PCA, distillation, quantization error, calibration) is a math idea wearing a costume. Fix the idea once and it pays off everywhere.

### A1. Linear algebra

| # | Topic | Can you... | Tier | Lvl |
| --- | --- | --- | --- | --- |
| A1.1 | Vectors, dot product, norms (L1, L2, Linf) | Explain why the dot product measures alignment, and why L1 penalties give sparsity while L2 gives shrinkage | ○ | 1 |
| A1.2 | Matrix as linear map, matrix multiplication | Describe a weight matrix as a transformation of space, and count the FLOPs of a matmul | ★ | 0 |
| A1.3 | Rank, column space, null space | Say what low rank means geometrically and why a rank-r update has far fewer parameters | ★ | 1 |
| A1.4 | Eigenvalues and eigenvectors | Explain what an eigenvector is, why symmetric matrices have real eigenvalues, and where this shows up (covariance, Hessian) | ★ | 1 |
| A1.5 | SVD | Derive the best rank-k approximation (Eckart-Young idea) and connect it to PCA, LoRA, and weight compression | ★ | 0 |
| A1.6 | Projections and orthogonality | Explain least squares as projection and why orthogonal bases are numerically nice | ○ | 1 |
| A1.7 | PCA | Say what it maximizes, what it assumes, and when it fails (nonlinear structure, scale sensitivity) | ★ | 1 |
| A1.8 | Condition number and numerical stability | Explain why ill-conditioned problems amplify error and how this relates to low-precision arithmetic | ★ | 0 |
| A1.9 | Matrix norms, spectral norm | Explain why the largest singular value bounds how much a layer can amplify its input | ◇ | 0 |

**Boss question:** why can a 4096x4096 weight update often be captured by a rank-8 matrix, and what would it mean if it could not?

### A2. Probability and statistics

| # | Topic | Can you... | Tier | Lvl |
| --- | --- | --- | --- | --- |
| A2.1 | Random variables, expectation, variance, covariance | Compute and interpret them, and know variance of a sum with correlated terms | ○ | 2 |
| A2.2 | Common distributions (Gaussian, Bernoulli, categorical, Poisson, heavy tails) | Say which fits what and why heavy tails break mean-based thinking | ○ | 1 |
| A2.3 | Bayes' rule, priors, posteriors | Do a worked example and explain what a prior contributes | ○ | 0 |
| A2.4 | MLE vs MAP | Show that MAP with a Gaussian prior is L2 regularization, and cross-entropy is MLE for classification | ★ | 0 |
| A2.5 | Law of large numbers, CLT | Explain why sample means are approximately Gaussian and when the approximation is bad | ★ | 1 |
| A2.6 | Confidence intervals and standard errors | Explain what a CI does and does not say, and compute one for an accuracy number | ★ | 1 |
| A2.7 | Hypothesis testing, p-values, power | Choose a test, explain Type I/II error, and compute how many samples you need to detect a given difference | ★ | 1 |
| A2.8 | Bootstrap and permutation tests | Use them to get uncertainty when no clean formula exists, and know their failure cases | ★ | 0 |
| A2.9 | Paired vs unpaired comparison | Explain why comparing two models on the same test items is a paired problem and what that buys you | ★ | 0 |
| A2.10 | Multiple comparisons | Explain why running 20 comparisons finds a fake winner and name a correction (Bonferroni, Benjamini-Hochberg) | ★ | 0 |
| A2.11 | Bias-variance tradeoff | Decompose error, and say honestly where the classic picture breaks down in overparameterized models | ○ | 1 |
| A2.12 | Calibration | Explain reliability diagrams and ECE, and why accuracy and calibration are different properties | ★ | 0 |
| A2.13 | Causal basics (confounding, Simpson's paradox) | Say why correlation is not causation and give a concrete example of a flipped trend | ○ | 0 |

**Boss question:** model A scores 71.3 and model B scores 70.6 on 1,000 test items. Can you claim A is better? What do you need to know to answer, and how would you find out?

### A3. Optimization

| # | Topic | Can you... | Tier | Lvl |
| --- | --- | --- | --- | --- |
| A3.1 | Gradients, Jacobians, chain rule | Derive a gradient by hand for a small network | ○ | 2 |
| A3.2 | Convex vs non-convex | Explain why convexity gives guarantees and why deep nets work anyway in practice | ○ | 0 |
| A3.3 | Gradient descent, SGD, minibatch noise | Explain what noise does (implicit regularization, escaping sharp minima) | ○ | 1 |
| A3.4 | Momentum, RMSProp, Adam, AdamW | Explain each fix and why AdamW decouples weight decay | ★ | 1 |
| A3.5 | Learning rate, warmup, decay schedules | Say why warmup matters for transformers and what a too-high LR looks like in the loss curve | ★ | 2 |
| A3.6 | Loss landscapes, sharp vs flat minima | Explain the (contested) link between flatness and generalization | ○ | 1 |
| A3.7 | Second-order ideas (Hessian, curvature) | Say why curvature matters for pruning (Optimal Brain Surgeon idea) and quantization (GPTQ idea) | ★ | 1 |
| A3.8 | Gradient clipping, exploding and vanishing gradients | Diagnose each from training behavior | ○ | 1 |
| A3.9 | Constrained optimization, Lagrange multipliers | Explain at a high level, because it underlies SVMs and many regularizers | ○ | 1 |

**Boss question:** a loss curve spikes and then recovers, again and again. List three candidate causes and the cheapest experiment to distinguish them.

### A4. Information theory

| # | Topic | Can you... | Tier | Lvl |
| --- | --- | --- | --- | --- |
| A4.1 | Entropy | Explain it as expected surprise or minimum average code length | ★ | 2 |
| A4.2 | Cross-entropy and perplexity | Explain why training an LM with cross-entropy is maximum likelihood, and what perplexity means | ★ | 2 |
| A4.3 | KL divergence | Say why it is asymmetric, and what forward vs reverse KL each reward (mean-seeking vs mode-seeking) | ★ | 1 |
| A4.4 | Mutual information | Explain how it differs from correlation and why it is hard to estimate in high dimensions | ○ | 0 |
| A4.5 | Temperature and softmax | Explain how temperature reshapes a distribution and why it matters in distillation and decoding | ★ | 3 |
| A4.6 | Compression viewpoint (bits per parameter, rate-distortion) | Explain quantization as rate-distortion tradeoff | ★ | 1 |

**Boss question:** why does distillation with a temperature above 1 transfer more than training on hard labels, in information-theoretic terms?

---

## B. Classical machine learning

| # | Topic | Can you... | Tier | Lvl |
| --- | --- | --- | --- | --- |
| B1 | Generalization and overfitting | Explain train/val/test roles, and why evaluating on data you tuned on lies to you | ★ | 4 |
| B2 | Regularization (L1, L2, early stopping, dropout, data augmentation) | Say what each constrains and when each helps | ★ | 3 |
| B3 | Cross-validation variants (k-fold, stratified, grouped, time-series) | Choose the right one and say why random k-fold fails on grouped or temporal data | ★ | 1 |
| B4 | Metrics (precision, recall, F1, ROC-AUC, PR-AUC, log loss, Brier) | Pick a metric for an imbalanced problem and explain why accuracy misleads | ★ | 1 |
| B5 | Threshold selection and cost-sensitive decisions | Turn a probability into a decision given asymmetric costs | ○ | 3 |
| B6 | Data leakage | Name five ways it happens (target leak, preprocessing before split, duplicates, group, temporal) | ★ | 1 |
| B7 | Distribution shift (covariate, label, concept) | Define each, and say how you would detect and respond to each | ★ | 1 |
| B8 | Decision trees, random forests | Explain splitting criteria, variance reduction by bagging, and feature importance pitfalls | ○ | 2 |
| B9 | Gradient boosting (XGBoost, LightGBM) | Explain boosting as stagewise fitting of residuals and why it often wins on tabular data | ○ | 2 |
| B10 | Why trees often beat deep nets on tabular data | Give the standard explanations (irregular functions, uninformative features, rotation invariance) | ○ | 1 |
| B11 | SVMs and kernels | Explain margin maximization and the kernel trick at an intuitive level | ○ | 2 |
| B12 | Linear and logistic regression, properly | Explain assumptions, coefficients interpretation, and regularized forms | ○ | 2 |
| B13 | Curse of dimensionality | Explain concentration of distances and why nearest-neighbor and density ideas degrade | ★ | 4 |
| B14 | k-means and its assumptions | Say why it assumes spherical, similar-size clusters and is sensitive to initialization | ○ | 3 |
| B15 | DBSCAN | Explain eps, minPts, core/border/noise points, and why one global eps fails on varying density | ★ | 4 |
| B16 | HDBSCAN | Explain mutual reachability distance, the minimum spanning tree, the condensed cluster tree, and stability-based selection | ★ | 3 |
| B17 | Cluster validity (silhouette, DBCV, Davies-Bouldin, stability) | Say what each measures and why most fail for density-based clusters | ★ | 1 |
| B18 | UMAP and t-SNE caveats | Explain why distances between far-apart clusters and cluster sizes in a plot are not meaningful | ★ | 2 |
| B19 | Clustering embeddings (cosine vs Euclidean, dimensionality reduction before clustering) | Justify your pipeline choices and say what changes if the embedding model changes | ★ | 3 |
| B20 | Nearest-neighbor search (HNSW, IVF, product quantization) | Explain approximate search tradeoffs, since it links to retrieval and to quantization | ○ | 2 |

**Boss question:** your HDBSCAN output changes completely when you swap the embedding model. Is that a bug? What does it tell you, and how would you quantify clustering stability?

---

## C. Deep learning and LLMs

| # | Topic | Can you... | Tier | Lvl |
| --- | --- | --- | --- | --- |
| C1 | Backpropagation | Derive it for an MLP and explain it as reverse-mode autodiff | ○ | 2 |
| C2 | Activations (ReLU, GELU, SwiGLU) | Say why each exists and what problem it solves | ○ | 2 |
| C3 | Initialization (Xavier, He) | Explain variance preservation across layers and what goes wrong without it | ○ | 0 |
| C4 | Normalization (BatchNorm, LayerNorm, RMSNorm) | Say why transformers use LayerNorm/RMSNorm and what pre-norm vs post-norm changes | ★ | 1 |
| C5 | Residual connections | Explain them as gradient highways and as iterative refinement of a representation | ○ | 1 |
| C6 | Self-attention | Write out Q, K, V, scaling by sqrt(d), softmax, and explain what each does and the O(n^2) cost | ★ | 1 |
| C7 | Multi-head, grouped-query, multi-query attention | Explain why GQA/MQA exist (KV-cache size and memory bandwidth) | ★ |  |
| C8 | Positional encoding (sinusoidal, learned, RoPE, ALiBi) | Explain how RoPE encodes relative position by rotation and why it affects context extension | ★ | 0 |
| C9 | KV cache | Compute its memory for a given model, batch, and sequence length, and explain why decoding is memory-bound | ★ | 1 |
| C10 | Tokenization (BPE, unigram, byte-level) | Explain how tokenizer choice affects cost, multilingual fairness, arithmetic, and evaluation | ○ | 0 |
| C11 | Pretraining objectives (causal LM, masked LM) | Explain what each learns and why decoder-only dominates | ○ | 1 |
| C12 | Scaling laws and compute-optimal training | Explain Chinchilla-style results, why they favor more data than earlier practice, and why inference cost shifts the optimum | ★ | 0 |
| C13 | Supervised fine-tuning | Explain what SFT does and does not teach, and loss masking on prompts | ○ | 2 |
| C14 | RLHF, reward models, PPO | Explain the pipeline and why reward hacking and KL penalties exist | ○ | 3 |
| C15 | DPO and GRPO | Explain how each removes parts of the RLHF pipeline and what each assumes | ○ | 4 |
| C16 | Decoding (greedy, beam, top-k, top-p, temperature) | Say how each changes diversity and quality, and why greedy can loop | ○ | 1 |
| C17 | Knowledge distillation (your project) | Explain dark knowledge, soft targets with temperature, logit vs feature vs sequence-level distillation, and when it fails (capacity gap) | ★ | 2 |
| C18 | Pruning (your project) | Explain magnitude vs gradient/Hessian-based criteria, structured vs unstructured, and why unstructured sparsity rarely gives wall-clock speedup without hardware support | ★ | 2 |
| C19 | LoRA and PEFT (your project) | Explain intrinsic dimension, why low-rank works, what rank and alpha do, where LoRA underperforms full fine-tuning, and variants (QLoRA, DoRA) | ★ | 3 |
| C20 | Fine-tuning pitfalls | Explain catastrophic forgetting, overfitting on small SFT sets, and data format sensitivity | ○ | 3 |
| C21 | Embedding models and contrastive learning | Explain InfoNCE, hard negatives, and anisotropy | ★ | 1 |
| C22 | Mixed precision (FP32, BF16, FP16, FP8) | Explain dynamic range vs precision and why BF16 replaced FP16 for training | ★ | 1 |

**Boss question:** you must explain each of your past projects (distillation, pruning, LoRA, clustering) from first principles, including when it fails and what you would use instead. Do this out loud for 5 minutes each, no notes.

---

## D. Systems basics

| # | Topic | Can you... | Tier | Lvl |
| --- | --- | --- | --- | --- |
| D1 | GPU memory hierarchy (registers, shared memory, L2, HBM) | Say why moving data costs more than computing on it | ★ | 0 |
| D2 | Compute-bound vs memory-bound, arithmetic intensity | Compute arithmetic intensity of a matmul and of a decoding step, and say which regime each sits in | ★ | 1 |
| D3 | Roofline model | Draw it, place an operation on it, and predict what optimization will help | ★ | 0 |
| D4 | Prefill vs decode | Explain why prefill is compute-bound and decode is memory-bound, and what batching does to each | ★ | 0 |
| D5 | FlashAttention idea | Explain tiling and recomputation to avoid materializing the n x n matrix | ★ | 0 |
| D6 | Data, tensor, and pipeline parallelism | Explain what each splits, its communication cost, and when each is used | ○ | 1 |
| D7 | ZeRO/FSDP idea | Explain what is sharded (optimizer, gradients, parameters) and why | ○ | 1 |
| D8 | Training memory budget | Estimate memory for weights, gradients, optimizer states, and activations for a given model | ★ | 1 |
| D9 | Latency vs throughput | Explain the tradeoff and how batching moves you along it | ★ | 3 |
| D10 | Interconnects (PCIe, NVLink, network) | Say roughly how bandwidth differences shape parallelism choices | ○ | 0 |

**Boss question:** without running anything, estimate tokens per second for decoding a 7B model in FP16 on a GPU with a given memory bandwidth. Then say what quantizing to 4-bit does to that number and why.

---

## E. Niche 1: Efficient and hardware-aware ML

| # | Topic | Can you... | Tier | Lvl |
| --- | --- | --- | --- | --- |
| E1 | Quantization basics (scale, zero-point, symmetric vs asymmetric, per-tensor/channel/group) | Quantize a tensor by hand and compute the error | ★ | 1 |
| E2 | PTQ vs QAT | Explain the cost, data, and quality tradeoffs and the straight-through estimator | ★ | 0 |
| E3 | Outliers in LLM activations | Explain why activation quantization is harder than weight quantization and what emergent outlier channels are | ★ | 1 |
| E4 | GPTQ, AWQ, SmoothQuant ideas | Explain each in a few sentences: second-order error compensation, protecting salient weights, shifting difficulty from activations to weights | ★ | 0 |
| E5 | Low-bit formats (INT8, INT4, FP8, NF4, ternary) | Say what each is good for and what hardware actually supports | ★ | 1 |
| E6 | Weight-only vs weight-and-activation quantization | Predict which helps decode speed and which helps prefill | ★ | 0 |
| E7 | KV-cache quantization and compression | Explain eviction, quantization, and sharing approaches and their quality risks on long context | ★ | 0 |
| E8 | Structured sparsity (2:4, block) and what hardware exploits | Say which sparsity patterns turn into real speedups and which do not | ★ | 3 |
| E9 | Speculative decoding | Explain draft-and-verify, why output distribution is preserved, and what determines the speedup (acceptance rate) | ★ | 0 |
| E10 | Attention variants (sliding window, linear attention, state-space hybrids) | Explain the tradeoffs vs full attention and where quality is lost | ○ | 0 |
| E11 | Mixture of Experts | Explain routing, load balancing, why MoE saves compute but not memory, and serving challenges | ★ | 1 |
| E12 | Inference serving (continuous batching, PagedAttention, prefix caching) | Explain what each fixes and how utilization and latency interact | ★ | 0 |
| E13 | Compression pipelines (combining distill + prune + quantize) | Explain why compressions interact, and why order matters | ★ | 1 |
| E14 | On-device and edge constraints | Reason about memory, power, thermal, and operator support limits | ○ | 0 |
| E15 | Cost and energy modeling | Convert tokens, FLOPs, and utilization into cost per million tokens and energy per request | ★ | 0 |
| E16 | Compression evaluation | Say what a compressed model loses beyond headline accuracy (long context, rare skills, calibration, safety behavior) | ★ | 1 |

**Boss question:** given a new accelerator's FLOPs, bandwidth, and memory, and a quality target, which compression technique would you pick for a 13B model, and what would you predict before measuring?

---

## F. Niche 2: Evaluation and measurement science

| # | Topic | Can you... | Tier | Lvl |
| --- | --- | --- | --- | --- |
| F1 | Construct validity | Explain whether a benchmark measures what it claims, with a concrete example of a mismatch | ★ | 1 |
| F2 | Benchmark saturation and contamination | Explain detection approaches (n-gram overlap, canary strings, perturbation) and their limits | ★ | 0 |
| F3 | Variance sources (seeds, prompts, sampling, ordering of few-shot examples) | Quantify each source and explain why single-run comparisons are unreliable | ★ | 2 |
| F4 | Statistical comparison of models | Do paired bootstrap or McNemar-style tests and report effect sizes with intervals | ★ | 1 |
| F5 | Sample size and power for evals | Compute how many items you need to detect a 1-point difference | ★ | 1 |
| F6 | Prompt and format sensitivity | Demonstrate that rephrasing changes rankings, and say how to design robust eval prompts | ★ | 2 |
| F7 | LLM-as-judge | Explain position bias, verbosity bias, self-preference, and how to validate a judge against humans | ★ | 2 |
| F8 | Human evaluation design | Design a rubric, pilot it, and handle rater training and drift | ★ | 1 |
| F9 | Inter-annotator agreement | Compute and interpret Cohen's and Fleiss' kappa and Krippendorff's alpha, and say why raw agreement misleads | ★ | 0 |
| F10 | Pairwise comparison and ranking (Elo, Bradley-Terry) | Explain how rankings are derived from preferences and their assumptions | ○ | 0 |
| F11 | Calibration and selective prediction | Evaluate confidence quality and risk-coverage tradeoffs | ★ | 0 |
| F12 | Failure-mode taxonomies and error analysis | Build a taxonomy from sampled failures, and measure how common each is | ★ | 0 |
| F13 | Slice-based and robustness evaluation | Evaluate by subgroup and perturbation, and say why an average can hide a failure | ★ | 0 |
| F14 | Metric gaming (Goodhart's law) | Give examples of optimizing a metric until it stops measuring the thing | ○ | 1 |
| F15 | Reproducibility | List what must be logged to reproduce an eval result | ★ | 0 |
| F16 | Task-specific metrics (BLEU/ROUGE limits, exact match, pass@k, win rate) | Say what each captures, and how pass@k is estimated without bias | ○ | 0 |

**Boss question:** pick a popular benchmark claim and show, with numbers, that it is more fragile than it looks (variance, contamination, prompt sensitivity, or construct mismatch).

---

## G. Niche 3: Data-centric ML and representation geometry

| # | Topic | Can you... | Tier | Lvl |
| --- | --- | --- | --- | --- |
| G1 | Embedding space geometry (anisotropy, hubness, intrinsic dimension) | Explain what each is and how it distorts similarity and clustering | ★ | 1 |
| G2 | Exact and near-duplicate detection (MinHash, LSH, embedding similarity) | Choose a method and tune thresholds, and say why dedup improves generalization and reduces memorization | ★ | 1 |
| G3 | Coverage and diversity measures | Define and compute coverage of an embedding set, and say why diversity is hard to measure | ★ | 2 |
| G4 | Data selection (quality filtering, influence, uncertainty, coreset) | Compare approaches and say what each assumes | ★ | 0 |
| G5 | Data mixture optimization | Explain how domain proportions affect capability and how mixtures are searched for | ○ | 1 |
| G6 | Curriculum and ordering effects | Explain what is known and what is contested | ○ | 0 |
| G7 | Label noise (detection and learning with it) | Explain confident learning and noise-robust losses | ★ | 2 |
| G8 | Synthetic data | Explain generation, filtering, and verification, and the risk of model collapse when models train on their own outputs | ★ | 1 |
| G9 | Data scaling laws | Explain how loss changes with data quantity and quality, and why returns diminish | ○ | 2 |
| G10 | Data contamination and privacy | Explain memorization, extraction risk, and deduplication's role | ○ | 3 |
| G11 | Active learning | Explain uncertainty and diversity sampling and when it beats random labeling | ○ | 1 |
| G12 | Annotation economics | Reason about cost, quality, and agreement tradeoffs when buying labels | ○ | 2 |

**Boss question:** show a curated subset that beats a larger random sample, and explain why in terms of coverage, redundancy, or noise.

---

## H. Niche 4: Interpretability, robustness, and safety (research hedge)

| # | Topic | Can you... | Tier | Lvl |
| --- | --- | --- | --- | --- |
| H1 | Probing classifiers | Explain what a probe shows and the control-task caveat (a probe may learn the task, not read the model) | ◇ | 0 |
| H2 | Superposition and polysemantic neurons | Explain why neurons mix features and why that motivates dictionary learning | ◇ | 3 |
| H3 | Sparse autoencoders | Explain the objective, the sparsity tradeoff, dead features, and how to judge feature quality | ◇ | 1 |
| H4 | Circuits and mechanistic analysis (activation patching, causal tracing, logit lens) | Explain what each intervention establishes causally | ◇ | 0 |
| H5 | Attribution methods (saliency, integrated gradients, SHAP) | Explain what each claims and why they can disagree or mislead | ◇ | 1 |
| H6 | Adversarial robustness | Explain adversarial examples, adversarial training, and the robustness-accuracy tradeoff | ◇ | 0 |
| H7 | Out-of-distribution detection | Explain softmax-confidence baselines, density and distance-based methods, and their limits | ◇ | 2 |
| H8 | Uncertainty (ensembles, MC dropout, conformal prediction) | Explain what coverage guarantee conformal prediction gives and its assumptions | ◇ | 1 |
| H9 | Jailbreaks, prompt injection, and red teaming | Explain attack classes and how to measure defense honestly | ◇ | 2 |
| H10 | Alignment concepts (reward hacking, sycophancy, goal misgeneralization) | Give a concrete example of each | ◇ | 1 |
| H11 | Research methodology | Explain what makes an interpretability claim falsifiable and what baselines it needs | ◇ | 0 |

**Boss question:** train a tiny sparse autoencoder on a small model's activations. Which features look interpretable, which are dead, and how would you tell a real feature from a story you told yourself?

---

## I. Cross-cutting skills (not topics, but they decide how far you go)

| # | Skill | Can you... | Tier | Lvl |
| --- | --- | --- | --- | --- |
| I1 | Reading papers critically | State claim, evidence, assumptions, and where it could break, in five sentences | ★ | 1 |
| I2 | Reproducing a result | Reproduce a published number and explain the gap if you cannot | ★ | 0 |
| I3 | Ablation and experiment design | Isolate one variable, control seeds, and say what each ablation rules out | ★ | 1 |
| I4 | Back-of-envelope estimation | Estimate params, FLOPs, memory, and cost before running anything | ★ | 0 |
| I5 | Writing up findings | Produce a one-page writeup with a clear claim and honest limitations | ★ | 0 |
| I6 | Debugging training and inference | Triage a bad loss curve, NaNs, or a quality drop without guessing | ★ | 1 |
| I7 | Explaining to different audiences | Explain a technique to a manager, a peer, and a researcher at different depths | ★ | 2 |

---

## J. The system that keeps this map alive

### Debts list

When a question exposes a missing prerequisite, log it here and move on. Pay down one per cycle, oldest first.

| Date | Debt (what I skipped) | Which cell it unlocks | Paid? |
| --- | --- | --- | --- |
|  |  |  |  |

### Question backlog (seed it with these, then add your own)

Each is answerable by a small experiment plus a short writeup. Tags show which cells it moves.

1. Does 4-bit quantization degrade some kinds of inputs more than others? *(E1, E2, F4, F13)*
2. How many eval items do I need to claim A beats B at a given effect size, and how does that change with paired testing? *(A2.6 to A2.9, F4, F5)*
3. Why does activation quantization hurt more than weight quantization, and can I see the outlier channels myself? *(E3, E6)*
4. Estimate decode throughput for a 7B model from memory bandwidth alone, then measure and explain the gap. *(D2, D3, D4, C9)*
5. How much of the KV cache can I drop or quantize before long-context quality collapses? *(C9, E7, F13)*
6. Does unstructured 50% pruning speed anything up on my hardware? Does 2:4? *(C18, E8)*
7. How sensitive is a benchmark ranking to prompt wording and few-shot order? *(F3, F6)*
8. Does my LLM judge prefer longer answers, and how do I correct for it? *(F7)*
9. What fraction of a popular dataset is near-duplicate, and does removing it change model quality? *(G2, G4)*
10. Why does my HDBSCAN output change when the embedding model changes, and how stable are the clusters? *(B16, B17, B19, G1)*
11. How does distillation temperature change what the student learns, and when does a big capacity gap break it? *(A4.3, A4.5, C17)*
12. At what LoRA rank does a task stop improving, and what does that say about its intrinsic dimension? *(A1.5, C19)*
13. Does model-on-model synthetic data cause collapse in a small experiment, and what shows up first? *(G8)*
14. Train a small SAE on one layer. What do the features look like? *(H2, H3)*
15. After compressing a model, what does it lose that the headline benchmark did not show? *(E13, E16, F12)*

### Weekly review (15 minutes, same place every week)

1. What did I learn? Which cells moved?
2. What confused me? Add to debts or new rows.
3. Pick the next question using the rule: weak cell + ★ + connects to a niche.

### Gap audit (every 6-8 weeks)

Run a mock interview with an LLM across this map and tell it to probe deeper whenever you hesitate. Re-grade every row you stumbled on, and add anything from real work or real interviews that caught you out.

---

## K. EDA practice set: five datasets with planted traps

For each dataset, first run your full EDA protocol, writing findings in a data card. Only then reveal what you planted (or check the known trap). **Pass condition:** your protocol caught the trap without being told it existed. If it did not, add a line to your personal checklist.

| # | Dataset | What it trains | Known or planted traps | Pass condition |
| --- | --- | --- | --- | --- |
| 1 | Ames Housing (or Kaggle House Prices) | Structural audit, missingness, univariate look | In this data, "NA" often means "feature absent" (no garage, no pool), not missing. Plant: sentinel values (-1, 9999) in a few columns, a feature derived from the target (price per square foot), and MNAR missingness where expensive homes skip a field | You distinguish true missing from structural NA, find the sentinels, and flag the leaky feature before modelling |
| 2 | Credit-card fraud (Kaggle, European cardholders) | Imbalance, metric choice, leakage, time | Very rare positive class, so accuracy is meaningless. Plant: exact duplicates split across train and test, label noise on a random 2% of positives, and a random split that ignores time order | You choose PR-AUC or cost-based metrics, find the duplicates, and show a random split overstates performance versus a time split |
| 3 | 20 Newsgroups (text) | Text variant, leakage, near-duplicates | Headers, footers, and quoted text leak the label (a well-known issue). Plant: near-duplicate posts across splits, boilerplate signatures, and a few mislabeled documents | You find the leakage, remove it, and show how much the score drops |
| 4 | Embeddings of a text set (e.g. a news corpus embedded with two different models) | Embedding variant, clustering, coverage | Plant: a junk cluster of near-identical documents, a handful of outliers, and a small but real minority topic. Compare HDBSCAN results across the two embedding models and across cosine vs Euclidean | You find the junk cluster and the minority topic, quantify how cluster assignments change across models, and explain hubness or density effects you observe |
| 5 | A time series with gaps (e.g. hourly bike-sharing or electricity demand) | Time series variant, shift, splits | Plant: a level shift partway through, a block of missing hours, and duplicated timestamps. Known: strong daily and weekly seasonality | You detect the level shift and gaps, show non-stationarity, and demonstrate leakage from a random split versus a forward-chaining split |

**Extra loop for each dataset:** before fitting a baseline, write down where you expect it to fail. Fit it, check, and write what surprised you. Mismatch between prediction and result is where the learning is.

---

## L. How to stay hooked

- **Look at your own graph.** Count cells at level 3 or above each month. Watching ★ rows turn into 3s and 4s is a stronger motivator than any plan.
- **Keep cycles short and finish them.** A 3-5 week cycle with one artifact beats a perfect plan.
- **Prefer combined questions.** The best cycles hit two niches at once, such as quantize and then evaluate rigorously.
- **Collect your surprises.** Every time the result disagrees with your prediction, record it. That list is the best evidence of real expertise you will ever have.
- **Share one artifact per cycle,** even a short post or notebook. Public work is what turns private knowledge into a reputation.
