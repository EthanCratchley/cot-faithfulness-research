# Do CoT Faithfulness Metrics Agree?

---

## 1. In one paragraph

Researchers have built several different ways to measure whether a model's written reasoning is "faithful" — whether it reflects what actually drove the answer. Each metric was invented in a different paper, applied to different models on different datasets. Recent work shows these metrics **disagree with each other on individual reasoning traces**, sometimes by 20–60 percentage points. Nobody has checked whether they also disagree about _which models are more faithful_. This study runs several metrics over one fixed set of reasoning traces and compares the rankings they produce.

## 2. Research question

> When a paper reports that Model A's reasoning is more faithful than Model B's, how much of that conclusion is determined by which faithfulness metric the authors happened to use?

If different metrics rank models differently on identical data, then "CoT faithfulness" is not a single property being measured — and any claim about which model is more monitorable is partly a claim about metric choice.

**Why it matters:** if you're selecting a model for deployment because its reasoning is easier to monitor, the metric you trusted determined your answer.

**Anticipated objection, answered up front.** A reviewer will say: these metrics measure different constructs (hint-honesty vs. causal load-bearing), so of course they disagree, and disagreement is uninteresting. The answer is that the literature does not treat them as different constructs — papers cite each other's faithfulness results as commensurable, "faithfulness" appears unqualified in abstracts and in monitorability arguments, and model-selection claims are made from single-metric evidence. The contribution is measuring the size of the error that conflation produces. **The writeup must make this argument explicitly in the first page, with citations of the conflation.**

## 3. Why this is open

Prior work established the pieces but not the comparison:

- Multiple faithfulness metrics exist: Biasing Features (hint verbalization), Filler Tokens, Early Answering, FUR, faithful@k, causal mediation
- [Is CoT Really Not Explainability?](https://arxiv.org/html/2512.23032v2) showed these metrics disagree at the trace level — among traces flagged unfaithful by Biasing Features, 20–60% are judged faithful by other metrics
- That paper also showed hint verbalization is heavily sampling-budget dependent: ~25–40% at k=1, rising toward 0.9 at k=16

**The gap:** every paper applies its own metric to its own models on its own data. No one has held the data fixed and varied only the metric. That comparison is what makes metric disagreement measurable.

## 4. Pre-registered hypothesis

**Predicted:** metric-induced rankings will diverge substantially — at least one metric pair with **Kendall's tau < 0.6 whose bootstrap 95% CI excludes 0.8** — and trace-level agreement will be poor (pooled **kappa < 0.4**).

**Secondary prediction:** disagreement will be larger for thinking-mode models than instruct models, because there is more reasoning content for the metrics to disagree about. Tested primarily on the matched Qwen pair.

**What falsifies this:** all metric pairs show tau > 0.8 with CIs excluding 0.6. That would mean the metrics are interchangeable in practice despite disagreeing on individual traces — a useful and publishable null.

**Point estimates are not enough.** With n=8 models, the null SD of Kendall's tau is ≈0.29 — a single rank swap moves a point estimate across the 0.6 threshold. Every tau is reported with a bootstrap CI resampled over **items** (§7). A hypothesis stated on point estimates alone is untestable at this n.

_Write final predictions down before the first full run. Do not revise after seeing results._

## 5. Design

### Execution: local weights, not API

**All generation runs on rented GPUs under vLLM.** This reverses the v1/v2 all-API decision. The
reason is recorded in §6 and the evidence is in §10 Step 1: API serving layers silently refuse to let
you write tokens into a model's context, which is exactly what two of the three metrics require. That
refusal is per-provider, undocumented, and it selected the model list for us.

Under vLLM we use the **raw completions endpoint**, which takes a prompt string rather than a message
list. We construct the entire prompt ourselves, including each model's thinking delimiters. There is no
layer between us and the tokenizer. Prefill is not a feature we have to request — it is just the prompt.

### Models

Eight models, seven labs, all ≤32B so each fits one 80GB GPU in bf16.
**Step 0 verified: every repo below exists on HuggingFace, none are gated, all weights are downloadable.**

| HuggingFace repo                             | Lab       | Params | VRAM bf16 | License    | Role                               |
| -------------------------------------------- | --------- | ------ | --------- | ---------- | ---------------------------------- |
| `Qwen/Qwen3.8-27B`                           | Alibaba   | 27.8B  | 56 GB     | apache-2.0 | Current strong open model          |
| `google/gemma-4-31B-it`                      | Google    | 31.3B  | 63 GB     | apache-2.0 | Different family                   |
| `openai/gpt-oss-20b`                         | OpenAI    | 20.9B  | 42 GB     | apache-2.0 | Reasoning-native                   |
| `allenai/Olmo-3.1-32B-Think`                 | Ai2       | 32.2B  | 64 GB     | apache-2.0 | **Matched pair** — think arm       |
| `allenai/Olmo-3.1-32B-Instruct`              | Ai2       | 32.2B  | 64 GB     | apache-2.0 | **Matched pair** — instruct arm    |
| `meta-models/Muse-Glimmer-30B`               | Meta      | 29.8B  | 60 GB     | apache-2.0 | Meta representation                |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | NVIDIA    | 31.6B  | 63 GB     | **other**  | MoE                                |
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | Mistral | 24.0B  | 48 GB     | apache-2.0 | Different family                   |

**Pin every model by HuggingFace revision hash** at download time and record it on every result row.
This is what replaces the API path's "date-stamp everything and hope" drift control.

**Two corrections found in Step 0:**

- `mistral-small-2603` is **119B**, not 24B — too large for one GPU. Replaced with
  `Mistral-Small-3.2-24B-Instruct-2506`.
- `NVIDIA-Nemotron-3-Nano-30B-A3B` ships under NVIDIA's own licence, not Apache. It is the only
  non-permissive entry; read the terms before relying on it, and note the licence in the release.

**The Olmo pair is restored.** `allenai/Olmo-3.1-32B-Instruct` was never missing — it simply has no
OpenRouter serving endpoint, which is what made it look unavailable through three revisions of this
spec. Same 32B, same Apache 2.0, same training run, differing only in post-training. This is the pair
that motivated the original design, and it is the strongest argument for the local execution decision.

**Every model rejected by the API path is restored.** `gpt-oss-20b`, `gemma-4`, `muse-glimmer` and the
Olmo pair were cut for provider policy, not capability. The only genuine losses are models too large for
one GPU: `deepseek-v3.2` (685B) and `glm-4.7`. Those go in §9.

**Selection basis returns to diversity of lab, architecture and training approach.** Capability-based
selection is unnecessary because every metric works on every model we can load.

### Matched pairs — two of them

Both are cheap locally and they answer different questions. Run both.

**1. Olmo-3.1-32B Think vs. Instruct.** Different weights, same base model, same training data, same
size, same licence — differing only in post-training. Asks: *does post-training a model to reason change
how faithful its reasoning is?* This is the pair the study was originally designed around.

**2. One model, thinking enabled vs. suppressed.** Same weights, same GPU, same precision, same sampling
params, one bit changed — controlled by our template rather than a provider's policy. Asks: *for
identical weights, does emitting a CoT change the answer process?* Model chosen after Step 2, on whichever
has the longest and most stable CoT.

Pair 1 varies training; pair 2 varies inference. Reporting both separates those, which no single pair can.

### Dataset

**MMLU-Pro** — 500 items, stratified across subjects.

1. Multiple choice — required for clean answer-switch detection
2. Baseline accuracy in the **50–80% band**
3. 10 answer options makes a hint pointing at a specific wrong answer more informative than 4 would

The band is more likely to hold than under the v3 API list: these are 20–32B current-generation models
rather than the 8–24B capability-selected set. Still verified in Step 2, and the fallback if it fails is
4-option MMLU, which moves the same models up roughly 10–15 points.

**OpenbookQA** — validation only, to reproduce the published Biasing Features number.

### Trace population and common support

All metrics are computed over **one shared trace set**: the 500 hinted traces per model. Metrics differ
in where they are *defined* — Biasing Features and faithful@k only on flipped traces (where the hint
changed the answer), Filler Tokens and Early Answering on all traces. Trace-level kappa is computed on
the intersection; **pooled across models as primary, per-model as supporting.** Rankings use each
metric's full native support.

### Metrics

All oriented so **higher = more faithful**.

| Metric                | What it tests                                                          | Score                                       |
| --------------------- | ---------------------------------------------------------------------- | ------------------------------------------- |
| **Biasing Features**  | Does the CoT verbalize the injected hint?                              | Verbalization rate over flipped traces      |
| **Filler Tokens**     | Replace the CoT with "…" — does the answer change?                     | Answer-change rate                          |
| **Early Answering**   | Truncate the CoT at 0/20/40/60/80% — is the answer already determined?  | AOC over truncation fractions               |
| **faithful@k**        | Across k samples, does *any* verbalize the hint?                        | Any-of-k rate — **sampling-budget axis**    |
| **CMA** *(stretch)*   | Splits the hint effect into direct vs. CoT-mediated                     | Mediation fraction                          |
| ~~FUR~~               | Excluded — needs per-step parameter unlearning                          | —                                           |

**faithful@k is not a fourth independent metric.** If per-sample verbalization rate is `p`, then
faithful@k ≈ `1 − (1−p)^k` — a monotone transform of `p`, so its rank correlation with Biasing Features
is forced toward +1 by construction. It is reported only as the sampling-budget sensitivity axis.

**The headline comparison is three independent constructs:** hint-honesty (Biasing Features), causal
load-bearing (Filler Tokens), answer-determination timing (Early Answering). Three pairwise taus, none
rigged.

**CMA returns as a stretch goal.** It was cut under the API design because several endpoints withheld
`logprobs`. vLLM returns logprobs for every model, so the blocker is gone; only implementation effort
remains.

**k rises from 8 to 32.** Under API pricing, sampling cost scaled with tokens and k=8 was a budget
ceiling. Locally the cost is GPU-time, and generation is throughput-bound rather than price-bound, so a
larger k costs minutes rather than dollars. k=32 is much closer to the source paper's 128, and the
"lower bound" caveat weakens accordingly.

**Trace-level binarization for kappa**, fixed in advance:

- Biasing Features — verbalized / not
- Filler Tokens — answer changed / not
- Early Answering — faithful iff the answer at the **0.6 truncation fraction** differs from the full-CoT answer
- faithful@k — any-of-k verbalized

### Intervention

Single hint type: **recommendation framing** ("a professor suggests the answer is X"), matching the
source paper for comparability.

## 6. Decisions log

| Decision                  | Choice                                                     | Rationale                                                                                                                                                                                                          |
| ------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Execution**             | **Local weights on rented GPUs, vLLM raw completions**      | Reverses v1/v2. Step 1 showed the API path vetoes two of three metrics and selects the model list by provider policy. Local restores full context control. Cost is comparable; the loss is a half-day of setup.    |
| **Host**                  | RunPod, 1× A100 80GB, network volume                        | Per-second billing suits stop-start development; network volumes persist the weight cache across pods so 320GB is downloaded once. Lambda re-pulls weights per instance and has thinner A100 availability.          |
| **Precision**             | bf16 for every model                                        | Uniform across the list — a cross-model difference can no longer be a quantization artifact. Under the API path precision was whatever the prefill-capable provider happened to offer.                              |
| **Model selection basis** | Diversity of lab, architecture, training approach           | Capability-based selection is no longer needed; every metric works on every loadable model.                                                                                                                        |
| **Matched pair**          | One model, thinking on vs. suppressed                       | Same weights, GPU, precision and sampling params. Strictly better control than any two-model pair, and free locally.                                                                                                |
| **Size ceiling**          | ≤32B, one GPU                                               | Keeps the run on a single A100 and the budget near $50. Costs us `deepseek-v3.2` and `glm-4.7`; recorded in §9.                                                                                                     |
| **k**                     | **32**, on a 150-item subset                                | Sampling is throughput-bound rather than price-bound locally. Closer to the source's 128.                                                                                                                          |
| **Sample size**           | 500, stratified, fixed across models                        | Fixed item set is required by the design.                                                                                                                                                                          |
| **Judge model**           | API, `claude-haiku-4.5`, validated vs Opus 5 on 200 items, gate κ ≥ 0.8 | Judging is read-only, so it stays on the API where it is cheap. Judge is outside the test set. Gate pre-set.                                                                                          |
| **Seeds**                 | Set and logged; vLLM is deterministic at fixed batch config | Real reproducibility, unlike the API path where batching made seeds advisory.                                                                                                                                      |
| **CMA**                   | Restored as a stretch goal                                  | Cut earlier for missing logprobs; vLLM exposes them for every model.                                                                                                                                               |
| **FUR**                   | Excluded                                                    | Requires Negative Preference Optimization per reasoning step. Infeasible.                                                                                                                                           |

## 7. Analysis plan

**Primary — Kendall's tau between metric-induced model rankings, with bootstrap CIs.** Each metric produces an ordering of the eight models. Tau measures similarity between two orderings: +1 identical, 0 unrelated, −1 reversed. Computed for the three independent metric pairs (BF↔Filler, BF↔Early, Filler↔Early).

**Bootstrap over items, not models.** Resample the 500 items with replacement ≥2,000 times; recompute every model's score under every metric on each replicate; recompute tau. Report the 95% percentile CI. With n=8 models the null SD of tau is ≈0.29, so a bare point estimate cannot distinguish "metrics disagree" from sampling noise.

_Answers: does it matter which metric a researcher picked?_

**Primary companion — pairwise rank-swap frequency.** For each model pair (A, B) and each metric pair (M1, M2), report the fraction of bootstrap replicates in which A outranks B. A statement of the form _"under Biasing Features, model X beats Y in 91% of resamples; under Filler Tokens, in 12%"_ is more robust to n=8 than a single tau and is the form a reviewer can actually check.

**Secondary — Cohen's kappa on trace-level agreement between metric pairs.** For each trace, each metric returns faithful/unfaithful (binarization fixed in §5). Kappa measures agreement corrected for chance. Computed on the common support defined in §5. **Pooled across models as primary; per-model as supporting.**

_Answers: do the metrics disagree about individual cases, or agree on cases while rankings flip because models are bunched together?_

Both are needed — they can come apart. High kappa with low tau means models are too close to separate. Low kappa with high tau means constant disagreement that averages out.

**Sampling-budget axis (faithful@k).** Verbalization rate as a function of k ∈ {1, 2, 4, 8} per model, and the induced ranking at each k. Report whether the model ranking itself changes with k. This is metric-choice sensitivity of a second kind and is reported separately from the three-metric headline — never folded into it.

**Supporting outputs:**

- Per-model, per-metric scores with bootstrap CIs
- Per-model baseline (unhinted) accuracy, reported next to every faithfulness score
- Hint flip rate per model (how many traces are even eligible for Biasing Features)
- Thinking vs. instruct split on the Qwen matched pair
- Cheap-judge vs. Opus agreement rate (κ, plus confusion matrix)
- Answer-extraction accuracy against hand-checked sample
- Mean CoT length per model (a confound for both Filler Tokens and Early Answering)

## 8. Confounds and handling

| Confound                                 | Handling                                                                                  |
| ---------------------------------------- | ----------------------------------------------------------------------------------------- |
| Serving stack / quantization differences | **Eliminated.** One vLLM version, one GPU, bf16 for every model. Log vLLM and CUDA versions |
| Model version drift mid-run              | **Eliminated.** Weights are pinned by HuggingFace revision hash and frozen on the volume   |
| Sampling nondeterminism                  | vLLM is deterministic at a fixed batch configuration. Set and log seeds; log batch settings |
| CoT extraction differing by model        | Per-model template config specifying thinking delimiters, hand-verified in Step 1-local. This is now inspectable rather than inferred from black-box behaviour |
| Prefill silently failing                 | Verify by tokenizing the constructed prompt and asserting our injected span is present, before any run |
| Thinking suppression changing the system | The matched pair measures the size of this effect directly rather than assuming it away    |
| Answer parsing errors                    | Build extractor, hand-check 100 outputs, report accuracy                                  |
| Judge model bias                         | Judge is outside the test set; validate against Opus 5, gate κ ≥ 0.8                      |
| CoT length driving Filler / Early scores | Report mean CoT length per model; check correlation with both metrics before interpreting |
| Near-ceiling models have few flips       | Report flip rate and baseline accuracy per model; flag any model with <30 eligible traces |

## 9. Limitations (drafted before results, deliberately)

- **Size ceiling.** Every model is ≤32B so it fits one GPU. `deepseek-v3.2` (685B) and `glm-4.7` are
  excluded on cost, so the study says nothing about whether metric disagreement behaves differently at
  frontier scale. This is the main generality limit and it is a budget decision, not a design one.
- One hint type. Other interventions may behave differently.
- One dataset. Generality across task types unestablished.
- FUR excluded; CMA only if the stretch goal lands. Three independent constructs, not five.
- Filler Tokens and Early Answering are measured **with thinking suppressed**, while Biasing Features
  and faithful@k are measured with it on. The constructs are therefore not measured on identical model
  configurations. The matched pair quantifies the cost of this.
- Metric implementations are ours, validated against one published number on one model.
- n=8 models. Kendall's tau is noisy at this n; conclusions rest on bootstrap CIs and rank-swap
  frequencies, not point estimates.
- Trace-level kappa is computed on a small common support. Pooled kappa is the defensible number.
- **Resolved by going local, and no longer limitations:** capability-based sampling bias, quantization
  heterogeneity, version drift, absent seed control, and the claim that these metrics cannot be run on
  closed models — all of which constrained the API design. See the Step 1 finding below.

### Finding that came out of Step 1

Of 106 API-served models screened, **55 (52%) support the two capabilities the prefill-based metrics
require**, and after excluding roleplay finetunes, translation models, vision models and models too
small for the accuracy band, roughly 20 remain. Every rejection has a distinct cause: `glm-4.7` honors
no prefill on any of 8 providers; `muse-glimmer` and `gpt-oss-20b` return *"Reasoning is mandatory for
this endpoint"* on every provider; `qwen3-next-80b-thinking` is locked on both axes; `olmo-3-32b-think`
is listed in the catalog with zero serving endpoints.

Because the same models are then run locally, the claim sharpens from "some APIs don't support this" to
**"the weights support these metrics; the serving layer is what breaks them."** Running the API-capable
subset both ways isolates the cause. The monitorability literature assumes these metrics port across
models; this shows they port across *weights* but not across *deployments*. **This belongs in the paper
as a result, not in the limitations.**

## 10. Steps from here

Each step is a gate. Do not proceed past a failing gate.

**Step 1 — API capability probe.** ✅ **PASSED — and its result is why the design moved local.**
Five successive probes: single-model prefill, injection-path variants, reasoning-disable rescue, an
8-model × all-providers sweep, and a 106-model catalog screen. Caught three failures that would
otherwise have been silent: reasoning-channel prefill is dropped with HTTP 200 and no warning; thinking
models re-reason when handed a filler trace (glm-4.7 produced 1,736 chars against a 1,777 baseline);
prefill support is per-provider, not per-model. Scripts and JSON outputs ship with the paper as evidence
for the §9 finding.

**Step 0 — Weight availability check.** ✅ **PASSED** *(~30 min, free)*
All eight repos exist on HuggingFace, none gated, all downloadable. Two corrections applied to §5:
`mistral-small-2603` is 119B (replaced with the 24B Mistral-Small-3.2), and Nemotron ships under
NVIDIA's own licence rather than Apache. Found `allenai/Olmo-3.1-32B-Instruct`, restoring the original
matched pair. Revision hashes to be recorded at download time in Step 1L.

**Step 1L — Local prefill smoke test.** *(~1 hour, ~$2)*
Provision the pod, load one model, and verify the whole premise: construct a prompt with an injected
CoT, tokenize it, assert our span is present, generate, and confirm the model continues rather than
re-reasoning. Repeat per model as each is added.
→ **Gate:** injection must be visible in the tokenized prompt. This is the local equivalent of the gate
the API failed, and unlike the API version it is directly inspectable.

**Step 2 — Baseline accuracy pilot.** *(~2 hours)*
Each model, ~200 MMLU-Pro items, no hints, no metrics. Also measure CoT length and tokens/sec per model.
→ **Gate:** the median model lands in the 50–80% band and no more than two fall outside. If not, switch
to 4-option MMLU.
→ Also selects the matched-pair model (longest, most stable CoT).

**Step 3 — Write the pre-registration.** *(~1 hour)*
Predicted answer, primary statistic (tau with bootstrap CI), falsification condition, confound list.
One page, before any real runs. Timestamp publicly (OSF is free).

**Step 4 — Build the harness on three models.**
Answer extractor + hand-check. Hint injection. Biasing Features, Filler Tokens, Early Answering.
Judge pipeline with cheap-vs-Opus validation.

**Step 5 — Validation gate.** *(hard stop)*
Reproduce the published Biasing Features number on the anchor model/dataset.
→ **Gate: within ±10 percentage points absolute.**
→ **Precondition:** obtain the source paper's judge prompt and hint template. Reproducing their number
with a different judge prompt tests a different pipeline. If unavailable, record the anchor as weakened.

**Step 6 — Full runs.** All models, all metrics, k=32. Log weights revision, seed, vLLM version, batch
config, and token counts on every row.

**Step 7 — Analysis.** Tau with bootstrap CIs, rank-swap frequencies, pooled kappa, sampling-budget
curve, supporting outputs in §7.

**Step 8 — Canary re-run.** Repeat a subset at the end. Under local execution this checks *our* pipeline
for drift, not the vendor's — it should be exactly reproducible, and any deviation is a bug.

**Step 9 — Write.** ~2,500 words. Finding in the first three sentences. The construct-conflation
argument (§2) and the Step 1 deployment finding both on the first page. Publish harness, traces, and
analysis notebook.

## 11. Resolved / open

| Was open                    | Resolved to                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| Prefill feasibility         | **Impossible via API, trivial locally.** Drove the execution change.                              |
| Model availability          | Restored by going local. Only >32B models are lost.                                               |
| Judge model                 | `claude-haiku-4.5` via API, validated vs Opus 5, gate κ ≥ 0.8                                      |
| Metric independence         | Early Answering added; faithful@k demoted to the sampling-budget axis                             |
| CMA feasibility             | Blocker (logprobs) removed by vLLM; restored as a stretch goal                                    |
| k                           | Raised 8 → 32; sampling is throughput-bound locally                                               |
| Host                        | RunPod, 1× A100 80GB, network volume                                                              |
| Weight availability         | **All eight confirmed on HuggingFace, ungated.** Olmo pair restored; Mistral swapped to the 24B   |

**Still open:**

- Whether MMLU-Pro holds the accuracy band on this list (Step 2); fallback is 4-option MMLU
- Whether the source paper's judge prompt can be obtained (blocks a clean Step 5)
- Whether CMA is worth the implementation effort once the three core metrics are running
- Whether the four 60–64 GB models need an H100 or 2×A100 for workable throughput (Step 2)
- Whether NVIDIA's licence for Nemotron permits the intended release

## 12. Budget

Cost is **GPU-hours**, not tokens. Per model: ~500 items × (1 unhinted + 1 hinted) + 150 × 32 × 2 for
faithful@k + 500 filler + 2,500 early-answering calls.

A 30B model under vLLM on an A100 80GB sustains roughly 2,000 output tok/s with high batch concurrency.

**VRAM headroom caveat.** Four models are 60–64 GB in bf16, leaving only ~14 GB for KV cache on an 80 GB
A100. That runs, but concurrency — and therefore throughput — drops on those models, so the GPU-hour
figures below skew optimistic. Options are to accept it, move the four large models to an H100, or use
tensor parallelism across 2×A100. Decision deferred to Step 2, where real throughput is measured rather
than guessed.

| Item                                              | Estimate      |
| ------------------------------------------------- | ------------- |
| Generation, 8 models @ ~1.5 GPU-hr each           | 12 GPU-hr     |
| Step 1L/2 pilots, matched-pair second condition   | 4 GPU-hr      |
| Setup, debugging, re-runs, weight downloads       | 10–15 GPU-hr  |
| **Total GPU time**                                | **~30 GPU-hr**|
| A100 80GB @ ~$1.20–1.90/hr                        | **$40–55**    |
| Network volume, 750GB, one month                  | $5–10         |
| Judge calls (Haiku 4.5 bulk + Opus 5 subsample)   | $10           |
| Step 1 API probes (already spent)                 | $2            |
| **Expected total**                                | **~$60–75**   |
| **Hard ceiling**                                  | **$120**      |

Comparable to the API budget, and it buys uniform precision, real seeds, no drift, k=32 instead of 8,
CMA back on the table, and a model list chosen for diversity rather than for which vendor implemented
prefill.

**Note on marginal cost.** Under the API path, more samples meant more dollars. Locally, more samples
mean more minutes on a box already running. If the analysis wants a larger k, more items, or an extra
hint type, the cost is time rather than budget — worth remembering before scoping anything down.
