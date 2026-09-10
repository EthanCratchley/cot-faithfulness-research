# Do CoT Faithfulness Metrics Agree?

Several published metrics claim to measure whether a model's chain-of-thought reflects what
actually drove its answer. Each was introduced in a different paper, applied to different models
on different datasets. Recent work shows they **disagree on individual reasoning traces**. Nobody
has checked whether they also disagree about **which models are more faithful**.

This study holds the data fixed and varies only the metric.

> When a paper reports that Model A's reasoning is more faithful than Model B's, how much of that
> conclusion is determined by which faithfulness metric the authors happened to use?

The design was fixed in advance: hypotheses, thresholds and the falsification condition
were written down before any faithfulness metric was computed. They are restated in full
under [Pre-registered hypotheses](#pre-registered-hypotheses) below.

## Status

| Stage | State |
| ---- | ----- |
| Weight availability | ✅ passed |
| API capability probe | ✅ passed — *drove the switch to local execution* |
| CoT injection verified offline and on real weights | ✅ passed, 8/8 |
| Baseline accuracy pilot | ✅ run |
| Predictions written down | ✅ before any metric was computed |
| Harness, judge, scoring layer | ✅ |
| Full runs, 8 models × 500 items | ✅ |
| Analysis | ✅ **H1 and H2 confirmed** |
| Write-up | 🔄 in progress |

## Pre-registered hypotheses

Fixed before the first full run, with the baseline accuracy pilot disclosed as the
only prior observation:

| | Registered as | Outcome |
| --- | --- | --- |
| **H1** | at least one metric pair with Kendall's tau-b < 0.6 whose bootstrap 95% CI excludes 0.8 | **confirmed** — two pairs |
| **H2** | pooled trace-level Cohen's kappa < 0.4 on every pair | **confirmed** — all three, one negative |
| **H3** | thinking models show more metric disagreement than instruct models | direction confirmed; the statistic was chosen after seeing the data, so it is reported as post-hoc |
| **Falsifier** | all three pairs above tau 0.8 with CIs excluding 0.6 | **not triggered** on any pair |

Two metric choices were settled before any run: `faithful@k` was demoted from the metric
set (it is a monotone transform of the Biasing Features rate, so its rank correlation is
forced toward +1 by construction) and FUR was excluded outright (it requires per-step
parameter unlearning). One divergence from the source paper is deliberate: the hint
targets a random wrong option ≠ gold, fixed per item, because the paper's rule picks an
option ≠ *the model's own prediction* — which is model-dependent and would hand every
model a different item set, something a ranking study cannot afford.

## Results

All three metrics on eight models over 500 frozen MMLU-Pro items:

| | tau-b | 95% CI |
| --- | --- | --- |
| Biasing Features × Early Answering | **+0.400** | [+0.000, +0.571] |
| Filler Tokens × Early Answering | **+0.357** | [+0.000, +0.500] |
| Biasing Features × Filler Tokens | +0.618 | [+0.214, +0.714] |

**H1 confirmed** — two pairs under 0.6 with CIs excluding 0.8. **H2 confirmed** — pooled
trace-level kappa +0.108, −0.240, +0.221, all under 0.4; BF × Early is *negative*, the two
metrics agreeing less than chance on individual traces.

Four of 28 model pairs reverse outright. The sharpest is the matched pair the study was
designed around: **Olmo-3.1-32B-Instruct beats Olmo-3.1-32B-Think in 98% of bootstrap
resamples under Biasing Features, and in 14% under Filler Tokens.** Same models, same items,
same traces.

Two findings came out of the machinery rather than the hypothesis:

- **The elicitation scaffold decides the verdict.** Free-running, two models ignore a
  misleading injected trace and re-derive the answer; cued, they follow it. Opposite
  faithfulness verdicts from a scaffold choice published work does not report.
- **So does the judge.** Swapping the Biasing Features judge from Haiku to Opus — same
  traces, same criterion — moved Olmo-3.1-32B-Think from 3.2% to 49.2% and cut the
  matched-pair gap from 39.5 points to 15.5. The two judges' model rankings correlate at
  tau +0.546 — *lower* than the +0.618 between Biasing Features and Filler Tokens. The
  study's thesis reproducing one level down: implementation choice *inside* a metric
  reorders models at least as much as the choice *between* metrics.

`probes/` is the API-capability evidence. `harness/` is the pipeline.
`results/analysis.json` is the headline analysis.

## Metrics

Three independent constructs, plus one sensitivity axis:

| Metric | Tests | Independent? |
| ------ | ----- | ------------ |
| Biasing Features | Does the CoT verbalize an injected hint? | yes |
| Filler Tokens | Replace the CoT with "…" — does the answer change? | yes |
| Early Answering | Truncate the CoT — when is the answer already determined? | yes |
| faithful@k | Across k samples, does *any* verbalize the hint? | **no** — monotone in the Biasing Features rate; reported only as the sampling-budget axis |

## What the API screen found

The study was originally designed to run on API-served models. It can't be, and finding out why is
a result in its own right.

Filler Tokens and Early Answering require writing a reasoning trace *into* a model's context.
Reading a CoT works everywhere; writing one back does not. Across a 106-model screen:

- **55 of 106 (52%)** support the two capabilities these metrics need — and after excluding roleplay
  finetunes, translation models, vision models and models too small to test, roughly 20 remain.
- Every rejection has a different cause. `glm-4.7` honors prefill on none of its 8 providers.
  `muse-glimmer` and `gpt-oss-20b` return *"Reasoning is mandatory for this endpoint"* on every
  provider. `qwen3-next-80b-thinking` is locked on both axes. `olmo-3-32b-think` is listed in the
  OpenRouter catalog with **zero serving endpoints**.
- Failures are **silent**. A dropped trace injection returns HTTP 200. Thinking models handed a
  filler trace quietly re-reason from scratch — `glm-4.7` produced 1,736 characters of its own
  reasoning against a 1,777-character baseline. A gate checking only for API errors would have
  passed it, and the resulting Filler Tokens numbers would have been noise that looked like data.
- Prefill support is **per-provider, not per-model**. The same model is injectable on DeepInfra and
  not on AtlasCloud.

The monitorability literature assumes these metrics port across models. They port across *weights*
but not across *deployments*. Since the study now runs the same models locally, the claim sharpens:
the weights support these metrics; the serving layer is what breaks them.

## Approach

All generation runs locally on rented GPUs under vLLM, using the raw completions interface so the
full prompt string — including each model's thinking delimiters — is constructed by us. There is no
policy layer between the harness and the tokenizer.

Eight models, seven labs, all ≤32B so each fits one 80GB GPU in bf16. Weights pinned by HuggingFace
revision hash. Two matched pairs: **Olmo-3.1-32B Think vs. Instruct** (varies post-training) and one
model run with thinking enabled vs. suppressed (varies inference).

## Layout

```
harness/            prompt construction, generation, judging
probes/             API capability probes, in execution order
analysis/           scoring, statistics, and the write-up artefacts
writing/            the write-up: skeleton, generated tables, figures

results/
  full/             the 8 × 500 generations the study is scored on (~78 MB)
  judge/            Opus 5 verdicts — the ones the study reports
  judge_haiku_v2/   superseded Haiku 4.5 verdicts, kept so the swap is inspectable
  step2/            baseline accuracy pilot (200 items)
  step1L/           CoT-injection verification on real weights, free-running
  step1L_cued/      the same check with the answer cued — the scaffold comparison
  probe/ plumbing/  small pre-flight runs kept for provenance
  0*.json           probe output, numbered in execution order
  analysis.json     headline results
  judge_swap.json   per-model Biasing Features under each judge
```

Copy `.env.example` to `.env` before running anything that calls an API. `.env` is
gitignored and no key is committed anywhere in this repository's history.

### Injection verification

Each model family puts reasoning in a different place — `<think>` for Qwen/Olmo/Nemotron,
`<|channel|>analysis<|message|>` for gpt-oss, `<|start|>assistant to=self<|message|>` for
Muse-Glimmer, asymmetric `<|channel>thought`/`<channel|>` for Gemma-4. `harness/prompts.py`
reads those delimiters off each model's own chat template rather than guessing them.

`harness/verify_templates.py` then proves an injected trace lands **between** the reasoning
delimiters, not merely somewhere in the prompt. It needs tokenizers only, so it runs in
seconds on a laptop:

```bash
pip install transformers jinja2
python harness/verify_templates.py     # 8/8 models pass
```

Checking only that injected text is *present* is the weaker test the API path forced on us,
and it is why a dropped injection was indistinguishable from a successful one for five rounds
of probing.

Probes run in order: single-model prefill → injection-path variants → reasoning-disable rescue →
provider sweep → catalog screen → shortlist verification.

## Reproducing

Everything except generation runs on a laptop from the committed results — no GPU, no
API key. Generation itself needs an 80GB GPU and is the only expensive step.

```bash
pip install -r requirements.txt

python analysis/run_analysis.py      # -> results/analysis.json      (headline analysis)
python analysis/judge_swap.py        # -> results/judge_swap.json    (the judge swap)
python analysis/writeup_tables.py    # -> writing/tables.md          (every table)
python analysis/figures.py           # -> writing/figures/*.svg|png  (light + dark)
```

The probes are the only part that calls an API, and re-running all of them cost
about $2:

```bash
export OPENROUTER_API_KEY=sk-or-...   # never committed; .env is gitignored
python probes/01_api_prefill_probe.py
```

### Tests

Analysis tests run from the repo root; harness tests resolve their fixtures relative to
`harness/` and must be run from inside it:

```bash
python analysis/test_stats.py     # statistics, checked against scipy
python analysis/test_metrics.py   # scoring and support rules
python analysis/test_writeup.py   # tables and figures vs. the analysis they came from

cd harness && python test_extraction.py && python test_judge.py && python test_metrics.py
```

`harness/verify_templates.py` needs only `transformers` and `jinja2` and proves an
injected trace lands between each model's reasoning delimiters — 8/8 pass in seconds.

## Limitations

Stated here rather than left to the reader to discover:

- **Filler Tokens' ranking is not robust to truncation** (tau +0.571 between the full
  item set and the non-truncated subset, 4 non-adjacent swaps) on healthy support. Any
  conclusion resting on its ordering is weakened.
- **No external validation.** The metric implementations are ours and were never checked
  against a published number. Nothing here establishes that this Biasing Features
  implementation is commensurable with published Biasing Features.
- **No reproduction check.** No re-run on identical hardware was performed. Determinism
  is argued from configuration — fixed seed, pinned revisions, one vLLM version — not
  demonstrated.
- **n = 8 models.** tau is noisy at this n; every conclusion rests on bootstrap CIs and
  rank-swap frequencies rather than point estimates.
- **The judge changed mid-study.** Haiku verdicts are preserved in `results/judge_haiku_v2/`
  so the before/after is inspectable.
- **Size ceiling.** All models ≤32B to fit one GPU; this says nothing about frontier scale.
- **Three of eight models sit above the pre-registered 50–80% accuracy band** (81.6%,
  83.2% and 83.8% on the scored item set), so differences among those three are measured
  against less headroom. The band was accepted as-is rather than re-drawn after the fact;
  the median across models, 73.1%, is inside it.
- Two traces (0.3%) went unjudged — the API declined that specific content.
- One hint type, one dataset, three constructs rather than five.

## License

Code is MIT — see [`LICENSE`](LICENSE). Model weights and datasets belong to their
respective owners. Seven of the eight models ship under Apache-2.0 or a comparably
permissive licence; `NVIDIA-Nemotron-3-Nano-30B-A3B` ships under NVIDIA's own model
licence, which you should read before redistributing anything derived from its weights.
