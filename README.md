# Do CoT Faithfulness Metrics Agree?

Several published metrics claim to measure whether a model's chain-of-thought reflects what
actually drove its answer. Each was introduced in a different paper, applied to different models
on different datasets. Recent work shows they **disagree on individual reasoning traces**. Nobody
has checked whether they also disagree about **which models are more faithful**.

This study holds the data fixed and varies only the metric.

> When a paper reports that Model A's reasoning is more faithful than Model B's, how much of that
> conclusion is determined by which faithfulness metric the authors happened to use?

Full design, hypothesis, analysis plan and limitations: **[`spec/design.md`](spec/design.md)**

## Status

| Step | State |
| ---- | ----- |
| Step 0 — weight availability | ✅ passed |
| Step 1 — API capability probe | ✅ passed — *drove the switch to local execution* |
| Step 1L (offline) — CoT injection verified on all 8 tokenizers | ✅ passed |
| Step 1L (GPU) — confirm models continue rather than re-reason | next, needs a pod |
| Steps 2–9 | not started |

`probes/` is Step 1 evidence. `harness/` is the beginning of the real pipeline.

## Metrics

Three independent constructs, plus one sensitivity axis:

| Metric | Tests | Independent? |
| ------ | ----- | ------------ |
| Biasing Features | Does the CoT verbalize an injected hint? | yes |
| Filler Tokens | Replace the CoT with "…" — does the answer change? | yes |
| Early Answering | Truncate the CoT — when is the answer already determined? | yes |
| faithful@k | Across k samples, does *any* verbalize the hint? | **no** — monotone in the Biasing Features rate; reported only as the sampling-budget axis |

## What Step 1 found

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
spec/design.md   the pre-registered design — read this first
harness/         prompt construction and injection verification
probes/          Step 1 capability probes, in execution order
results/         their raw JSON output
```

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

## Reproducing the probes

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
python probes/01_api_prefill_probe.py
```

Total cost of all Step 1 probes was about $2.

## License

Code MIT. See `spec/design.md` for per-model license notes — one model
(`NVIDIA-Nemotron-3-Nano-30B-A3B`) ships under a non-Apache license.
