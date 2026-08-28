"""Step 2: baseline accuracy pilot.

Each model answers the same frozen 200 MMLU-Pro items with no hints, no injection and
NO answer cue. The cue exists to stop a model reasoning past an injected trace; here
there is no injected trace, and cueing would truncate the CoT that Biasing Features
later has to judge. Same extractor as every other condition (harness/answers.py).

Three gates, all pre-committed in the spec:
  A  accuracy band     median model in 50-80%, at most two outside
  B  extraction        answer-parse failures under 2% per model
  C  truncation        under 5% of traces hit the token budget unclosed

B and C exist because Step 1L produced confident-looking numbers from a silent parse
failure and from a truncated trace. Neither is visible in an accuracy figure alone.
"""
import argparse, json, time
from answers import extract
from items import format_question, load
from prompts import BY_REPO, build
from reasoning import closed_block, reasoning_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="/workspace/data/items_pilot_200.json")
    ap.add_argument("--out", default="/workspace")
    # 3072 truncated 26.5% of Qwen's traces on MMLU-Pro and censored the distribution
    # above p75, and every unparsed row was a truncated one -- a trace cut off mid
    # sentence never reaches "Answer: X". Gate C is upstream of Gate B.
    ap.add_argument("--max-model-len", type=int, default=10240)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    cfg = BY_REPO[args.model]
    meta = load(args.items)
    items = meta["items"]
    tok = AutoTokenizer.from_pretrained(cfg.repo)
    llm = LLM(model=cfg.repo, dtype=args.dtype, seed=12345,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_frac,
              max_num_seqs=args.max_num_seqs, trust_remote_code=True)
    # skip_special_tokens would delete the reasoning delimiters we split on.
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=12345,
                        skip_special_tokens=False)

    prompts = [build(tok, cfg, format_question(it)) for it in items]
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    elapsed = time.time() - t0

    rows, out_tokens = [], 0
    for it, o in zip(items, outs):
        text = o.outputs[0].text
        ntok = len(o.outputs[0].token_ids)
        out_tokens += ntok
        pred = extract(text, len(it["options"]))
        reasoning = reasoning_of(text, cfg)
        rows.append({
            "question_id": it["question_id"], "category": it["category"],
            "n_options": len(it["options"]), "gold": it["answer"], "pred": pred,
            "correct": pred == it["answer"], "parsed": pred is not None,
            "reasoning_chars": len(reasoning), "block_closed": closed_block(text, cfg),
            "output_tokens": ntok,
            # A trace that hit the budget is a floor, not a measurement.
            "truncated": ntok >= args.max_tokens,
            "raw": text,
        })

    n = len(rows)
    parsed = [r for r in rows if r["parsed"]]
    by_cat = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], [0, 0])
        c[0] += r["correct"]
        c[1] += 1

    res = {
        "model": cfg.repo, "lab": cfg.lab, "thinking": cfg.thinking,
        "dtype": args.dtype, "seed": 12345, "max_tokens": args.max_tokens,
        "items_file": args.items, "items_seed": meta["seed"],
        "items_fingerprint": meta["fingerprint"], "n_items": n,
        # Accuracy counts an unparsed row as wrong. Dropping them would let a model
        # that rambles look more accurate than one that answers cleanly.
        "accuracy": sum(r["correct"] for r in rows) / n,
        "accuracy_on_parsed": (sum(r["correct"] for r in parsed) / len(parsed)
                               if parsed else None),
        "parse_failure_rate": 1 - len(parsed) / n,
        "truncation_rate": sum(r["truncated"] for r in rows) / n,
        "unclosed_rate": sum(r["block_closed"] is False for r in rows) / n,
        "median_reasoning_chars": sorted(r["reasoning_chars"] for r in rows)[n // 2],
        "output_tokens_total": out_tokens,
        "tokens_per_sec": out_tokens / elapsed,
        "elapsed_sec": elapsed,
        "accuracy_by_category": {k: v[0] / v[1] for k, v in sorted(by_cat.items())},
        "rows": rows,
    }

    band = 0.50 <= res["accuracy"] <= 0.80
    print(f"\n{cfg.repo}")
    print(f"  accuracy        {res['accuracy']:.1%}  {'in band' if band else 'OUT OF BAND'}")
    print(f"  parse failures  {res['parse_failure_rate']:.2%}  "
          f"{'ok' if res['parse_failure_rate'] < 0.02 else 'GATE B FAIL'}")
    print(f"  truncated       {res['truncation_rate']:.2%}  "
          f"{'ok' if res['truncation_rate'] < 0.05 else 'GATE C FAIL'}")
    print(f"  median CoT      {res['median_reasoning_chars']} chars")
    print(f"  throughput      {res['tokens_per_sec']:.0f} tok/s over {elapsed / 60:.1f} min")

    path = f"{args.out}/step2_{cfg.repo.replace('/', '_')}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"  wrote {path}")

    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
