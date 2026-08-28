"""Step 4/6: generate every trace the three core metrics need, for one model.

§5 fixes one shared trace set -- the HINTED traces -- so Filler Tokens and Early
Answering both operate on the hinted CoT, not on a separate unhinted one. The
unhinted baseline is generated too, but only to detect which items the hint flipped:
Biasing Features is defined solely on flipped traces.

Two phases, because phase 2's prompts are built out of phase 1's output:

  phase 1   baseline  unhinted, uncued  -> answer, for flip detection
            hinted    hinted,   uncued  -> THE trace, plus the hinted answer
  phase 2   filler    hinted CoT -> dots,      cued -> answer
            early     hinted CoT -> 5 prefixes, cued -> 5 answers

8 generations per item. Uncued where we want the model's own behaviour, cued where an
injected trace must be the thing the answer is read from (§5 Answer elicitation).

This script only produces traces. Scoring lives in analysis/; nothing here computes a
faithfulness number, so running it cannot leak a result into the pre-registration.
"""
import argparse, gc, json, time

from answers import ANSWER_CUE, extract
from early_answering import FRACTIONS, truncate
from hints import apply as apply_hint, pick_target
from items import format_question, load
from prompts import BY_REPO, build
from reasoning import closed_block, reasoning_of

DOTS = "." * 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="/workspace/data/items_pool_500.json")
    ap.add_argument("--out", default="/workspace")
    ap.add_argument("--limit", type=int, default=0, help="first N items, for plumbing checks")
    ap.add_argument("--max-model-len", type=int, default=12288)
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
    items = meta["items"][:args.limit] if args.limit else meta["items"]
    tok = AutoTokenizer.from_pretrained(cfg.repo)
    llm = LLM(model=cfg.repo, dtype=args.dtype, seed=12345,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_frac,
              max_num_seqs=args.max_num_seqs, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=12345,
                        skip_special_tokens=False)
    # An injected trace is followed only by an answer, so a large budget buys nothing
    # and a runaway generation costs real GPU time across 6 prompts per item.
    sp_short = SamplingParams(temperature=0.0, max_tokens=64, seed=12345,
                              skip_special_tokens=False)

    t0 = time.time()
    hinted_bodies, targets = [], []
    for it in items:
        body, t = apply_hint(it, format_question(it))
        hinted_bodies.append(body)
        targets.append(t)

    p1 = ([build(tok, cfg, format_question(it)) for it in items]
          + [build(tok, cfg, b) for b in hinted_bodies])
    o1 = llm.generate(p1, sp)
    n = len(items)
    base_txt = [o.outputs[0].text for o in o1[:n]]
    hint_txt = [o.outputs[0].text for o in o1[n:]]
    print(f"phase 1 done in {(time.time()-t0)/60:.1f} min")

    # The hinted CoT is the object every downstream prompt is built from.
    cots = [reasoning_of(t, cfg) for t in hint_txt]

    t1 = time.time()
    p2, index = [], []
    for i, (it, cot) in enumerate(zip(items, cots)):
        p2.append(build(tok, cfg, hinted_bodies[i], DOTS, ANSWER_CUE))
        index.append((i, "filler"))
        for fr in FRACTIONS:
            p2.append(build(tok, cfg, hinted_bodies[i], truncate(tok, cot, fr), ANSWER_CUE))
            index.append((i, f"early_{fr}"))
    o2 = llm.generate(p2, sp_short)
    print(f"phase 2 done in {(time.time()-t1)/60:.1f} min")

    gen2 = {}
    for (i, key), o in zip(index, o2):
        gen2.setdefault(i, {})[key] = o.outputs[0].text

    rows = []
    for i, it in enumerate(items):
        nopt = len(it["options"])
        g = gen2[i]
        rows.append({
            "question_id": it["question_id"], "category": it["category"],
            "n_options": nopt, "gold": it["answer"], "hint_target": targets[i],
            "baseline_answer": extract(base_txt[i], nopt),
            "hinted_answer": extract(hint_txt[i], nopt),
            "hinted_cot_chars": len(cots[i]),
            "hinted_block_closed": closed_block(hint_txt[i], cfg),
            "hinted_truncated": len(o1[n + i].outputs[0].token_ids) >= args.max_tokens,
            "filler_answer": extract(ANSWER_CUE + g["filler"], nopt),
            "early_answers": {str(fr): extract(ANSWER_CUE + g[f"early_{fr}"], nopt)
                              for fr in FRACTIONS},
            # Judging Biasing Features needs the hinted CoT itself; keep every raw
            # generation so a scoring change never costs another GPU run.
            "raw": {"baseline": base_txt[i], "hinted": hint_txt[i], **g},
        })

    res = {"model": cfg.repo, "lab": cfg.lab, "thinking": cfg.thinking,
           "dtype": args.dtype, "seed": 12345, "max_tokens": args.max_tokens,
           "items_file": args.items, "items_seed": meta["seed"],
           "items_fingerprint": meta["fingerprint"], "n_items": len(items),
           "fractions": list(FRACTIONS), "elapsed_sec": time.time() - t0,
           "rows": rows}

    flipped = sum(r["baseline_answer"] != r["hint_target"]
                  and r["hinted_answer"] == r["hint_target"] for r in rows)
    print(f"\n{cfg.repo}: {len(rows)} items, {flipped} flipped to the hint "
          f"({flipped/len(rows):.1%}) -- Biasing Features support")
    print(f"  hinted traces truncated: {sum(r['hinted_truncated'] for r in rows)}")
    print(f"  unparsed hinted answers: {sum(r['hinted_answer'] is None for r in rows)}")

    path = f"{args.out}/metrics_{cfg.repo.replace('/', '_')}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"  wrote {path}  ({time.time()-t0:.0f}s total)")

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
