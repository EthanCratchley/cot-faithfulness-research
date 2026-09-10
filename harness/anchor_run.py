"""Step 5 anchor: reproduce arXiv:2512.23032's professor-hint result on ARC-Easy.

Runs the PAPER's pipeline, not ours, which is the whole point -- an anchor that uses our
choices tests nothing about theirs. Two places this deliberately differs from
run_metrics.py:

  - The hint target is a random option != the model's OWN unhinted prediction
    (hints.pick_target_paper), so baseline must be generated before targets can be
    chosen. Our main study uses a fixed, gold-avoiding target because a model-dependent
    intervention would break the fixed-item design; the paper does not rank models
    against each other, so it can afford theirs.
  - Only Biasing Features is produced. Filler and Early are not part of the anchor, so
    there is no phase 2 and no answer cue.
"""
import argparse, gc, json, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="/workspace/data/anchor_arc_easy_500.json")
    ap.add_argument("--out", default="/workspace")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from answers import extract_with_rule
    from hints import apply as apply_hint, pick_target_paper, verbalization_support
    from items import format_question, load
    from prompts import BY_REPO, build
    from reasoning import cot_of

    cfg = BY_REPO[args.model]
    meta = load(args.items)
    items = meta["items"][:args.limit] if args.limit else meta["items"]
    revision = HfApi().model_info(cfg.repo).sha
    print(f"{cfg.repo} pinned at {revision}")
    tok = AutoTokenizer.from_pretrained(cfg.repo, revision=revision)
    llm = LLM(model=cfg.repo, revision=revision, dtype=args.dtype, seed=12345,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_frac,
              trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=12345,
                        skip_special_tokens=False)

    t0 = time.time()
    base_txt = [o.outputs[0].text for o in
                llm.generate([build(tok, cfg, format_question(it)) for it in items], sp)]
    base = [extract_with_rule(t, len(it["options"]))
            for t, it in zip(base_txt, items)]
    print(f"baseline done in {(time.time()-t0)/60:.1f} min")

    # The paper's rule: a random option != the model's own answer. Unparsed baselines
    # have no prediction to avoid, so they cannot be hinted and are recorded as skipped.
    targets, hinted_bodies, idx = [], [], []
    for i, (it, (a, _)) in enumerate(zip(items, base)):
        if a is None:
            continue
        t = pick_target_paper(it, a)
        targets.append(t); idx.append(i)
        hinted_bodies.append(apply_hint(it, format_question(it), target=t)[0])
    print(f"{len(idx)}/{len(items)} items hinted ({len(items)-len(idx)} unparsed baselines)")

    t1 = time.time()
    hint_txt = [o.outputs[0].text for o in
                llm.generate([build(tok, cfg, b) for b in hinted_bodies], sp)]
    print(f"hinted done in {(time.time()-t1)/60:.1f} min")

    rows = []
    for k, i in enumerate(idx):
        it = items[i]; nopt = len(it["options"])
        ha, hrule = extract_with_rule(hint_txt[k], nopt)
        rows.append({
            "question_id": it["question_id"], "n_options": nopt, "gold": it["answer"],
            "hint_target": targets[k],
            "baseline_answer": base[i][0], "baseline_answer_rule": base[i][1],
            "hinted_answer": ha, "hinted_answer_rule": hrule,
            "hinted_cot_chars": len(cot_of(hint_txt[k], cfg)),
            "raw": {"baseline": base_txt[i], "hinted": hint_txt[k]},
        })

    flipped = sum(verbalization_support(r["baseline_answer"], r["hinted_answer"],
                                        r["hint_target"]) for r in rows)
    res = {"model": cfg.repo, "revision": revision, "dataset": meta["dataset"],
           "config": meta.get("config"), "items_fingerprint": meta["fingerprint"],
           "hint_rule": "paper (random option != model's own unhinted answer)",
           "seed": 12345, "dtype": args.dtype, "n_items": len(items),
           "n_hinted": len(rows), "n_flipped": flipped,
           "elapsed_sec": time.time() - t0, "rows": rows}
    path = f"{args.out}/anchor_{cfg.repo.replace('/', '_')}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n{cfg.repo}: {flipped}/{len(rows)} flipped to the hint "
          f"({flipped/len(rows):.1%}) -- Biasing Features support")
    print(f"  wrote {path}")

    del llm; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
