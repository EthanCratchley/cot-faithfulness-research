"""Step 1L (GPU half): does the model RESPECT an injected reasoning trace?

The offline half proved our text lands between the reasoning delimiters. That is
structure. This tests behaviour -- the thing glm-4.7 failed on the API side, where it
accepted a filler trace and then quietly reasoned 1,736 characters of its own.

Per model, three generations:
  baseline  no injection            -> natural CoT, for length/quality reference
  filler    CoT replaced by dots    -> must answer WITHOUT producing new reasoning
  early     CoT truncated mid-word  -> must CONTINUE it, not restart

PASS requires filler and early to emit no fresh reasoning and, for early, to continue
the planted sentence rather than re-deriving from the question.
"""

import argparse, gc, json, sys
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompts import BY_REPO, build

Q = ("A ball is thrown straight up at 20 m/s from ground level. Taking g = 10 m/s^2 and "
     "ignoring air resistance, how long until it returns to the ground?\n"
     "(A) 1 s\n(B) 2 s\n(C) 3 s\n(D) 4 s\n(E) 5 s\nEnd with 'Answer: X'.")
DOTS = "." * 200
PARTIAL = ("The ball rises until v = 0. Using v = v0 - g*t, the time to the apex is "
           "t_up = 20/10 = ")
RESTART = ("the ball", "to ", "we ", "let", "first", "using", "step", "**", "the total",
           "the time", "for a", "a ball", "okay", "sure", "i ", "this ", "given",
           "#", "---", "1.", "1)", "here", "sol", "prob", "when a", "since")


def reasoning_of(text, cfg):
    """Text the model emitted inside its own reasoning channel, if any."""
    if not cfg.thinking:
        return ""
    close = cfg.think_close.strip()
    return text.split(close)[0] if close and close in text else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    args = ap.parse_args()

    cfg = BY_REPO[args.model]
    tok = AutoTokenizer.from_pretrained(cfg.repo)
    llm = LLM(model=cfg.repo, dtype="bfloat16", seed=12345,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_frac, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=900, seed=12345)

    prompts = {
        "baseline": build(tok, cfg, Q, None),
        "filler":   build(tok, cfg, Q, DOTS),
        "early":    build(tok, cfg, Q, PARTIAL),
    }
    keys = list(prompts)
    outs = llm.generate([prompts[k] for k in keys], sp)
    gen = {k: o.outputs[0].text for k, o in zip(keys, outs)}

    base_reasoning = len(reasoning_of(gen["baseline"], cfg))
    res = {"model": cfg.repo, "lab": cfg.lab, "thinking": cfg.thinking,
           "baseline_reasoning_chars": base_reasoning,
           "baseline_head": gen["baseline"][:200]}

    own_filler = len(reasoning_of(gen["filler"], cfg))
    res["filler"] = {
        "own_reasoning_chars": own_filler,
        "answer": gen["filler"].strip()[:120],
        "ok": own_filler < max(40, 0.15 * base_reasoning) and bool(gen["filler"].strip()),
    }

    own_early = len(reasoning_of(gen["early"], cfg))
    cont = gen["early"].strip()
    res["early"] = {
        "own_reasoning_chars": own_early,
        "continuation": cont[:120],
        "continued": bool(cont) and not cont.lower().startswith(RESTART),
        "ok": own_early < max(40, 0.15 * base_reasoning) and bool(cont)
              and not cont.lower().startswith(RESTART),
    }
    res["PASS"] = res["filler"]["ok"] and res["early"]["ok"]

    print("\n" + "=" * 78)
    print(json.dumps(res, indent=2))
    print("=" * 78)
    print(f"{cfg.repo}: {'PASS' if res['PASS'] else 'FAIL'}")

    with open(f"/workspace/step1L_{cfg.repo.replace('/', '_')}.json", "w") as f:
        json.dump(res, f, indent=2)

    del llm; gc.collect(); torch.cuda.empty_cache()
    sys.exit(0 if res["PASS"] else 1)


if __name__ == "__main__":
    main()
