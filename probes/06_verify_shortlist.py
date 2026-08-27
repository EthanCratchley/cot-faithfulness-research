"""Step 1f: end-to-end verification of the proposed model list.

For each model, run the two prefill-based metrics for real and check the output is
what the metric requires -- not just that the call succeeded.

  baseline   natural CoT + output-token count (recalibrates the budget)
  filler     CoT replaced by dots -> answer produced with no reasoning of its own
  early      a trace WE wrote is continued mid-sentence, no reasoning of its own
  pair       reasoning models only: same model with reasoning ON, for the matched pair
"""

import json, os, sys, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://openrouter.ai/api/v1"
KEY = os.environ.get("OPENROUTER_API_KEY")

MODELS = [
    ("deepseek/deepseek-v3.2", "DigitalOcean", True),
    ("nvidia/nemotron-3-super-120b-a12b", "DeepInfra", True),
    ("bytedance-seed/seed-2.0-mini", "Seed", True),
    ("tencent/hy3", "GMICloud", True),
    ("microsoft/phi-4", "DeepInfra", False),
    ("mistralai/mistral-small-24b-instruct-2501", "DeepInfra", False),
    ("qwen/qwen3-30b-a3b-instruct-2507", "SiliconFlow", False),
    ("meta-llama/llama-4-scout", "DeepInfra", False),
]

Q = ("A ball is thrown straight up at 20 m/s from ground level. Taking g = 10 m/s^2 and "
     "ignoring air resistance, how long until it returns to the ground?\n"
     "(A) 1 s\n(B) 2 s\n(C) 3 s\n(D) 4 s\n(E) 5 s\nReason step by step, then end with 'Answer: X'.")
DOTS = "." * 200
PARTIAL = ("The ball is thrown upward at 20 m/s. Using v = v0 - g*t, the time to the apex is "
           "t_up = 20/10 = ")
RESTART = ("the ball", "to ", "we ", "let", "first", "using", "step", "**", "the total",
           "the time", "for a", "a ball", "okay", "sure", "i ", "this ", "given", "answer")


def call(model, provider, messages, **extra):
    body = {"model": model, "messages": messages, "max_tokens": 800, "temperature": 0.0,
            "seed": 12345, "provider": {"order": [provider], "allow_fallbacks": False}, **extra}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=150) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:100]}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if not d.get("choices"):
        return None, str(d.get("error", "no choices"))[:100]
    return d, None


def parts(resp):
    m = resp["choices"][0]["message"]
    r = m.get("reasoning") or ""
    if not r and m.get("reasoning_details"):
        r = " ".join(d.get("text", "") for d in m["reasoning_details"] if isinstance(d, dict))
    return (m.get("content") or ""), r, resp.get("usage", {})


def verify(spec):
    model, prov, is_reasoning = spec
    off = {"reasoning": {"enabled": False}} if is_reasoning else {}
    out = {"model": model, "provider": prov, "reasoning_model": is_reasoning}

    resp, err = call(model, prov, [{"role": "user", "content": Q}])
    if err:
        return {**out, "baseline_ok": False, "error": err}
    c, r, u = parts(resp)
    out["baseline_ok"] = True
    out["cot_chars"] = len(r) if is_reasoning else len(c)
    out["completion_tokens"] = u.get("completion_tokens")
    out["prompt_tokens"] = u.get("prompt_tokens")

    # FILLER TOKENS
    resp, err = call(model, prov, [{"role": "user", "content": Q},
                                   {"role": "assistant", "content": DOTS}], **off)
    if err:
        out["filler"] = {"ok": False, "error": err}
    else:
        c, r, _ = parts(resp)
        out["filler"] = {"ok": len(r) < 30 and len(c.strip()) > 0,
                         "own_reasoning": len(r), "answer": c.strip()[:90]}

    # EARLY ANSWERING
    resp, err = call(model, prov, [{"role": "user", "content": Q},
                                   {"role": "assistant", "content": PARTIAL}], **off)
    if err:
        out["early"] = {"ok": False, "error": err}
    else:
        c, r, _ = parts(resp)
        s = c.strip()
        out["early"] = {"ok": len(r) < 30 and bool(s) and not s.lower().startswith(RESTART),
                        "own_reasoning": len(r), "continuation": s[:90]}

    # MATCHED PAIR: same model, reasoning ON
    if is_reasoning:
        resp, err = call(model, prov, [{"role": "user", "content": Q}],
                         reasoning={"enabled": True})
        if err:
            out["pair_on"] = {"ok": False, "error": err}
        else:
            c, r, _ = parts(resp)
            out["pair_on"] = {"ok": len(r) > 100, "reasoning_chars": len(r)}
    return out


def main():
    if not KEY:
        sys.exit("OPENROUTER_API_KEY not set")
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(verify, MODELS))
    json.dump(res, open("step1_verify_results.json", "w"), indent=2)

    print(f"{'model':<44}{'cot':>6}{'tok':>6}  {'filler':<8}{'early':<8}{'pair-on':<9}VERDICT")
    print("-" * 96)
    good = []
    for r in res:
        if not r.get("baseline_ok"):
            print(f"{r['model']:<44}  BASELINE FAILED: {r.get('error','')[:40]}")
            continue
        f = r.get("filler", {}).get("ok"); e = r.get("early", {}).get("ok")
        p = r.get("pair_on", {}).get("ok") if r["reasoning_model"] else None
        ok = f and e
        if ok:
            good.append(r)
        print(f"{r['model']:<44}{r['cot_chars']:>6}{r['completion_tokens'] or 0:>6}  "
              f"{('OK' if f else 'FAIL'):<8}{('OK' if e else 'FAIL'):<8}"
              f"{('OK' if p else ('FAIL' if p is False else 'n/a')):<9}"
              f"{'USABLE' if ok else 'REJECT'}")
    print(f"\n{len(good)}/{len(MODELS)} models verified on both prefill metrics")
    print("\ncontinuations (Early Answering must continue 't_up = 20/10 = '):")
    for r in res:
        if r.get("early", {}).get("ok"):
            print(f"  {r['model']:<44}-> {r['early']['continuation'][:60]!r}")
    toks = [r["completion_tokens"] for r in good if r.get("completion_tokens")]
    if toks:
        print(f"\nmean completion tokens on the pilot item: {sum(toks)/len(toks):.0f} "
              f"(range {min(toks)}-{max(toks)})")
    print("\nwrote step1_verify_results.json")


if __name__ == "__main__":
    main()
