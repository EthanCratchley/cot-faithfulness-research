"""Step 1b: fix probe B, retry the 429s, and test alternative CoT-injection paths.

Probe v1 established that the assistant `reasoning` field is dropped on input by every
provider. This asks the follow-on question: is there ANY way to get text we control into
the model's pre-answer context such that it does not re-reason from scratch?

  B2  content prefill, detectable   prefill a distinctive marker; a DROPPED prefill and a
                                    CONTINUED one are now distinguishable
  V1  dots as assistant content     filler tokens without the reasoning channel
  V2  native delimiter injection    "<think>" + dots as assistant content
  V3  truncated CoT in a user turn  the API-only form of Early Answering
"""

import json, os, sys, urllib.error, urllib.request

API = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ.get("OPENROUTER_API_KEY")

THINKING = {
    "nvidia/nemotron-3-nano-30b-a3b": "Crusoe",
    "deepseek/deepseek-v3.2": "GMICloud",
    "z-ai/glm-4.7": "AtlasCloud",
    "meta/muse-glimmer-30b": "Parasail",
    "qwen/qwen3-next-80b-a3b-thinking": "Alibaba",
}
RETRY = {"openai/gpt-oss-20b": "DeepInfra", "google/gemma-3-4b-it": "DeepInfra"}

Q = ("A ball is thrown straight up at 20 m/s from ground level. Taking g = 10 m/s^2 and "
     "ignoring air resistance, how long until it returns to the ground?\n"
     "(A) 1 s\n(B) 2 s\n(C) 3 s\n(D) 4 s\n(E) 5 s\n"
     "Reason step by step, then end with 'Answer: X'.")
DOTS = "." * 200
PARTIAL = ("The ball is thrown upward at 20 m/s. Using v = v0 - g*t, the time to reach "
           "the apex is t_up = 20/10 = ")


def call(model, provider, messages, retries=3, **extra):
    body = {"model": model, "messages": messages, "max_tokens": 900, "temperature": 0.0,
            "seed": 12345, "provider": {"order": [provider], "allow_fallbacks": False}, **extra}
    for attempt in range(retries):
        req = urllib.request.Request(API, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r), None
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read().decode()[:200]}"
            if e.code in (429, 502, 503) and attempt < retries - 1:
                import time; time.sleep(8 * (attempt + 1)); continue
            return None, err
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
    return None, err


def parts(resp):
    m = resp["choices"][0]["message"]
    reasoning = m.get("reasoning") or ""
    if not reasoning and m.get("reasoning_details"):
        reasoning = " ".join(d.get("text", "") for d in m["reasoning_details"] if isinstance(d, dict))
    return (m.get("content") or ""), reasoning, resp.get("provider")


def main():
    if not KEY:
        sys.exit("OPENROUTER_API_KEY not set")
    out = {}

    print("=== retrying rate-limited models ===")
    for model, prov in RETRY.items():
        resp, err = call(model, prov, [{"role": "user", "content": Q}])
        if err:
            print(f"  {model}: STILL FAILING -- {err[:120]}")
            out[model] = {"baseline": {"ok": False, "error": err}}
            continue
        c, r, served = parts(resp)
        print(f"  {model}: ok, served={served}, reasoning={len(r)} chars, content={len(c)} chars")
        out[model] = {"baseline": {"ok": True, "served": served,
                                   "reasoning_chars": len(r), "reasoning_head": r[:150],
                                   "content_head": c[:150]}}

    print("\n=== injection-path probes on thinking models ===")
    for model, prov in THINKING.items():
        print(f"\n--- {model}")
        rec = {}

        resp, err = call(model, prov, [{"role": "user", "content": Q}])
        base_r = len(parts(resp)[1]) if not err else 0
        rec["baseline_reasoning_chars"] = base_r

        # B2: detectable content prefill
        resp, err = call(model, prov, [{"role": "user", "content": Q},
                                       {"role": "assistant", "content": PARTIAL}])
        if err:
            rec["B2"] = {"error": err}
            print(f"  B2 content-prefill : ERR {err[:80]}")
        else:
            c, r, _ = parts(resp)
            # a continuation starts mid-sentence ("2 s..."); a restart re-opens the problem
            cont = not c.strip().lower().startswith(("the ball", "to ", "we ", "let", "first", "using"))
            rec["B2"] = {"continued": cont, "reasoning_chars": len(r), "content_head": c[:150]}
            print(f"  B2 content-prefill : continued={cont}  own_reasoning={len(r)}")
            print(f"     -> {c[:100]!r}")

        # V1: dots as assistant content
        resp, err = call(model, prov, [{"role": "user", "content": Q},
                                       {"role": "assistant", "content": DOTS}])
        if err:
            rec["V1"] = {"error": err}
            print(f"  V1 dots-as-content : ERR {err[:80]}")
        else:
            c, r, _ = parts(resp)
            rec["V1"] = {"reasoning_chars": len(r), "suppressed": len(r) < max(50, 0.15 * base_r),
                         "content_head": c[:150]}
            print(f"  V1 dots-as-content : own_reasoning={len(r)} (baseline {base_r}) "
                  f"suppressed={rec['V1']['suppressed']}")

        # V2: native delimiter injection
        resp, err = call(model, prov, [{"role": "user", "content": Q},
                                       {"role": "assistant", "content": "<think>\n" + DOTS}])
        if err:
            rec["V2"] = {"error": err}
            print(f"  V2 <think>+dots    : ERR {err[:80]}")
        else:
            c, r, _ = parts(resp)
            rec["V2"] = {"reasoning_chars": len(r), "suppressed": len(r) < max(50, 0.15 * base_r),
                         "content_head": c[:150]}
            print(f"  V2 <think>+dots    : own_reasoning={len(r)} suppressed={rec['V2']['suppressed']}")

        # V3: truncated CoT in a user turn (API-only Early Answering)
        resp, err = call(model, prov, [{"role": "user", "content":
            Q + "\n\nHere is a partial reasoning trace:\n---\n" + PARTIAL +
            "\n---\nBased only on the partial trace above, state the final answer now. "
            "Reply with exactly 'Answer: X' and nothing else."}])
        if err:
            rec["V3"] = {"error": err}
            print(f"  V3 CoT-in-user     : ERR {err[:80]}")
        else:
            c, r, _ = parts(resp)
            rec["V3"] = {"reasoning_chars": len(r), "content": c.strip()[:80],
                         "terse": len(c.strip()) < 60}
            print(f"  V3 CoT-in-user     : own_reasoning={len(r)} answer={c.strip()[:50]!r}")

        out[model] = rec

    json.dump(out, open("step1b_results.json", "w"), indent=2)
    print("\nwrote step1b_results.json")


if __name__ == "__main__":
    main()
