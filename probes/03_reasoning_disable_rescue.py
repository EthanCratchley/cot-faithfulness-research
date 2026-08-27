"""Step 1c: the rescue tests.

R1  reasoning-disable      does `reasoning:{enabled:false}` / `{max_tokens:0}` /
                           `reasoning_effort:"none"` actually suppress thinking?
R2  filler tokens revived  dots-as-content + reasoning disabled -> does it answer?
R3  early answering revived truncated CoT in user turn + reasoning disabled
R4  cross-provider prefill is prefill honored by ANY provider for a given model?
R5  repin retries          gpt-oss-20b / gemma-3-4b-it away from the 429ing endpoint
"""

import json, os, sys, time, urllib.error, urllib.request

API = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ.get("OPENROUTER_API_KEY")

THINKING = {
    "nvidia/nemotron-3-nano-30b-a3b": "Crusoe",
    "deepseek/deepseek-v3.2": "GMICloud",
    "z-ai/glm-4.7": "AtlasCloud",
    "meta/muse-glimmer-30b": "Parasail",
    "qwen/qwen3-next-80b-a3b-thinking": "Alibaba",
}
# alternate providers to test whether prefill support is provider-specific
CROSS = {
    "deepseek/deepseek-v3.2": ["AtlasCloud", "Novita", "DeepInfra"],
    "z-ai/glm-4.7": ["Novita", "DeepInfra", "StreamLake"],
}
REPIN = {"openai/gpt-oss-20b": ["Darkbloom", "CoreWeave", "Novita", "Parasail"],
         "google/gemma-3-4b-it": ["DeepInfra"]}

Q = ("A ball is thrown straight up at 20 m/s from ground level. Taking g = 10 m/s^2 and "
     "ignoring air resistance, how long until it returns to the ground?\n"
     "(A) 1 s\n(B) 2 s\n(C) 3 s\n(D) 4 s\n(E) 5 s\nEnd with 'Answer: X'.")
DOTS = "." * 200
PARTIAL = ("The ball is thrown upward at 20 m/s. Using v = v0 - g*t, the time to the apex is "
           "t_up = 20/10 = ")

DISABLE = [
    ("enabled_false", {"reasoning": {"enabled": False}}),
    ("max_tokens_0", {"reasoning": {"max_tokens": 0}}),
    ("effort_none", {"reasoning_effort": "none"}),
]


def call(model, provider, messages, retries=2, **extra):
    body = {"model": model, "messages": messages, "max_tokens": 900, "temperature": 0.0,
            "seed": 12345, "provider": {"order": [provider], "allow_fallbacks": False}, **extra}
    err = None
    for i in range(retries):
        req = urllib.request.Request(API, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r), None
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read().decode()[:150]}"
            if e.code in (429, 502, 503) and i < retries - 1:
                time.sleep(6); continue
            return None, err
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
    return None, err


def parts(resp):
    m = resp["choices"][0]["message"]
    r = m.get("reasoning") or ""
    if not r and m.get("reasoning_details"):
        r = " ".join(d.get("text", "") for d in m["reasoning_details"] if isinstance(d, dict))
    return (m.get("content") or ""), r


def main():
    if not KEY:
        sys.exit("OPENROUTER_API_KEY not set")
    out = {}

    print("=== R1: can we disable reasoning? ===")
    working = {}
    for model, prov in THINKING.items():
        resp, err = call(model, prov, [{"role": "user", "content": Q}])
        base = len(parts(resp)[1]) if not err else 0
        print(f"\n{model}  (baseline {base} chars)")
        out.setdefault(model, {})["baseline"] = base
        for name, kw in DISABLE:
            resp, err = call(model, prov, [{"role": "user", "content": Q}], **kw)
            if err:
                print(f"   {name:<14} ERR {err[:70]}")
                out[model][name] = {"error": err}
                continue
            c, r = parts(resp)
            off = len(r) < max(30, 0.1 * base)
            print(f"   {name:<14} reasoning={len(r):<6} suppressed={off}  answer={c.strip()[-12:]!r}")
            out[model][name] = {"reasoning_chars": len(r), "suppressed": off}
            if off and model not in working:
                working[model] = (prov, kw, name)

    print(f"\n=== R2/R3: metrics revived on the {len(working)} model(s) where disable works ===")
    for model, (prov, kw, name) in working.items():
        print(f"\n{model}  (via {name})")
        # R2 filler tokens: dots as the entire reasoning, no thinking allowed
        resp, err = call(model, prov, [{"role": "user", "content": Q},
                                       {"role": "assistant", "content": DOTS}], **kw)
        if err:
            print(f"   R2 filler      ERR {err[:70]}")
        else:
            c, r = parts(resp)
            print(f"   R2 filler      reasoning={len(r)} answer={c.strip()[:60]!r}")
            out[model]["R2_filler"] = {"reasoning_chars": len(r), "content": c.strip()[:120]}
        # R3 early answering: truncated CoT, no thinking allowed
        resp, err = call(model, prov, [{"role": "user", "content":
            Q + "\n\nPartial reasoning so far:\n---\n" + PARTIAL +
            "\n---\nContinue from the partial reasoning and give the final answer now."}], **kw)
        if err:
            print(f"   R3 early-ans   ERR {err[:70]}")
        else:
            c, r = parts(resp)
            print(f"   R3 early-ans   reasoning={len(r)} answer={c.strip()[:60]!r}")
            out[model]["R3_early"] = {"reasoning_chars": len(r), "content": c.strip()[:120]}

    print("\n=== R4: is prefill honored by any alternate provider? ===")
    for model, provs in CROSS.items():
        for prov in provs:
            resp, err = call(model, prov, [{"role": "user", "content": Q},
                                           {"role": "assistant", "content": PARTIAL}])
            if err:
                print(f"   {model:<26} {prov:<12} ERR {err[:60]}")
                continue
            c, r = parts(resp)
            restarted = c.strip().lower().startswith(
                ("the ball", "to ", "we ", "let", "first", "using", "step"))
            print(f"   {model:<26} {prov:<12} reasoning={len(r):<6} "
                  f"prefill_honored={not restarted}  -> {c.strip()[:55]!r}")
            out.setdefault(model, {}).setdefault("cross", {})[prov] = {
                "reasoning_chars": len(r), "prefill_honored": not restarted,
                "content_head": c[:150]}

    print("\n=== R5: repin the rate-limited models ===")
    for model, provs in REPIN.items():
        for prov in provs:
            resp, err = call(model, prov, [{"role": "user", "content": Q}])
            if err:
                print(f"   {model:<24} {prov:<12} ERR {err[:60]}")
                continue
            c, r = parts(resp)
            print(f"   {model:<24} {prov:<12} OK reasoning={len(r)} answer={c.strip()[-12:]!r}")
            out.setdefault(model, {})["working_provider"] = prov
            break

    json.dump(out, open("step1c_results.json", "w"), indent=2)
    print("\nwrote step1c_results.json")


if __name__ == "__main__":
    main()
