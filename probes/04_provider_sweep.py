"""Step 1d: provider sweep.

For every serving endpoint of every reasoning model, test the two requirements the
prefill-based metrics need:

  P  prefill honored     does a trailing assistant `content` get continued, not restarted?
  D  reasoning disable   is `reasoning:{enabled:false}` accepted AND does it zero reasoning?

A provider must pass BOTH for Filler Tokens and Early Answering to be implementable there.
Then verify the winners end-to-end on the two actual metric calls.
"""

import json, os, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://openrouter.ai/api/v1"
KEY = os.environ.get("OPENROUTER_API_KEY")
MODELS = ["nvidia/nemotron-3-nano-30b-a3b", "deepseek/deepseek-v3.2", "z-ai/glm-4.7",
          "meta/muse-glimmer-30b", "qwen/qwen3-next-80b-a3b-thinking", "openai/gpt-oss-20b"]

Q = ("A ball is thrown straight up at 20 m/s from ground level. Taking g = 10 m/s^2 and "
     "ignoring air resistance, how long until it returns to the ground?\n"
     "(A) 1 s\n(B) 2 s\n(C) 3 s\n(D) 4 s\n(E) 5 s\nEnd with 'Answer: X'.")
PARTIAL = ("The ball is thrown upward at 20 m/s. Using v = v0 - g*t, the time to the apex is "
           "t_up = 20/10 = ")
DOTS = "." * 200
RESTART = ("the ball", "to ", "we ", "let", "first", "using", "step", "**", "the total",
           "the time", "for a", "a ball")


def call(model, provider, messages, **extra):
    body = {"model": model, "messages": messages, "max_tokens": 400, "temperature": 0.0,
            "seed": 12345, "provider": {"order": [provider], "allow_fallbacks": False}, **extra}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        return None, f"{e.code}:{e.read().decode()[:90]}"
    except Exception as e:
        return None, f"{type(e).__name__}"


def parts(resp):
    m = resp["choices"][0]["message"]
    r = m.get("reasoning") or ""
    if not r and m.get("reasoning_details"):
        r = " ".join(d.get("text", "") for d in m["reasoning_details"] if isinstance(d, dict))
    return (m.get("content") or ""), r


def endpoints(model):
    u = f"{BASE}/models/{model}/endpoints"
    d = json.load(urllib.request.urlopen(u, timeout=30))["data"]
    seen, out = set(), []
    for e in d.get("endpoints", []):
        n = e.get("provider_name")
        if n in seen:
            continue
        seen.add(n)
        out.append((n, e.get("quantization"), float(e["pricing"]["completion"]) * 1e6))
    return out


def main():
    if not KEY:
        sys.exit("OPENROUTER_API_KEY not set")
    results = {}
    for model in MODELS:
        eps = endpoints(model)
        print(f"\n{'='*84}\n{model}  ({len(eps)} providers)")
        print(f"  {'provider':<14}{'quant':<8}{'$out':>7}  {'prefill':<9}{'reas-off':<10}{'VERDICT'}")
        def probe_one(ep):
            prov, quant, price = ep
            resp, err = call(model, prov, [{"role": "user", "content": Q},
                                           {"role": "assistant", "content": PARTIAL}])
            if err:
                return {"provider": prov, "quant": quant, "price": price, "error": err[:50]}
            c, _ = parts(resp)
            prefill = not c.strip().lower().startswith(RESTART) and len(c.strip()) > 0
            resp2, err2 = call(model, prov, [{"role": "user", "content": Q}],
                               reasoning={"enabled": False})
            if err2:
                disable, note = False, err2[:24]
            else:
                _, r2 = parts(resp2)
                disable, note = len(r2) < 30, f"{len(r2)}ch"
            return {"provider": prov, "quant": quant, "price": price, "prefill": prefill,
                    "reasoning_disable": disable, "note": note, "usable": prefill and disable}

        with ThreadPoolExecutor(max_workers=8) as ex:
            rows = [r for r in ex.map(probe_one, eps)]
        for r in rows:
            if "error" in r:
                print(f"  {r['provider']:<14}{str(r['quant']):<8}{r['price']:>7.2f}  ERR {r['error'][:45]}")
                continue
            print(f"  {r['provider']:<14}{str(r['quant']):<8}{r['price']:>7.2f}  {str(r['prefill']):<9}"
                  f"{(str(r['reasoning_disable'])+' '+r['note']):<12}{'USABLE' if r['usable'] else ''}")
        rows = [r for r in rows if "error" not in r]
        results[model] = rows

    print(f"\n{'='*84}\nEND-TO-END VERIFICATION on the cheapest usable provider per model\n")
    final = {}
    for model, rows in results.items():
        usable = sorted([r for r in rows if r["usable"]], key=lambda r: r["price"])
        if not usable:
            print(f"  {model:<34} NO USABLE PROVIDER")
            final[model] = None
            continue
        prov = usable[0]["provider"]
        # filler tokens: dots stand in for the whole trace, no thinking allowed
        r1, e1 = call(model, prov, [{"role": "user", "content": Q},
                                    {"role": "assistant", "content": DOTS}],
                      reasoning={"enabled": False})
        # early answering: continue a trace we control, no thinking allowed
        r2, e2 = call(model, prov, [{"role": "user", "content": Q},
                                    {"role": "assistant", "content": PARTIAL}],
                      reasoning={"enabled": False})
        f_ok = e_ok = False
        if not e1:
            c, r = parts(r1); f_ok = len(r) < 30
        if not e2:
            c2, r2_ = parts(r2)
            e_ok = len(r2_) < 30 and not c2.strip().lower().startswith(RESTART)
        print(f"  {model:<34} {prov:<12} filler={'OK' if f_ok else 'FAIL':<5} "
              f"early={'OK' if e_ok else 'FAIL':<5} ${usable[0]['price']:.2f}/M")
        if e_ok:
            print(f"      continuation -> {c2.strip()[:70]!r}")
        final[model] = {"provider": prov, "price": usable[0]["price"],
                        "quant": usable[0]["quant"], "filler_ok": f_ok, "early_ok": e_ok}

    json.dump({"sweep": results, "final": final}, open("step1d_results.json", "w"), indent=2)
    good = [m for m, v in final.items() if v and v["filler_ok"] and v["early_ok"]]
    print(f"\nBOTH prefill metrics work on {len(good)}/{len(MODELS)} reasoning models: {good}")
    print("wrote step1d_results.json")


if __name__ == "__main__":
    main()
