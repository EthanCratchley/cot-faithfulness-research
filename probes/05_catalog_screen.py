"""Step 1e: capability screen across the OpenRouter catalog.

Selection criterion becomes: can the metrics be computed at all?

For each candidate model, walk its providers until one passes BOTH:
  P  prefill honored    a trailing assistant `content` is continued, not restarted
  D  reasoning off      `reasoning:{enabled:false}` zeroes the trace
                        (N/A -> auto-pass for instruct models: their CoT is in `content`,
                         so a content prefill already IS trace injection)

Writes screen_results.json and prints a capable-model table grouped by lab.
"""

import json, os, sys, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://openrouter.ai/api/v1"
KEY = os.environ.get("OPENROUTER_API_KEY")
MAX_PROVIDERS = 4
MAX_OUT_PRICE = 3.0  # $/M -- keeps the study affordable

Q = ("A ball is thrown straight up at 20 m/s from ground level. Taking g = 10 m/s^2, "
     "how long until it returns to the ground?\n(A) 1 s\n(B) 2 s\n(C) 3 s\n(D) 4 s\n(E) 5 s\n"
     "End with 'Answer: X'.")
PARTIAL = "The ball is thrown upward at 20 m/s. Using v = v0 - g*t, the time to the apex is t_up = 20/10 = "
RESTART = ("the ball", "to ", "we ", "let", "first", "using", "step", "**", "the total",
           "the time", "for a", "a ball", "okay", "sure", "i ", "this ", "given")

SKIP = ("vision", "-vl", "coder", "guard", "embed", "rerank", "audio", "image", "tts",
        "whisper", "safeguard", ":free", ":batch", "~", "-rp-", "roleplay", "uncensored")


def http(url, body=None, timeout=90):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        return None, f"{e.code}"
    except Exception as e:
        return None, type(e).__name__


def call(model, provider, messages, **extra):
    resp, err = http(f"{BASE}/chat/completions", {
        "model": model, "messages": messages, "max_tokens": 220, "temperature": 0.0,
        "seed": 12345, "provider": {"order": [provider], "allow_fallbacks": False}, **extra})
    if err:
        return None, err
    if not isinstance(resp, dict) or not resp.get("choices"):
        return None, str(resp.get("error", "no choices"))[:60] if isinstance(resp, dict) else "bad body"
    return resp, None


def parts(resp):
    m = resp["choices"][0]["message"]
    r = m.get("reasoning") or ""
    if not r and m.get("reasoning_details"):
        r = " ".join(d.get("text", "") for d in m["reasoning_details"] if isinstance(d, dict))
    return (m.get("content") or ""), r


def candidates():
    cat = json.load(urllib.request.urlopen(f"{BASE}/models", timeout=40))["data"]
    out, per_lab = [], {}
    for m in sorted(cat, key=lambda x: float(x["pricing"]["completion"])):
        mid = m["id"]
        if any(s in mid.lower() for s in SKIP):
            continue
        if (m.get("architecture", {}).get("output_modalities") or ["text"]) != ["text"]:
            continue
        price = float(m["pricing"]["completion"]) * 1e6
        if not (0 < price <= MAX_OUT_PRICE):
            continue
        lab = mid.split("/")[0]
        if per_lab.get(lab, 0) >= 4:      # cap per lab, keep the screen diverse
            continue
        per_lab[lab] = per_lab.get(lab, 0) + 1
        out.append({"id": mid, "lab": lab, "price": price,
                    "reasoning": "include_reasoning" in (m.get("supported_parameters") or [])})
    return out


def screen(c):
    try:
        return _screen(c)
    except Exception as e:
        return {**c, "usable": False, "why": f"exception {type(e).__name__}"}


def _screen(c):
    mid = c["id"]
    d, err = http(f"{BASE}/models/{mid}/endpoints", timeout=40)
    if err:
        return {**c, "usable": False, "why": f"endpoints {err}"}
    eps, seen = [], set()
    for e in (d["data"].get("endpoints") or []):
        n = e.get("provider_name")
        if n and n not in seen:
            seen.add(n)
            eps.append((n, e.get("quantization"), float(e["pricing"]["completion"]) * 1e6))
    if not eps:
        return {**c, "usable": False, "why": "no endpoints"}

    for prov, quant, price in eps[:MAX_PROVIDERS]:
        r1, e1 = call(mid, prov, [{"role": "user", "content": Q},
                                  {"role": "assistant", "content": PARTIAL}])
        if e1:
            continue
        content, _ = parts(r1)
        s = content.strip()
        if not s or s.lower().startswith(RESTART):
            continue                                    # prefill dropped
        if not c["reasoning"]:
            return {**c, "usable": True, "provider": prov, "quant": quant,
                    "price": price, "mode": "instruct (no reasoning channel)"}
        r2, e2 = call(mid, prov, [{"role": "user", "content": Q}],
                      reasoning={"enabled": False})
        if e2:
            continue
        _, reasoning = parts(r2)
        if len(reasoning) < 30:
            return {**c, "usable": True, "provider": prov, "quant": quant,
                    "price": price, "mode": "reasoning (disable OK)"}
    return {**c, "usable": False, "why": "no provider passed both"}


def main():
    if not KEY:
        sys.exit("OPENROUTER_API_KEY not set")
    cands = candidates()
    print(f"screening {len(cands)} models across {len(set(c['lab'] for c in cands))} labs\n", flush=True)
    with ThreadPoolExecutor(max_workers=14) as ex:
        res = list(ex.map(screen, cands))
    json.dump(res, open("screen_results.json", "w"), indent=2)

    ok = sorted([r for r in res if r["usable"]], key=lambda r: (r["lab"], r["price"]))
    print(f"{'='*92}\nCAPABLE MODELS: {len(ok)} of {len(cands)}\n")
    print(f"  {'model':<44}{'provider':<13}{'quant':<8}{'$out':>6}  mode")
    lab = None
    for r in ok:
        if r["lab"] != lab:
            lab = r["lab"]; print(f"  --- {lab}")
        print(f"  {r['id']:<44}{r['provider']:<13}{str(r['quant']):<8}{r['price']:>6.2f}  {r['mode']}")
    print(f"\n  labs represented: {sorted(set(r['lab'] for r in ok))}")
    reasoning_ok = [r for r in ok if r["reasoning"]]
    print(f"  reasoning models: {len(reasoning_ok)} | instruct models: {len(ok)-len(reasoning_ok)}")
    print(f"\n  VERDICT: {'ENOUGH -- rebuild the model list from these' if len(ok) >= 8 else 'NOT ENOUGH -- fall back to simulatability'}")
    print("\nwrote screen_results.json")


if __name__ == "__main__":
    main()
