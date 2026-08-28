"""Offline checks for Early Answering and hint injection.

Truncation is exercised against REAL Step 2 traces, not synthetic strings: the
failure modes that matter (empty reasoning, non-monotone cuts, tokenizer round-trip
loss) only show up on real model output.
"""
import glob, json, sys
from early_answering import FRACTIONS, KAPPA_FRACTION, aoc, score_item, truncate
from hints import apply, pick_target, verbalization_support
from items import load

fails = []


def brief(v, n=70):
    r = repr(v)
    return r if len(r) <= n else r[:n] + f"...({len(r)} chars)"


def check(label, got, want):
    ok = got == want or (isinstance(want, float) and abs(got - want) < 1e-9)
    print(f"{'ok  ' if ok else 'FAIL'} {label}: got {brief(got)}, want {brief(want)}")
    if not ok:
        fails.append(label)


print("== AOC ==")
# Answer fixed from the very start: the CoT did nothing, maximally unfaithful.
check("answer decided at 0% -> AOC 0", aoc({f: 1.0 for f in FRACTIONS}), 0.0)
# Answer only appears with the complete trace: maximally load-bearing.
check("answer only at 100% -> AOC 0.9", aoc({f: 0.0 for f in FRACTIONS}), 0.9)
# Half the items match at every truncation.
check("flat 0.5 curve -> AOC 0.45", aoc({f: 0.5 for f in FRACTIONS}), 0.45)
mid = aoc({0.0: 0.0, 0.2: 0.0, 0.4: 0.0, 0.6: 1.0, 0.8: 1.0})
check("answer settles at 60%", round(mid, 4), 0.5)
print(f"     ordering holds: early={aoc({f: 1.0 for f in FRACTIONS}):.2f} < "
      f"mid={mid:.2f} < late={aoc({f: 0.0 for f in FRACTIONS}):.2f}")

print("\n== per-item scoring ==")
s = score_item({0.0: "A", 0.2: "A", 0.4: "D", 0.6: "D", 0.8: "D"}, "D")
check("matches counted against the full-CoT answer", s["curve"],
      {0.0: 0.0, 0.2: 0.0, 0.4: 1.0, 0.6: 1.0, 0.8: 1.0})
check("kappa flag false when answer already settled at 0.6",
      s["faithful_at_kappa_fraction"], False)
s2 = score_item({0.0: "A", 0.2: "B", 0.4: "C", 0.6: "A", 0.8: "D"}, "D")
check("kappa flag true when 0.6 still disagrees",
      s2["faithful_at_kappa_fraction"], True)
# The inflation trap: unparsed must count as a mismatch, not vanish.
s3 = score_item({0.0: None, 0.2: None, 0.4: "D", 0.6: "D", 0.8: "D"}, "D")
check("unparsed counts as no-match", s3["curve"][0.0], 0.0)

print("\n== hint injection ==")
meta = load("../data/items_pilot_200.json")
items = meta["items"]
bad = [i for i in items if pick_target(i) == i["answer"]]
check("hint never points at the gold answer", len(bad), 0)
check("target is deterministic", pick_target(items[0]), pick_target(items[0]))
targets = {i["question_id"]: pick_target(i) for i in items}
check("all targets are valid options",
      sum(targets[i["question_id"]] not in
          [chr(65 + k) for k in range(len(i["options"]))] for i in items), 0)
body, t = apply(items[0], "Q?\n(A) x\n(B) y")
check("hint text carries the target letter", f"({t})" in body, True)
check("question survives the wrapper", "Q?" in body, True)
spread = len({pick_target(i) for i in items})
print(f"     targets span {spread} distinct letters across {len(items)} items")

print("\n== common support ==")
check("flip to the hinted option is in support",
      verbalization_support("D", "B", "B"), True)
check("ignoring the hint is out of support",
      verbalization_support("D", "D", "B"), False)
check("flip to a third option is out of support",
      verbalization_support("D", "C", "B"), False)
check("already answering the target is out of support",
      verbalization_support("B", "B", "B"), False)
check("unparsed baseline is out of support",
      verbalization_support(None, "B", "B"), False)

print("\n== truncation on real Step 2 traces ==")
try:
    from transformers import AutoTokenizer
    from prompts import BY_REPO
    f = "../results/step2/step2_allenai_Olmo-3.1-32B-Think.json"
    d = json.load(open(f))
    tok = AutoTokenizer.from_pretrained(d["model"])
    cfg = BY_REPO[d["model"]]
    from reasoning import reasoning_of
    traces = [reasoning_of(r["raw"], cfg) for r in d["rows"][:40]]
    traces = [t for t in traces if len(t) > 200]
    lens, nonmono = [], 0
    for t in traces:
        cut = [len(truncate(tok, t, fr)) for fr in FRACTIONS]
        lens.append(cut)
        nonmono += any(cut[i] > cut[i + 1] for i in range(len(cut) - 1))
    check("truncation lengths are monotone in fraction", nonmono, 0)
    check("fraction 0 gives empty", truncate(tok, traces[0], 0.0), "")
    check("fraction 1 returns the trace unchanged",
          truncate(tok, traces[0], 1.0), traces[0])
    mid = truncate(tok, traces[0], 0.5)
    check("a 50% cut is a prefix of the trace", traces[0].startswith(mid[:200]), True)
    print(f"     {len(traces)} real traces, median full length "
          f"{sorted(len(t) for t in traces)[len(traces)//2]} chars")
except ImportError:
    print("     (transformers not installed; skipping)")

print("\nall passed" if not fails else f"\n{len(fails)} failed: {fails}")
sys.exit(1 if fails else 0)
