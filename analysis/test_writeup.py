"""The write-up's tables and figures must not drift from the run that produced them.

Not a recomputation of the metric definitions -- those are test_metrics.py's job. What
is checked here is that the derived artefacts a reader actually sees still agree with
results/analysis.json and with the raw generations underneath it.

Run: python analysis/test_writeup.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import metrics as M
from analysis.judge_swap import scores

REPO = Path(__file__).resolve().parent.parent
fails = []


def check(label, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {label}"
          + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        fails.append(label)


def close(label, got, want, tol=1e-9):
    check(label, abs(got - want) <= tol, True)


A = json.loads((REPO / "results/analysis.json").read_text())
T = json.loads((REPO / "results/writeup_tables.json").read_text())
S = json.loads((REPO / "results/judge_swap.json").read_text())

print("== the judge-swap table is anchored to the reported analysis ==")
# The failure worth catching is a swap table whose "after" column is not the number the
# write-up reports elsewhere -- a wrong judge directory produces exactly that, silently.
for model, v in S["per_model"].items():
    close(f"{model.split('/')[-1]} Opus side matches analysis.json",
          v["opus"], A["scores"][model][M.BIASING_FEATURES])

print("\n== the Haiku side is a real second scoring, not a copy ==")
haiku = scores(REPO / "results/judge_haiku_v2")
check("the two judges do not produce identical scores",
      haiku != {m: v["opus"] for m, v in S["per_model"].items()}, True)
for model, v in S["per_model"].items():
    close(f"{model.split('/')[-1]} Haiku side recomputes", v["haiku"], haiku[model])

print("\n== the reversal table uses run_analysis's rule ==")
expected = {k for k, v in A["rank_swaps"].items()
            if (v["biasing_features"] > 0.8 and v["filler_tokens"] < 0.2)
            or (v["biasing_features"] < 0.2 and v["filler_tokens"] > 0.8)}
check("same pairs as the headline analysis", set(T["reversals"]), expected)
check("and the rule still selects something", bool(expected), True)

print("\n== the roster cannot claim more verdicts than there were traces ==")
for model, d in T["roster"].items():
    check(f"{model.split('/')[-1]} judged <= flipped", d["judged"] <= d["bf_support"], True)
    check(f"{model.split('/')[-1]} flipped <= items", d["bf_support"] <= d["n_items"], True)

print("\n== every model ran the same frozen item set ==")
fps = {json.loads(p.read_text())["items_fingerprint"]
       for p in (REPO / "results/full").glob("metrics_*.json")}
check("one fingerprint across all eight runs", fps, {T["provenance"]["items_fingerprint"]})

print("\n== both themes of every figure are on disk ==")
missing = [f"{s}{d}.{e}"
           for s in ("fig1-reversals", "fig2-tau-ci", "fig3-judge-swap")
           for d in ("", "-dark") for e in ("svg", "png")
           if not (REPO / "writing/figures" / f"{s}{d}.{e}").exists()]
if missing:
    print("     (run `python analysis/figures.py` first -- writing/ is gitignored)")
check("no figure files missing", missing, [])

print("\nall passed" if not fails else f"\n{len(fails)} failed: {fails}")
sys.exit(1 if fails else 0)
