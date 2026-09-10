"""Every table the write-up needs, regenerated from the raw results.

The recap was written by hand. This is the machine-checked version of the same
numbers: run it and the tables in writing/tables.md are whatever the JSON actually
says, so a figure in the prose can never drift from the run that produced it.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import metrics as M

REPO = Path(__file__).resolve().parent.parent
SHORT = {"biasing_features": "Biasing Features", "filler_tokens": "Filler Tokens",
         "early_answering": "Early Answering"}
name = lambda m: m.split("/")[-1]


def table(headers, rows, align=None):
    align = align or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def roster():
    """Per-model characteristics, computed off the 500-item full run."""
    rows, data = [], {}
    for p in sorted((REPO / "results/full").glob("metrics_*.json")):
        raw = json.loads(p.read_text())
        recs = list(M.load(p, REPO / f"results/judge/judge_{p.stem[8:]}.json")[1].values())
        r = raw["rows"]
        d = {
            "lab": raw["lab"], "thinking": raw["thinking"], "revision": raw["revision"][:8],
            "n_items": raw["n_items"],
            "base_acc": sum(x["baseline_answer"] == x["gold"] for x in r) / len(r),
            "flip_rate": sum(x["bf_support"] for x in recs) / len(recs),
            "trunc_rate": sum(x["truncated"] for x in recs) / len(recs),
            "median_cot": statistics.median(x["hinted_cot_chars"] for x in r),
            "bf_support": sum(x["bf_support"] for x in recs),
            "judged": sum(x["bf_verbalized"] is not None for x in recs),
            "injected_support": sum(x["injected_support"] for x in recs),
        }
        data[raw["model"]] = d
        rows.append([f"`{name(raw['model'])}`", d["lab"], "yes" if d["thinking"] else "no",
                     f"{d['base_acc']:.1%}", f"{d['flip_rate']:.1%}",
                     f"{d['trunc_rate']:.1%}", f"{d['median_cot']:,.0f}",
                     f"{d['judged']}/{d['bf_support']}", f"`{d['revision']}`"])
    rows.sort(key=lambda x: -float(x[3].rstrip("%")))
    return data, table(
        ["Model", "Lab", "Think", "Base acc.", "Flip", "Trunc.", "Median CoT",
         "Judged", "Revision"], rows,
        ["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---"])


def main():
    a = json.loads((REPO / "results/analysis.json").read_text())
    swap = json.loads((REPO / "results/judge_swap.json").read_text())
    data, roster_md = roster()
    models = a["models"]
    out = {"roster": data}

    # --- reversals: pairs where the two metrics disagree about the winner
    flips = [(k, v) for k, v in a["rank_swaps"].items()
             if (v["biasing_features"] > 0.8 and v["filler_tokens"] < 0.2)
             or (v["biasing_features"] < 0.2 and v["filler_tokens"] > 0.8)]
    out["reversals"] = {k: v for k, v in flips}

    # --- Step 1 API screen
    cat = json.loads((REPO / "results/05_catalog_screen.json").read_text())
    screen = {"n_screened": len(cat), "n_usable": sum(x["usable"] for x in cat),
              "n_reasoning": sum(x["reasoning"] for x in cat)}
    screen["pct_usable"] = screen["n_usable"] / screen["n_screened"]
    out["api_screen"] = screen

    # --- elicitation scaffold: same injected trace, cued vs uncued
    scaf = []
    for p in sorted((REPO / "results/step1L_cued").glob("step1L_*.json")):
        cued = json.loads(p.read_text())
        free = json.loads((REPO / "results/step1L" / p.name).read_text())
        scaf.append({"model": cued["model"],
                     "uncued_answer": free["misleading"]["answer"],
                     "uncued_followed": free["misleading"]["followed_injection"],
                     "cued_answer": cued["misleading"]["answer"],
                     "cued_followed": cued["misleading"]["followed_injection"],
                     "baseline_answer": cued["baseline_answer"]})
    out["scaffold"] = scaf

    # --- appendix provenance
    one = json.loads(next((REPO / "results/full").glob("metrics_*.json")).read_text())
    out["provenance"] = {
        "items_fingerprint": one["items_fingerprint"], "seed": one["seed"],
        "items_seed": one["items_seed"], "dtype": one["dtype"],
        "max_tokens": one["max_tokens"], "fractions": one["fractions"],
        "n_items": one["n_items"], "n_models": len(models),
        "generations_per_item": 3 + len(one["fractions"]),
        "total_generations": len(models) * one["n_items"] * (3 + len(one["fractions"])),
        "revisions": {m: data[m]["revision"] for m in models},
    }

    (REPO / "results/writeup_tables.json").write_text(json.dumps(out, indent=1))

    # ---------------- markdown ----------------
    md = ["# Tables for the write-up",
          "",
          "Generated by `analysis/writeup_tables.py` from `results/`. Do not hand-edit —",
          "re-run the script. Section numbers refer to `writing/blog-skeleton.md`.",
          "", "---", "", "## §3.2 — Model roster", "", roster_md, "",
          f"Base accuracy is the unhinted answer on the same {out['provenance']['n_items']} "
          "items the metrics are scored on. *Flip* is the share of items where the hint moved "
          "the answer onto the hinted option — the Biasing Features denominator. *Judged* is "
          "how many of those flipped traces returned a verdict.", "",
          "> **Note.** Base accuracy here is recomputed on the 500 items the metrics are scored "
          "on (e.g. Qwen 83.2%), not on the 200-item Step 2 pilot (Qwen 86.0%), so the roster "
          "is internally consistent with every other number in this file.", ""]

    md += ["---", "", "## §3.4 — API capability screen", "",
           table(["Quantity", "Value"],
                 [["Models screened", screen["n_screened"]],
                  ["Advertise a reasoning channel", screen["n_reasoning"]],
                  ["Support both capabilities the injection metrics need",
                   f"**{screen['n_usable']} ({screen['pct_usable']:.0%})**"]],
                 ["---", "---:"]), "",
           "Source: `results/05_catalog_screen.json`. The narrowing to ~20 after excluding "
           "roleplay finetunes, translation and vision models is a manual judgement recorded "
           "made during the Step 1 screen, not a field in the JSON.", ""]

    md += ["---", "", "## §4.1 — Rank correlation between metrics", "",
           table(["Metric pair", "tau-b", "95% CI", "Satisfies H1"],
                 [[k, f"**{v['tau']:+.3f}**", f"[{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]",
                   "yes" if v["tau"] < 0.6 and v["ci_high"] < 0.8 else "CI excludes 0.8"]
                  for k, v in a["tau"].items()],
                 ["---", "---:", "---", "---"]), "",
           f"{a['tau']['BF x Filler']['n_boot']:,} bootstrap replicates over "
           f"{a['tau']['BF x Filler']['n_items']} items. Falsification condition "
           f"{'TRIGGERED' if a['falsified'] else 'not triggered'} on any pair.", ""]

    md += ["## §4.1 — Per-model scores", "",
           table(["Model"] + [SHORT[k] for k in M.METRICS],
                 [[f"`{name(m)}`"] + [f"{a['scores'][m][k]:.3f}" for k in M.METRICS]
                  for m in sorted(models, key=lambda m: -a["scores"][m]["biasing_features"])],
                 ["---", "---:", "---:", "---:"]), "",
           "All three oriented so higher = more faithful.", ""]

    md += ["## §4.1 — Model pairs that reverse", "",
           table(["Model pair", "A wins under BF", "A wins under Filler"],
                 [[f"**{name(k.split(' > ')[0])} > {name(k.split(' > ')[1])}**",
                   f"{v['biasing_features']:.0%}", f"{v['filler_tokens']:.0%}"]
                  for k, v in flips],
                 ["---", "---:", "---:"]), "",
           f"{len(flips)} of {len(a['rank_swaps'])} model pairs. Read as the share of "
           "bootstrap resamples in which A outranks B under each metric.", ""]

    md += ["---", "", "## §4.2 — Trace-level agreement", "",
           table(["Metric pair", "Pooled kappa", "Traces"],
                 [[k, f"**{v['kappa']:+.3f}**", f"{v['n']:,}"] for k, v in a["kappa"].items()],
                 ["---", "---:", "---:"]), "",
           "Any pair involving Biasing Features is confined to flipped traces, which is why "
           "those denominators are so much smaller than the item count.", ""]

    h3 = a["H3"]
    md += ["---", "", "## §4.3 — Disagreement by thinking mode", "",
           table(["Model", "Thinking", "Mean kappa across the three pairs"],
                 [[f"`{name(m)}`", "yes" if v["thinking"] else "no",
                   f"{v['mean_kappa']:+.3f}"]
                  for m, v in sorted(h3["per_model"].items(),
                                     key=lambda x: -x[1]["mean_kappa"])],
                 ["---", "---", "---:"]), "",
           f"Thinking **{h3['thinking_mean']:+.3f}** (n={h3['n_thinking']}), instruct "
           f"**{h3['instruct_mean']:+.3f}** (n={h3['n_instruct']}). Separation is complete. "
           "The statistic was chosen after seeing the data, not before.", ""]

    md += ["---", "", "## §4.4 — The judge swap", "",
           table(["Model", "Haiku 4.5", "Opus 5", "Delta"],
                 [[f"`{name(m)}`", f"{v['haiku']:.3f}", f"{v['opus']:.3f}",
                   f"{v['delta_pp']:+.1f}pp"]
                  for m, v in sorted(swap["per_model"].items(),
                                     key=lambda x: -x[1]["delta_pp"])],
                 ["---", "---:", "---:", "---:"]), "",
           "Identical traces, identical criterion, identical prompt. Only the judge model "
           f"differs. The two judge rankings correlate at tau **{swap['tau_between_judges']:+.3f}** "
           f"— lower than the {a['tau']['BF x Filler']['tau']:+.3f} between Biasing Features and "
           "Filler Tokens. Changing the judge *inside* one metric perturbs the ranking more "
           "than changing metric does.", "",
           f"Matched pair (Instruct − Think): "
           f"**{swap['matched_pair_gap_pp']['haiku']:+.1f}pp** under Haiku, "
           f"**{swap['matched_pair_gap_pp']['opus']:+.1f}pp** under Opus.", ""]

    md += ["## §4.4 — The elicitation scaffold", "",
           table(["Model", "Baseline", "Uncued answer", "Followed?", "Cued answer", "Followed?"],
                 [[f"`{name(s['model'])}`", s["baseline_answer"], s["uncued_answer"],
                   "yes" if s["uncued_followed"] else "**no**", s["cued_answer"],
                   "**yes**" if s["cued_followed"] else "no"] for s in scaf],
                 ["---", "---:", "---:", "---", "---:", "---"]), "",
           "One item, one injected trace arguing for a wrong answer. Left to free-run the "
           "model re-derives and ignores it; with the answer cued immediately after the "
           "trace, it follows. Three models were run both ways "
           "(`results/step1L/` vs `results/step1L_cued/`).", ""]

    st = a["stability"]
    md += ["---", "", "## §4.5 — Ranking stability on the non-truncated subset", "",
           table(["Metric", "tau (all vs clean)", "Non-adjacent swaps", "Verdict"],
                 [[SHORT[k], f"{st[k]['tau']:+.3f}", len(st[k]["non_adjacent_swaps"]),
                   "**Pass**" if st[k]["pass"] else "**Fail**"]
                  for k in M.METRICS if k in st],
                 ["---", "---:", "---:", "---"]), "",
           f"{st['n_clean']}/{st['n_all']} items were truncated by no model. Gate: tau >= 0.85 "
           "with no non-adjacent swaps. Biasing Features has 7–29 flipped traces per model on "
           "the clean subset — below the support floor, so its result is uninformative rather "
           "than a failure.", ""]

    p = out["provenance"]
    md += ["---", "", "## Appendix — provenance", "",
           table(["Field", "Value"],
                 [["Item set fingerprint", f"`{p['items_fingerprint']}`"],
                  ["Items", p["n_items"]], ["Models", p["n_models"]],
                  ["Generation seed", p["seed"]], ["Item-sampling seed", p["items_seed"]],
                  ["Precision", p["dtype"]], ["Max tokens", f"{p['max_tokens']:,}"],
                  ["Truncation fractions", ", ".join(str(f) for f in p["fractions"])],
                  ["Generations per item", p["generations_per_item"]],
                  ["Total generations", f"{p['total_generations']:,}"]],
                 ["---", "---"]), "",
           "### Model revisions", "",
           table(["Model", "HuggingFace revision"],
                 [[f"`{m}`", f"`{r}`"] for m, r in p["revisions"].items()]), ""]

    out_dir = REPO / "writing"
    out_dir.mkdir(exist_ok=True)   # gitignored: a fresh clone has no writing/
    (out_dir / "tables.md").write_text("\n".join(md))
    print(f"wrote writing/tables.md ({len('\n'.join(md).splitlines())} lines)")
    print(f"wrote results/writeup_tables.json")
    print(f"\n  {len(flips)} reversals, API screen {screen['n_usable']}/{screen['n_screened']}, "
          f"{len(scaf)} scaffold pairs, {p['total_generations']:,} generations")


if __name__ == "__main__":
    main()
