"""Step 7: the headline comparison. Reads traces + verdicts, writes the analysis.

Nothing here decides anything the pre-registration did not fix in advance: the metric
pairs, the thresholds, the binarization, and the bootstrap unit were all written down
before the first full run.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import metrics as M
from analysis.stats import (bootstrap_tau, cohens_kappa, kendall_tau_b,
                            rank_swap_frequency, ranking_stability)

REPO = Path(__file__).resolve().parent.parent
PAIRS = [(M.BIASING_FEATURES, M.FILLER_TOKENS),
         (M.BIASING_FEATURES, M.EARLY_ANSWERING),
         (M.FILLER_TOKENS, M.EARLY_ANSWERING)]
SHORT = {M.BIASING_FEATURES: "BF", M.FILLER_TOKENS: "Filler", M.EARLY_ANSWERING: "Early"}


def load_all():
    per = {}
    for p in sorted((REPO / "results/full").glob("metrics_*.json")):
        slug = p.stem[len("metrics_"):]
        j = REPO / f"results/judge/judge_{slug}.json"
        model, recs = M.load(p, j if j.exists() else None)
        per[model] = recs
    return per


def main():
    per = load_all()
    models = sorted(per)
    out = {"n_models": len(models), "models": models}
    print(f"{len(models)} models\n")

    print(f"{'model':<32}{'BF':>9}{'Filler':>9}{'Early':>9}{'BFsupp':>8}{'injsupp':>9}")
    scores = {}
    for m in models:
        r = list(per[m].values())
        s = {k: M.score(r, k) for k in M.METRICS}
        scores[m] = s
        print(f"{m.split('/')[-1][:30]:<32}"
              f"{s[M.BIASING_FEATURES]:>9.3f}{s[M.FILLER_TOKENS]:>9.3f}"
              f"{s[M.EARLY_ANSWERING]:>9.3f}"
              f"{sum(1 for x in r if x['bf_verbalized'] is not None):>8}"
              f"{sum(x['injected_support'] for x in r):>9}")
    out["scores"] = {m: scores[m] for m in models}

    print("\n=== rankings (most faithful first) ===")
    out["rankings"] = {}
    for k in M.METRICS:
        rk = M.ranking(per, k)
        out["rankings"][k] = rk
        print(f"  {SHORT[k]:<7} " + " > ".join(x.split('/')[-1][:13] for x in rk))

    print("\n=== H1: Kendall tau-b with 2000-replicate item bootstrap ===")
    out["tau"] = {}
    h1_hits = []
    for a, b in PAIRS:
        r = bootstrap_tau(per, M.score, a, b, n_boot=2000)
        key = f"{SHORT[a]} x {SHORT[b]}"
        out["tau"][key] = r
        passes = r["tau"] < 0.6 and r["ci_high"] < 0.8
        h1_hits.append(passes)
        print(f"  {key:<16} tau={r['tau']:+.3f}  95% CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
              f"   {'<-- satisfies H1' if passes else ''}")
    out["H1_pass"] = any(h1_hits)
    print(f"\n  H1 (>=1 pair with tau<0.6 and CI excluding 0.8): "
          f"{'CONFIRMED' if any(h1_hits) else 'NOT CONFIRMED'}")
    falsified = all(r["tau"] > 0.8 and r["ci_low"] > 0.6 for r in out["tau"].values())
    out["falsified"] = falsified
    print(f"  Falsification (all pairs tau>0.8, CI excluding 0.6): "
          f"{'TRIGGERED' if falsified else 'not triggered'}")

    print("\n=== H2: pooled trace-level Cohen's kappa ===")
    out["kappa"] = {}
    for a, b in PAIRS:
        recs = [r for m in models for r in per[m].values()]
        sup = M.common_support(recs, a, b)
        k = cohens_kappa([M.binarize(r, a) for r in sup], [M.binarize(r, b) for r in sup])
        out["kappa"][f"{SHORT[a]} x {SHORT[b]}"] = {"kappa": k, "n": len(sup)}
        print(f"  {SHORT[a]:>6} x {SHORT[b]:<7} kappa={k:+.3f}  on {len(sup):,} traces")
    ks = [v["kappa"] for v in out["kappa"].values()]
    out["H2_pass"] = all(k < 0.4 for k in ks)
    print(f"\n  H2 (pooled kappa < 0.4): {'CONFIRMED' if out['H2_pass'] else 'NOT CONFIRMED'}")

    print("\n=== rank-swap frequency (BF x Filler) ===")
    sw = rank_swap_frequency(per, M.score, M.BIASING_FEATURES, M.FILLER_TOKENS, n_boot=2000)
    out["rank_swaps"] = sw
    flips = {k: v for k, v in sw.items()
             if (v[M.BIASING_FEATURES] > 0.8 and v[M.FILLER_TOKENS] < 0.2)
             or (v[M.BIASING_FEATURES] < 0.2 and v[M.FILLER_TOKENS] > 0.8)}
    print(f"  {len(flips)} of {len(sw)} model pairs reverse under the two metrics:")
    for k, v in list(flips.items())[:6]:
        a, b = k.split(" > ")
        print(f"    {a.split('/')[-1][:20]:<22} > {b.split('/')[-1][:20]:<22} "
              f"BF={v[M.BIASING_FEATURES]:.0%}  Filler={v[M.FILLER_TOKENS]:.0%}")

    print("\n=== Amendment 1: ranking stability on the non-truncated subset ===")
    all_items = sorted(set.intersection(*[set(per[m]) for m in models]))
    clean = [i for i in all_items if not any(per[m][i]["truncated"] for m in models)]
    print(f"  {len(clean)}/{len(all_items)} items truncated by no model "
          f"({len(clean)/len(all_items):.1%})")
    out["stability"] = {"n_all": len(all_items), "n_clean": len(clean)}
    if len(clean) >= 30:
        for k in M.METRICS:
            sa = {m: M.score(list(per[m].values()), k) for m in models}
            sc = {m: M.score([per[m][i] for i in clean], k) for m in models}
            st = ranking_stability(sa, sc)
            out["stability"][k] = st
            print(f"  {SHORT[k]:<7} tau={st['tau']:+.3f}  "
                  f"{'PASS' if st['pass'] else 'FAIL'}  "
                  f"non-adjacent swaps: {len(st['non_adjacent_swaps'])}")
    else:
        print("  subset too small to rank on -- reported as a limitation")

    print("\n=== H3: disagreement by thinking mode (SECONDARY; statistic set post-hoc) ===")
    sys.path.insert(0, str(REPO / "harness"))
    from prompts import BY_REPO
    h3 = {}
    for m in models:
        recs = list(per[m].values())
        ks = []
        for a, b in PAIRS:
            sup = M.common_support(recs, a, b)
            if len(sup) >= 10:
                ks.append(cohens_kappa([M.binarize(r, a) for r in sup],
                                       [M.binarize(r, b) for r in sup]))
        h3[m] = {"thinking": BY_REPO[m].thinking,
                 "mean_kappa": sum(ks) / len(ks) if ks else float("nan")}
    th = [v["mean_kappa"] for v in h3.values() if v["thinking"]]
    ins = [v["mean_kappa"] for v in h3.values() if not v["thinking"]]
    out["H3"] = {"per_model": h3, "thinking_mean": sum(th) / len(th),
                 "instruct_mean": sum(ins) / len(ins),
                 "n_thinking": len(th), "n_instruct": len(ins),
                 "direction_as_predicted": sum(th) / len(th) < sum(ins) / len(ins),
                 "separation_complete": max(th) < min(ins)}
    print(f"  thinking (n={len(th)}) mean kappa {sum(th)/len(th):+.3f}   "
          f"instruct (n={len(ins)}) {sum(ins)/len(ins):+.3f}")
    print(f"  direction as predicted: {out['H3']['direction_as_predicted']}   "
          f"complete separation: {out['H3']['separation_complete']}")
    pair = ("allenai/Olmo-3.1-32B-Think", "allenai/Olmo-3.1-32B-Instruct")
    if all(x in h3 for x in pair):
        out["H3"]["matched_pair"] = {x: h3[x]["mean_kappa"] for x in pair}
        print(f"  matched pair: Think {h3[pair[0]]['mean_kappa']:+.3f} vs "
              f"Instruct {h3[pair[1]]['mean_kappa']:+.3f}")

    (REPO / "results/analysis.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote results/analysis.json")


if __name__ == "__main__":
    main()
