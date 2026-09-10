"""Section 4.4: the same traces, the same criterion, two judge models.

Biasing Features is the only metric with a model in the loop. Swapping that model
re-scores every trace without touching a single generation, so whatever moves here is
attributable to the judge alone. Haiku verdicts are the superseded run kept in
results/judge_haiku_v2/; Opus verdicts are the ones the study reports.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import metrics as M
from analysis.stats import kendall_tau_b

REPO = Path(__file__).resolve().parent.parent
JUDGES = {"haiku": REPO / "results/judge_haiku_v2", "opus": REPO / "results/judge"}
MATCHED = ("allenai/Olmo-3.1-32B-Instruct", "allenai/Olmo-3.1-32B-Think")


def scores(judge_dir):
    per = M.load_all(REPO / "results/full", judge_dir)
    return {m: M.score(list(r.values()), M.BIASING_FEATURES) for m, r in per.items()}


def main():
    by_judge = {name: scores(d) for name, d in JUDGES.items()}
    models = sorted(by_judge["opus"])
    haiku, opus = by_judge["haiku"], by_judge["opus"]

    rank = lambda s: sorted(models, key=lambda m: s[m], reverse=True)
    out = {
        "judges": {"haiku": "claude-haiku-4-5-20251001", "opus": "claude-opus-5"},
        "note": "identical traces and prompt; only the judge model differs",
        "per_model": {m: {"haiku": haiku[m], "opus": opus[m],
                          "delta_pp": (opus[m] - haiku[m]) * 100} for m in models},
        "ranking_haiku": rank(haiku),
        "ranking_opus": rank(opus),
        "tau_between_judges": kendall_tau_b([haiku[m] for m in models],
                                            [opus[m] for m in models]),
        "matched_pair_gap_pp": {
            j: (s[MATCHED[0]] - s[MATCHED[1]]) * 100 for j, s in by_judge.items()},
    }
    (REPO / "results/judge_swap.json").write_text(json.dumps(out, indent=1))

    print(f"{'model':<34}{'Haiku':>9}{'Opus':>9}{'delta':>9}")
    for m in sorted(models, key=lambda m: opus[m] - haiku[m], reverse=True):
        d = out["per_model"][m]
        print(f"  {m.split('/')[-1][:30]:<32}{d['haiku']:>9.3f}{d['opus']:>9.3f}"
              f"{d['delta_pp']:>+8.1f}pp")
    print(f"\n  tau between the two judge rankings: {out['tau_between_judges']:+.3f}")
    g = out["matched_pair_gap_pp"]
    print(f"  matched pair (Instruct - Think): {g['haiku']:+.1f}pp under Haiku, "
          f"{g['opus']:+.1f}pp under Opus")
    print("\n  wrote results/judge_swap.json")


if __name__ == "__main__":
    main()
