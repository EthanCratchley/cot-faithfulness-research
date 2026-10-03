"""The three figures for the write-up, drawn from results/ and nothing else.

Static SVG + PNG in both themes: the write-up is read on GitHub and on a light or dark
page, and a transparent background that inherits the host theme makes light-mode axis
text vanish on dark. Each theme gets an opaque surface it was contrast-checked against.

Run: python analysis/figures.py [--dark]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "writing/figures"

# Validated default palette (dataviz reference instance), light and dark steps.
THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  grid="#e3e2de", series1="#2a78d6", series2="#eb6834",
                  before="#9ec5f4", after="#256abf"),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 grid="#383835", series1="#3987e5", series2="#d95926",
                 before="#3987e5", after="#9ec5f4"),
}
name = lambda m: m.split("/")[-1].replace("-Instruct-2506", "").replace("-BF16", "")


def style(fig, axes, t):
    fig.patch.set_facecolor(t["surface"])
    for ax in axes if isinstance(axes, (list, tuple)) else [axes]:
        ax.set_facecolor(t["surface"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(t["grid"])
        ax.tick_params(colors=t["secondary"], labelsize=9, length=0)
        for key in ("center", "left", "right"):
            ax.set_title(ax.get_title(loc=key), loc=key, color=t["primary"])


def save(fig, stem, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if theme == "light" else "-dark"
    for ext in ("svg", "png", "pdf"):
        # PDFs go into the LaTeX paper, whose page is white; the tinted surface is for web.
        if ext == "pdf":
            if theme != "light":
                continue
            out, fc = REPO / "paper/figures", "#ffffff"
        else:
            out, fc = OUT, fig.get_facecolor()
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{stem}{suffix}.{ext}", format=ext, dpi=200,
                    bbox_inches="tight", facecolor=fc)
    plt.close(fig)
    print(f"  writing/figures/{stem}{suffix}.svg + .png")


def fig0_workflow(t, theme):
    """The pipeline: one hinted trace per item, three metrics applied to it, two comparisons."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, title, body, accent=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=t["surface"], ec=t["series1"] if accent else t["grid"],
                                    lw=1.8 if accent else 1.2, zorder=2))
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=9.5,
                color=t["primary"], fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h - 0.52, body, ha="center", va="top", fontsize=8.3,
                color=t["secondary"], zorder=3, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                     color=t["secondary"], lw=1.1, zorder=1,
                                     shrinkA=2, shrinkB=2))

    box(0.1, 1.7, 1.8, 1.2, "Inputs", "500 MMLU-Pro items\n8 open-weight models")
    box(2.4, 2.75, 2.0, 1.1, "Baseline", "question only\nmodel reasons freely")
    box(2.4, 0.75, 2.0, 1.4, "Hinted", "\u201cA Stanford professor\nthinks the answer is (X)\u201d\n+ question")
    box(5.0, 3.15, 2.5, 1.25, "Biasing Features",
        "hint changed the answer?\njudge: does the trace\nadmit the hint?", accent=True)
    box(5.0, 1.65, 2.5, 1.25, "Filler Tokens",
        "trace \u2192 200 dots\ndoes the answer change?", accent=True)
    box(5.0, 0.15, 2.5, 1.25, "Early Answering",
        "trace cut at 0\u201380%\nis the answer still changing?", accent=True)
    box(8.1, 1.05, 2.8, 2.5, "Compare metrics",
        "rank the 8 models per metric\n\u2192 agreement of rankings\n(Kendall\u2019s tau)\n\nlabel each trace per metric\n\u2192 agreement of labels\n(Cohen\u2019s kappa)")

    arrow(1.9, 2.5, 2.4, 3.3)
    arrow(1.9, 2.1, 2.4, 1.45)
    arrow(4.4, 3.3, 5.0, 3.75)
    for y in (3.75, 2.27, 0.77):
        arrow(4.4, 1.45, 5.0, y)
        arrow(7.5, y, 8.1, 2.3)
    ax.text(9.5, 0.6, "uncertainty: 2,000 bootstrap\nresamples of the 500 items", ha="center",
            va="center", fontsize=8, color=t["secondary"], style="italic")
    style(fig, ax, t)
    save(fig, "fig0-workflow", theme)


def fig1_reversals(a, t, theme):
    """One row per reversing pair: where each metric puts it. Rows, not a slopegraph --
    three of the four pairs sit within 6pp of each other under Biasing Features, and on
    shared axes their labels land on top of one another."""
    flips = [(k, v) for k, v in a["rank_swaps"].items()
             if (v["biasing_features"] > 0.8 and v["filler_tokens"] < 0.2)
             or (v["biasing_features"] < 0.2 and v["filler_tokens"] > 0.8)]
    flips.sort(key=lambda kv: kv[1]["biasing_features"])
    matched = "Olmo-3.1-32B-Instruct > allenai/Olmo-3.1-32B-Think"
    fig, ax = plt.subplots(figsize=(9.2, 4.4))

    for i, (k, v) in enumerate(flips):
        bf, fl = v["biasing_features"] * 100, v["filler_tokens"] * 100
        ax.plot([bf, fl], [i, i], color=t["grid"], lw=2.4, solid_capstyle="round", zorder=1)
        ax.plot(bf, i, "o", ms=10, color=t["series1"], mec=t["surface"], mew=2, zorder=3)
        ax.plot(fl, i, "o", ms=10, color=t["series2"], mec=t["surface"], mew=2, zorder=3)
        for x, val in ((bf, bf), (fl, fl)):
            ax.annotate(f"{val:.0f}%", (x, i), xytext=(0, 13), textcoords="offset points",
                        ha="center", fontsize=8.5, color=t["secondary"])

    labels = []
    for k, _ in flips:
        a_, b_ = (name(x) for x in k.split(" > "))
        labels.append(f"{a_}  >  {b_}")
    ax.set_yticks(range(len(flips)), labels)
    for i, lbl in enumerate(ax.get_yticklabels()):
        hi = matched in flips[i][0]
        lbl.set_color(t["primary"] if hi else t["secondary"])
        lbl.set_fontsize(9)
        lbl.set_fontweight("bold" if hi else "normal")

    ax.axvline(50, color=t["grid"], lw=1.2, ls=(0, (4, 4)), zorder=0)
    ax.annotate("50% — the metrics are\nindifferent between the two", (50, -0.62),
                xytext=(7, 0), textcoords="offset points", fontsize=8,
                color=t["secondary"], va="bottom")
    ax.set_xlim(-4, 104)
    ax.set_ylim(-0.85, len(flips) - 0.35)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("bootstrap resamples in which the first model ranks as more faithful",
                  color=t["secondary"], fontsize=9)
    ax.set_title("Four model pairs reverse outright between the two metrics",
                 fontsize=13, loc="left", pad=14)
    style(fig, ax, t)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, color=t["series1"],
                              label="under Biasing Features"),
                       Line2D([], [], marker="o", ls="", ms=9, color=t["series2"],
                              label="under Filler Tokens")],
              loc="upper center", frameon=False, fontsize=9, labelcolor=t["secondary"],
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, "fig1-reversals", theme)


def fig2_tau(a, t, theme):
    """Interval plot: tau-b per metric pair with its bootstrap CI and the H1 thresholds."""
    pairs = list(a["tau"].items())[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    for i, (k, v) in enumerate(pairs):
        ax.plot([v["ci_low"], v["ci_high"]], [i, i], color=t["series1"], lw=2.6,
                solid_capstyle="round", alpha=0.5, zorder=2)
        ax.plot(v["tau"], i, "o", ms=11, color=t["series1"], mec=t["surface"], mew=2,
                zorder=3)
        ax.annotate(f"{v['tau']:+.3f}", (0.965, i), ha="right", va="center",
                    fontsize=9.5, color=t["primary"], fontweight="bold")
    for x, lab in ((0.6, "H1 threshold\ntau 0.6"), (0.8, "CI must\nexclude 0.8")):
        ax.axvline(x, color=t["grid"], lw=1.2, ls=(0, (4, 4)), zorder=1)
        ax.annotate(lab, (x, -0.52), xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=t["secondary"], va="bottom")
    ax.set_yticks(range(len(pairs)),
                  [k.replace(" x ", " × ") for k, _ in pairs])
    for lbl in ax.get_yticklabels():
        lbl.set_color(t["primary"])
        lbl.set_fontsize(10)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.75, len(pairs) - 0.4)
    ax.set_xlabel("Kendall's tau-b (2,000-replicate bootstrap over items)",
                  color=t["secondary"], fontsize=9)
    ax.set_title("No metric pair ranks the eight models the same way",
                 fontsize=13, loc="left", pad=14)
    style(fig, ax, t)
    save(fig, "fig2-tau-ci", theme)


def fig3_judge(swap, t, theme):
    """Dumbbell: one metric, one criterion, two judge models."""
    per = sorted(swap["per_model"].items(), key=lambda x: x[1]["delta_pp"])
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, (m, v) in enumerate(per):
        ax.plot([v["haiku"] * 100, v["opus"] * 100], [i, i], color=t["grid"], lw=2.4,
                solid_capstyle="round", zorder=1)
        ax.plot(v["haiku"] * 100, i, "o", ms=10, color=t["before"], mec=t["surface"],
                mew=2, zorder=3)
        ax.plot(v["opus"] * 100, i, "o", ms=10, color=t["after"], mec=t["surface"],
                mew=2, zorder=3)
        ax.annotate(f"{v['delta_pp']:+.0f}pp", (max(v["haiku"], v["opus"]) * 100, i),
                    xytext=(13, 0), textcoords="offset points", va="center", fontsize=9,
                    color=t["secondary"])
    ax.set_yticks(range(len(per)), [name(m) for m, _ in per])
    for lbl in ax.get_yticklabels():
        lbl.set_color(t["primary"])
        lbl.set_fontsize(9)
    ax.set_xlim(-3, 100)
    ax.set_ylim(-0.7, len(per) - 0.3)
    ax.set_xticks([0, 20, 40, 60, 80], ["0%", "20%", "40%", "60%", "80%"])
    ax.set_xlabel("Biasing Features score — share of flipped traces that verbalize the hint",
                  color=t["secondary"], fontsize=9)
    ax.set_title("Same traces, same criterion — only the judge model changed",
                 fontsize=13, loc="left", pad=14)
    style(fig, ax, t)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, color=t["before"],
                              label="judged by Haiku 4.5"),
                       Line2D([], [], marker="o", ls="", ms=9, color=t["after"],
                              label="judged by Opus 5")],
              loc="lower right", frameon=False, fontsize=9, labelcolor=t["secondary"])
    save(fig, "fig3-judge-swap", theme)


def main():
    a = json.loads((REPO / "results/analysis.json").read_text())
    swap = json.loads((REPO / "results/judge_swap.json").read_text())
    for theme, t in THEMES.items():
        print(f"{theme}:")
        fig0_workflow(t, theme)
        fig1_reversals(a, t, theme)
        fig2_tau(a, t, theme)
        fig3_judge(swap, t, theme)


if __name__ == "__main__":
    main()
