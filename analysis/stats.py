"""Statistics for the headline comparison. No dependency on any model or run.

A bug here is nearly invisible: a wrong tau still returns a plausible number in
[-1, +1] and no smoke test catches it. Every function is therefore checked against
hand-computable cases in test_stats.py.
"""
from itertools import combinations
from random import Random


def kendall_tau_b(x, y):
    """Tau-b: the tie-corrected variant.

    Tau-b, not tau-a, because metric scores tie in practice -- two models can post the
    same verbalization rate. Tau-a treats a tie as neither agreement nor disagreement
    but still divides by the untied total, so it cannot reach 1.0 on data containing
    ties and would understate agreement for reasons that have nothing to do with the
    metrics.
    """
    if len(x) != len(y):
        raise ValueError("x and y must be the same length")
    con = dis = tx = ty = 0
    for i, j in combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0:
            continue          # tied in both: informative about neither
        if dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx * dy > 0:
            con += 1
        else:
            dis += 1
    denom = ((con + dis + tx) * (con + dis + ty)) ** 0.5
    return (con - dis) / denom if denom else float("nan")


def cohens_kappa(a, b):
    """Agreement between two binary labellings, corrected for chance.

    Returns nan when both raters use a single label throughout: every cell lands on
    the diagonal, chance agreement is 1.0, and kappa is 0/0. That is genuinely
    undefined rather than perfect agreement, and must not be reported as 1.0.
    """
    if len(a) != len(b):
        raise ValueError("a and b must be the same length")
    n = len(a)
    if not n:
        return float("nan")
    obs = sum(i == j for i, j in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    exp = pa * pb + (1 - pa) * (1 - pb)
    return (obs - exp) / (1 - exp) if exp != 1 else float("nan")


def bootstrap_tau(per_item, score_fn, metric_a, metric_b, n_boot=2000, seed=12345):
    """Percentile CI for tau, resampling ITEMS.

    Items, not models. With eight models the null SD of tau is about 0.29, so a point
    estimate cannot separate "the metrics disagree" from sampling noise; resampling
    models would also destroy the fixed-model design. Items are the sampling unit the
    study actually drew.

    per_item : {model: {item_id: record}}
    score_fn : (records, metric) -> score for one model under one metric
    """
    models = sorted(per_item)
    items = sorted(set.intersection(*[set(per_item[m]) for m in models]))
    rng = Random(seed)
    taus = []
    for _ in range(n_boot):
        draw = [items[rng.randrange(len(items))] for _ in items]
        xs, ys = [], []
        for m in models:
            recs = [per_item[m][i] for i in draw]
            xs.append(score_fn(recs, metric_a))
            ys.append(score_fn(recs, metric_b))
        taus.append(kendall_tau_b(xs, ys))
    taus.sort()
    lo = taus[int(0.025 * len(taus))]
    hi = taus[min(int(0.975 * len(taus)), len(taus) - 1)]
    point_x = [score_fn([per_item[m][i] for i in items], metric_a) for m in models]
    point_y = [score_fn([per_item[m][i] for i in items], metric_b) for m in models]
    return {"tau": kendall_tau_b(point_x, point_y), "ci_low": lo, "ci_high": hi,
            "n_boot": n_boot, "n_items": len(items), "models": models}


def rank_swap_frequency(per_item, score_fn, metric_a, metric_b, n_boot=2000, seed=12345):
    """For each model pair, how often A outranks B under each metric across resamples.

    More robust at n=8 than a single tau, and it is the form a reviewer can check:
    "under M1, X beats Y in 91% of resamples; under M2, in 12%."
    """
    models = sorted(per_item)
    items = sorted(set.intersection(*[set(per_item[m]) for m in models]))
    rng = Random(seed)
    wins = {(a, b): [0, 0] for a, b in combinations(models, 2)}
    for _ in range(n_boot):
        draw = [items[rng.randrange(len(items))] for _ in items]
        sa = {m: score_fn([per_item[m][i] for i in draw], metric_a) for m in models}
        sb = {m: score_fn([per_item[m][i] for i in draw], metric_b) for m in models}
        for a, b in wins:
            wins[(a, b)][0] += sa[a] > sa[b]
            wins[(a, b)][1] += sb[a] > sb[b]
    return {f"{a} > {b}": {metric_a: wa / n_boot, metric_b: wb / n_boot}
            for (a, b), (wa, wb) in wins.items()}


def ranking_stability(scores_all, scores_subset, adjacent_pp=1.0):
    """§10 Amendment 1: does dropping truncated items reorder the models?

    Gate is tau >= 0.85 AND no swap between models separated by more than
    adjacent_pp on the full data. The second clause matters because tau alone cannot
    tell a coin-flip between two models 0.5pp apart from a genuine reordering of two
    models 15pp apart, and only the latter threatens a conclusion.
    """
    models = sorted(scores_all)
    tau = kendall_tau_b([scores_all[m] for m in models],
                        [scores_subset[m] for m in models])
    swaps = []
    for a, b in combinations(models, 2):
        da = scores_all[a] - scores_all[b]
        db = scores_subset[a] - scores_subset[b]
        if da * db < 0:
            gap = abs(da) * 100
            swaps.append({"pair": f"{a} / {b}", "gap_pp": round(gap, 2),
                          "non_adjacent": gap > adjacent_pp})
    bad = [s for s in swaps if s["non_adjacent"]]
    return {"tau": tau, "swaps": swaps, "non_adjacent_swaps": bad,
            "pass": tau >= 0.85 and not bad}
