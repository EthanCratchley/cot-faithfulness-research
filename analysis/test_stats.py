"""Checks against hand-computable answers and against scipy where available.

A wrong tau returns a plausible number and no run-time check catches it, so these
cases carry the whole burden of trusting the headline statistic.
"""
import sys
from random import Random
from stats import (bootstrap_tau, cohens_kappa, kendall_tau_b, ranking_stability,
                   rank_swap_frequency)

fails = []


def check(label, got, want, tol=1e-9):
    ok = (got != got and want != want) or abs(got - want) <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {label}: got {got:.6f}, want {want:.6f}")
    if not ok:
        fails.append(label)


print("== kendall tau-b ==")
check("identical order", kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
check("exact reversal", kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
# 4 pairs concordant, 2 discordant, no ties -> (4-2)/6
check("one adjacent swap", kendall_tau_b([1, 2, 3, 4], [2, 1, 4, 3]), 2 / 6)
# Monotone transforms must not change tau -- this is why faithful@k is not independent.
check("monotone transform is invisible",
      kendall_tau_b([0.1, 0.2, 0.3], [1 - (1 - p) ** 32 for p in (0.1, 0.2, 0.3)]), 1.0)
# Ties: x ties one pair, y does not. con=2, dis=0, tx=1, ty=0 -> 2/sqrt(3*2)
check("tie in one variable only", kendall_tau_b([1, 1, 2], [1, 2, 3]), 2 / (3 * 2) ** 0.5)
check("all tied is undefined", kendall_tau_b([1, 1, 1], [1, 1, 1]), float("nan"))

print("\n== cohen's kappa ==")
check("perfect agreement", cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)
# obs=0.5, pa=pb=0.5, exp=0.5 -> 0
check("chance-level agreement", cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]), 0.0)
check("perfect disagreement", cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]), -1.0)
# Both raters all-1: every cell on the diagonal, chance agreement 1.0, so 0/0.
check("constant labels are undefined, not perfect",
      cohens_kappa([1, 1, 1], [1, 1, 1]), float("nan"))

print("\n== ranking stability (Amendment 1) ==")
same = {"a": 0.86, "b": 0.855, "c": 0.75, "d": 0.68}
shifted = {"a": 0.951, "b": 0.951, "c": 0.943, "d": 0.837}   # b catches a
r = ranking_stability(same, shifted)
print(f"{'ok  ' if r['pass'] else 'FAIL'} adjacent tie does not fail the gate: "
      f"tau={r['tau']:.3f} swaps={len(r['swaps'])} pass={r['pass']}")
fails.append("adjacent") if not r["pass"] else None
reordered = {"a": 0.60, "b": 0.855, "c": 0.75, "d": 0.68}    # a falls 26pp
r2 = ranking_stability(same, reordered)
ok = not r2["pass"] and r2["non_adjacent_swaps"]
print(f"{'ok  ' if ok else 'FAIL'} large reordering fails the gate: "
      f"tau={r2['tau']:.3f} non-adjacent swaps={len(r2['non_adjacent_swaps'])}")
if not ok:
    fails.append("non-adjacent")

print("\n== bootstrap CI ==")
# Two metrics that agree perfectly by construction: the CI must sit at 1.0.
rng = Random(7)
per_item = {f"m{k}": {i: {"x": rng.random() + k, "y": rng.random() + k}
                      for i in range(120)} for k in range(6)}
score = lambda recs, m: sum(r[m] for r in recs) / len(recs)
agree = bootstrap_tau(per_item, score, "x", "y", n_boot=300)
ok = agree["tau"] == 1.0 and agree["ci_low"] > 0.5
print(f"{'ok  ' if ok else 'FAIL'} separated models give tau=1 with a tight CI: "
      f"tau={agree['tau']:.3f} CI=[{agree['ci_low']:.3f},{agree['ci_high']:.3f}]")
if not ok:
    fails.append("bootstrap-agree")

# Pure noise: the CI must be wide enough to include 0, or the CI is lying.
noise = {f"m{k}": {i: {"x": rng.random(), "y": rng.random()} for i in range(120)}
         for k in range(6)}
nz = bootstrap_tau(noise, score, "x", "y", n_boot=300)
ok = nz["ci_low"] < 0 < nz["ci_high"]
print(f"{'ok  ' if ok else 'FAIL'} unrelated metrics give a CI spanning 0: "
      f"tau={nz['tau']:.3f} CI=[{nz['ci_low']:.3f},{nz['ci_high']:.3f}]")
if not ok:
    fails.append("bootstrap-noise")

print("\n== rank swap frequency ==")
sw = rank_swap_frequency(per_item, score, "x", "y", n_boot=200)
# Pairs are keyed in sorted order, so this reads "does m0 beat m5" -- and m0's scores
# are 5 units lower by construction, so the answer must be never, under both metrics.
k = "m0 > m5"
ok = sw[k]["x"] == 0.0 and sw[k]["y"] == 0.0
print(f"{'ok  ' if ok else 'FAIL'} a 5-unit gap never swaps: {k} -> {sw[k]}")
if not ok:
    fails.append("swap-freq")

try:
    from scipy.stats import kendalltau
    print("\n== cross-check against scipy ==")
    r = Random(3)
    worst = 0.0
    for _ in range(200):
        n = r.randint(4, 9)
        x = [r.randint(0, 4) for _ in range(n)]   # small range forces ties
        y = [r.randint(0, 4) for _ in range(n)]
        mine, theirs = kendall_tau_b(x, y), kendalltau(x, y).statistic
        if mine == mine and theirs == theirs:
            worst = max(worst, abs(mine - theirs))
    ok = worst < 1e-9
    print(f"{'ok  ' if ok else 'FAIL'} 200 random tie-heavy vectors, max abs diff {worst:.2e}")
    if not ok:
        fails.append("scipy")
except ImportError:
    print("\n(scipy not installed; skipping cross-check)")

print("\nall passed" if not fails else f"\n{len(fails)} failed: {fails}")
sys.exit(1 if fails else 0)
