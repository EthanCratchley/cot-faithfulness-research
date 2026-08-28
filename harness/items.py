"""Frozen MMLU-Pro item sets.

The pool is built once, committed, and read from disk by every run. Sampling live from
HuggingFace on each run would let a dataset revision quietly change the item set
between models, which would show up as a model difference.

The pilot is a SUBSET of the pool, not an independent draw, so Step 2 accuracy
actually predicts Step 6 accuracy on the same items.
"""
import json
from collections import Counter, defaultdict
from random import Random

DATASET = "TIGER-Lab/MMLU-Pro"
SPLIT = "test"
SEED = 12345
POOL_N = 500
PILOT_N = 200


def allocate(counts, total):
    """Proportional allocation, largest remainder for the leftovers.

    Proportional rather than equal-per-category so the pool stays representative of
    MMLU-Pro and accuracy remains comparable to published numbers; equal allocation
    would over-weight history and under-weight math by roughly four times.
    """
    n = sum(counts.values())
    exact = {k: v * total / n for k, v in counts.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    for k in sorted(counts, key=lambda k: (-(exact[k] - alloc[k]), k))[:total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def stratified(rows, total, seed):
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    alloc = allocate({k: len(v) for k, v in by_cat.items()}, total)
    rng, out = Random(seed), []
    for cat in sorted(by_cat):
        # Sort before sampling: HF row order is not a stable contract, so seeding
        # alone would not make this reproducible.
        pool = sorted(by_cat[cat], key=lambda r: r["question_id"])
        out += rng.sample(pool, alloc[cat])
    return sorted(out, key=lambda r: r["question_id"])


def format_question(item):
    """Prompt body. Identical across every condition -- see harness/answers.py."""
    opts = "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(item["options"]))
    return (f"{item['question']}\n{opts}\n"
            f"End with 'Answer: X' where X is the option letter.")


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    from datasets import load_dataset
    ds = load_dataset(DATASET, split=SPLIT)
    rows = [{"question_id": r["question_id"], "question": r["question"],
             "options": [o for o in r["options"] if o and o != "N/A"],
             "answer": r["answer"], "category": r["category"], "src": r["src"]}
            for r in ds]
    # A hint has to be able to point at a specific wrong option, so drop anything
    # with too few to make that meaningful.
    rows = [r for r in rows if len(r["options"]) >= 4
            and r["answer"] in [chr(65 + i) for i in range(len(r["options"]))]]

    pool = stratified(rows, POOL_N, SEED)
    pilot = stratified(pool, PILOT_N, SEED)
    assert {r["question_id"] for r in pilot} <= {r["question_id"] for r in pool}

    for name, sel, n in [("pool", pool, POOL_N), ("pilot", pilot, PILOT_N)]:
        assert len(sel) == n, (name, len(sel))
        out = {"dataset": DATASET, "split": SPLIT, "seed": SEED, "n": n,
               "fingerprint": ds._fingerprint, "source_rows": len(rows),
               "categories": dict(sorted(Counter(r["category"] for r in sel).items())),
               "items": sel}
        path = f"data/items_{name}_{n}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"{path}: {n} items, {len(out['categories'])} categories")
        print("   ", out["categories"])


if __name__ == "__main__":
    main()
