"""Frozen ARC-Easy item set for the Step 5 anchor.

ARC-Easy, not MMLU-Pro: the anchor reproduces arXiv:2512.23032 Table 1, whose per-model
professor-hint numbers are reported on ARC-Easy. Reproducing their number on a different
dataset would not be a reproduction.

Built once and committed, for the same reason the MMLU-Pro pool is: a live draw would
let a dataset revision change the item set between models.
"""
import hashlib
import json
from random import Random

DATASET = "allenai/ai2_arc"
CONFIG = "ARC-Easy"
SPLIT = "test"
SEED = 12345
N = 500


def build(out_path):
    from datasets import load_dataset
    ds = load_dataset(DATASET, CONFIG, split=SPLIT)
    rows = []
    for r in ds:
        labels = list(r["choices"]["label"])
        texts = list(r["choices"]["text"])
        # A handful of ARC items are numbered 1-4 rather than lettered; normalise to
        # letters so one extractor and one option format serve every condition.
        if not all(l.isalpha() for l in labels):
            labels = [chr(65 + i) for i in range(len(texts))]
            answer = chr(65 + list(r["choices"]["label"]).index(r["answerKey"])) \
                if r["answerKey"] in r["choices"]["label"] else None
        else:
            answer = r["answerKey"]
        if answer is None or answer not in labels:
            continue
        rows.append({"question_id": r["id"], "category": "arc-easy",
                     "question": r["question"], "options": texts, "answer": answer})
    rows.sort(key=lambda r: r["question_id"])
    items = sorted(Random(SEED).sample(rows, min(N, len(rows))),
                   key=lambda r: r["question_id"])
    fp = hashlib.sha256(
        json.dumps([i["question_id"] for i in items]).encode()).hexdigest()[:16]
    meta = {"dataset": DATASET, "config": CONFIG, "split": SPLIT, "seed": SEED,
            "n": len(items), "fingerprint": fp, "source_rows": len(rows),
            "items": items}
    with open(out_path, "w") as f:
        json.dump(meta, f)
    print(f"wrote {out_path}: {len(items)} items, fingerprint {fp}")
    return meta


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "data/anchor_arc_easy_500.json")
