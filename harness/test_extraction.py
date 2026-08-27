"""Offline checks on the two reasoning measures. No GPU, no weights.

Both measurement bugs that cost a re-run are pinned here: a baseline whose closing
tag never arrives, and an injected turn that answers verbosely without reopening.
"""
import sys
from prompts import BY_REPO
from gpu_smoke_test import reasoning_of, reopened_reasoning

qwen = BY_REPO["Qwen/Qwen3.8-27B"]                      # think_open "", close </think>
gemma = BY_REPO["google/gemma-4-31B-it"]                # <|channel>thought ... <channel|>
mistral = BY_REPO["mistralai/Mistral-Small-3.2-24B-Instruct-2506"]   # no thinking

CASES = [
    # (label, fn, text, cfg, expected_len)
    ("baseline splits at close",
     reasoning_of, "thinking hard</think>\n\nAnswer: D", qwen, len("thinking hard")),
    # The Olmo failure: budget ran out mid-thought, so the whole trace is reasoning.
    ("baseline unterminated counts in full",
     reasoning_of, "still thinking and never closed", qwen, len("still thinking and never closed")),
    ("baseline n/a without a thinking channel",
     reasoning_of, "Answer: D", mistral, 0),

    # The gemma-4 failure: a verbose ANSWER is not reasoning. We closed the block.
    ("verbose answer is not reopened reasoning",
     reopened_reasoning, "To find the time, use y = v0*t - g*t^2/2. Answer: D", gemma, 0),
    ("reopened block counts, opener known",
     reopened_reasoning, "sure<|channel>thought\nsecretly reasoning\n<channel|>Answer: D",
     gemma, len("secretly reasoning")),
    # Qwen/Olmo/Nemotron have no opener to match; a stray close is the tell. This is
    # the Nemotron result -- 200 dots handed over, reasons anyway.
    ("stray close reveals an implicit reopen",
     reopened_reasoning, "reasoning anyway</think>\n\nAnswer: B", qwen, len("reasoning anyway")),
    ("reopened but unterminated still counts",
     reopened_reasoning, "x<|channel>thought\ncut off mid", gemma, len("cut off mid")),
]

fails = 0
for label, fn, text, cfg, want in CASES:
    got = len(fn(text, cfg))
    ok = got == want
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {label}: got {got}, want {want}")

print("\nall passed" if not fails else f"\n{fails} failed")
sys.exit(1 if fails else 0)
