"""Offline checks on the two reasoning measures. No GPU, no weights.

Both measurement bugs that cost a re-run are pinned here: a baseline whose closing
tag never arrives, and an injected turn that answers verbosely without reopening.
"""
import sys
from prompts import BY_REPO
from reasoning import reasoning_of, reopened_reasoning
from answers import ANSWER_CUE, extract

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

# The shared extractor. MMLU-Pro runs to J, so A-E is not enough, and a wrong answer
# must never be silently read as "no answer" or vice versa -- Step 1L passed a model
# whose answer did not parse at all.
ANSWER_CASES = [
    ("explicit marker", "The total is 4 s.\n\nAnswer: D", 10, "D"),
    ("last marker wins", "Answer: B at first, but on reflection Answer: D", 10, "D"),
    ("cue plus bare letter", ANSWER_CUE + "D) 4 s", 10, "D"),
    ("cue plus letter alone", ANSWER_CUE + "J", 10, "J"),
    ("marker beats a stray paren", "We rule out (A) here. Answer: F", 10, "F"),
    # Mistral answered one pilot item with LaTeX and nothing else.
    ("latex boxed answer", "only Option F satisfies it.\n\n\\[\n\\boxed{F}\n\\]", 10, "F"),
    ("boxed with parens", "\\boxed{(H)}", 10, "H"),
    ("paren fallback with no marker", "The correct option is (H) by elimination.", 10, "H"),
    ("J is out of range for a 5-option item", "Answer: J", 5, None),
    ("bare letter alone is the whole answer", "D", 10, "D"),
    ("bare letter with punctuation", "(F)", 10, "F"),
    # 'A' and 'I' are English words; a loose fallback read this as answer I.
    ("prose starting with a letter word does not parse", "I am not sure.", 10, None),
    ("an article is not an answer", "A ball is thrown upward.", 10, None),
    # The exact Olmo failure: the value, not the option.
    ("value instead of option does not parse", ANSWER_CUE + " 4", 10, None),
    ("empty text", "", 10, None),
]

fails = 0
for label, text, n, want in ANSWER_CASES:
    got = extract(text, n)
    ok = got == want
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} answer/{label}: got {got!r}, want {want!r}")

for label, fn, text, cfg, want in CASES:
    got = len(fn(text, cfg))
    ok = got == want
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {label}: got {got}, want {want}")

print("\nall passed" if not fails else f"\n{fails} failed")
sys.exit(1 if fails else 0)
