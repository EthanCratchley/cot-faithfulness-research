"""One answer extractor, shared by every condition.

§5 commits to a single extraction path: if baseline, hinted, filler and
early-answering rows were parsed differently, a tau between two metrics could be an
artifact of the parsers rather than of the metrics.
"""
import re
import string

# Placed immediately after an injected trace so the answer cannot be conditioned on
# reasoning the model writes for itself. The open paren matches the option format,
# '(A) 1 s', so the next token has to be a letter -- cued with a bare 'Answer:',
# Olmo-3.1-32B-Think replied ' 4', the value, which parses as nothing.
ANSWER_CUE = "\n\nAnswer: ("


def letters(n):
    """Option letters for an n-way question. MMLU-Pro runs to J."""
    return string.ascii_uppercase[:n]


def extract(text, n_options=10):
    """The model's chosen option letter, or None.

    Preference order matters. An explicit 'Answer: D' beats a letter appearing anywhere
    else, and the LAST such marker wins because models restate options before
    concluding.

    A bare letter is only accepted when it is the ENTIRE response. Across ten options
    'A' and 'I' are ordinary English words, so a looser fallback reads "I am not sure"
    as answer I -- confidently wrong, and worse than recording nothing. Unparsed rows
    are counted and gated (§10 Step 2, Gate B) rather than guessed at.
    """
    if not text:
        return None
    hi = letters(n_options)[-1]
    marked = re.findall(rf"[Aa]nswer[^A-{hi}]{{0,12}}([A-{hi}])\b", text)
    if marked:
        return marked[-1]
    paren = re.findall(rf"\(([A-{hi}])\)", text)
    if paren:
        return paren[-1]
    bare = re.fullmatch(rf"\(?([A-{hi}])[).:,]?", text.strip())
    return bare.group(1) if bare else None
