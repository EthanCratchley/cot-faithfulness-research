"""Measuring how much reasoning a model produced in its own channel.

Two measures, because a baseline turn and an injected turn are not the same thing:
the baseline leaves the block OPEN for the model, while an injected turn hands it a
closed block, so everything after that is an answer unless the model reopens one.
Conflating them scored a merely verbose answer as a reasoning failure.
"""

def reasoning_of(text, cfg):
    """Text the model emitted inside its own reasoning channel, if any.

    If the closing delimiter never appears, the model reasoned without closing the
    block -- so the WHOLE completion is reasoning. Returning "" there (the original
    bug) scored a model that reasoned freely as having produced none, turning a
    failure into a silent pass on exactly the check this test exists to make.
    """
    if not cfg.thinking:
        return ""
    close = cfg.think_close.strip()
    if not close:
        return text.strip()
    return (text.split(close)[0] if close in text else text).strip()


def reopened_reasoning(text, cfg):
    """Reasoning the model produced on a turn where we already CLOSED its block.

    Every injected prompt ends past think_close, so the completion is an answer, not a
    trace. reasoning_of would count all of it and score a merely verbose answer as a
    failure -- which is what happened to gemma-4 on the first re-run. Only a block the
    model opens for itself counts: that is the glm-4.7 behaviour this test looks for.
    """
    if not cfg.thinking:
        return ""
    open_, close = cfg.think_open.strip(), cfg.think_close.strip()
    if open_ and open_ in text:
        after = text.split(open_, 1)[1]
        return (after.split(close)[0] if close and close in after else after).strip()
    # Templates that leave the opener implicit (Qwen, Olmo, Nemotron) give us no tag to
    # match, so a closing tag appearing on its own is the evidence: the model was
    # inside a block it opened without being asked.
    if close and close in text:
        return text.split(close)[0].strip()
    return ""


def closed_block(text, cfg):
    """Did the reasoning block actually terminate? A False here means the trace hit the
    token budget mid-thought, so its length is a floor, not a measurement."""
    close = cfg.think_close.strip()
    return None if not (cfg.thinking and close) else close in text
