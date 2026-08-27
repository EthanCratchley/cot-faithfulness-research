"""Prompt construction with structurally-verified CoT injection.

The local design rests on one thing: text we write must land in the model's reasoning
channel, exactly where we put it. Each model family uses a different channel syntax,
so the delimiters below are read off each model's own chat template, not guessed.

`build`  returns the literal string handed to vLLM's completions endpoint.
`verify` proves the injected span sits *between* the reasoning delimiters -- not
merely that it appears somewhere in the prompt. Run it per model before generating.

Checking only for presence is what the API path did by proxy, and it is why a dropped
injection looked identical to a successful one for five rounds of probing.
"""

from dataclasses import dataclass, field

# Mistral publishes no `chat_template`; it ships a tekken tokenizer and expects
# mistral-common. V7 instruct format, transcribed here so the model needs no extra dep.
MISTRAL_V7 = (
    "{%- for m in messages %}"
    "{%- if m['role'] == 'user' %}{{- '[INST]' + m['content'] + '[/INST]' }}"
    "{%- elif m['role'] == 'assistant' %}{{- m['content'] + '</s>' }}"
    "{%- endif %}{%- endfor %}"
)


@dataclass
class ModelCfg:
    repo: str
    lab: str
    thinking: bool
    # Delimiters around the reasoning channel. Empty think_open means the chat
    # template already leaves the channel open after add_generation_prompt.
    think_open: str = ""
    think_close: str = "</think>\n\n"
    # Emitted after the reasoning channel closes, where the family requires the
    # final answer to start a new block (harmony-style formats).
    final_open: str = ""
    template_kwargs: dict = field(default_factory=dict)
    chat_template: str | None = None


MODELS = [
    # ChatML + <think>: the template leaves <think> open, so think_open is empty.
    ModelCfg("Qwen/Qwen3.8-27B", "Alibaba", True,
             template_kwargs={"enable_thinking": True}),
    ModelCfg("allenai/Olmo-3.1-32B-Think", "Ai2", True),
    ModelCfg("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "NVIDIA", True,
             template_kwargs={"enable_thinking": True}),

    # Gemma-4: asymmetric thought tags, '<|channel>' opens and '<channel|>' closes.
    ModelCfg("google/gemma-4-31B-it", "Google", True,
             think_open="<|channel>thought\n", think_close="\n<channel|>",
             template_kwargs={"enable_thinking": True}),

    # Harmony: analysis channel, then a fresh assistant block for the final answer.
    ModelCfg("openai/gpt-oss-20b", "OpenAI", True,
             think_open="<|channel|>analysis<|message|>", think_close="<|end|>",
             final_open="<|start|>assistant<|channel|>final<|message|>"),

    # Muse ATEM: the reasoning channel is addressed to self.
    # ATEM addresses every message to a recipient. Omitting 'to=user' on the final
    # block is not cosmetic: the model read the recipient-less header as another self
    # turn, reasoned 697 more characters, closed it and opened a proper 'to=user'
    # block of its own -- visible in its baseline output, which ends
    # '<|eom|><|start|>assistant to=user<|message|>'.
    ModelCfg("meta-models/Muse-Glimmer-30B", "Meta", True,
             think_open=" to=self<|message|>", think_close="<|eom|>",
             final_open="<|start|>assistant to=user<|message|>"),

    # No reasoning channel: the CoT is the assistant turn, so injection is a prefix
    # of the response and is left open for the model to continue.
    ModelCfg("allenai/Olmo-3.1-32B-Instruct", "Ai2", False),
    ModelCfg("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "Mistral", False,
             chat_template=MISTRAL_V7),
]

BY_REPO = {m.repo: m for m in MODELS}


def chat_prefix(tok, cfg, question, thinking=None):
    """Generation prompt for a single user turn, as a string."""
    kwargs = dict(cfg.template_kwargs)
    if thinking is not None and "enable_thinking" in kwargs:
        kwargs["enable_thinking"] = thinking
    if cfg.chat_template:
        kwargs["chat_template"] = cfg.chat_template
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True, tokenize=False, **kwargs)


def build(tok, cfg, question, injected_cot=None):
    """Full prompt string.

    injected_cot=None -> the model produces its own reasoning
                         (Biasing Features, faithful@k)
    injected_cot=str  -> our text becomes the reasoning, and the model may only
                         produce the answer that follows it
                         (Filler Tokens, Early Answering)
    """
    prompt = chat_prefix(tok, cfg, question)
    if injected_cot is None:
        return prompt
    if not cfg.thinking:
        return prompt + injected_cot
    return prompt + cfg.think_open + injected_cot + cfg.think_close + cfg.final_open


def verify(tok, cfg, question, injected_cot):
    """Prove the injected span is inside the reasoning channel.

    `ok` requires all of:
      - the injected text survives encode/decode
      - for thinking models, it sits between think_open and think_close
      - nothing follows the final_open block, so the model's next token is the answer
    """
    prompt = build(tok, cfg, question, injected_cot)
    decoded = tok.decode(tok.encode(prompt, add_special_tokens=False))
    needle = injected_cot.strip()

    survives = needle in decoded
    if not cfg.thinking:
        structural = prompt.endswith(injected_cot)
        detail = "CoT is the assistant turn; injection is its prefix"
    else:
        opener = cfg.think_open or "<think>"
        oi, ni = prompt.rfind(opener), prompt.rfind(needle)
        ci = prompt.find(cfg.think_close, ni) if ni >= 0 else -1
        structural = oi >= 0 and ni > oi and ci > ni
        detail = f"opener@{oi} injected@{ni} closer@{ci}"

    return {
        "ok": survives and structural,
        "survives_tokenization": survives,
        "inside_reasoning_channel": structural,
        "detail": detail,
        "prompt_tokens": len(tok.encode(prompt, add_special_tokens=False)),
        "tail": prompt[-110:],
    }
