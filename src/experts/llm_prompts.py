"""Module A — LLM-enriched class prototypes (CuPL-style).

A frozen CLIP turns a class *name* into a text prototype, but "a photo of a {name}"
is a thin description — especially for rare tail classes the model has weak priors
on. CuPL (Pratt et al., ICCV 2023) instead asks a **language model** to *describe*
each class ("a leopard is a large wild cat with golden fur and black rosettes…"),
encodes many such sentences with CLIP's text encoder, and averages them into a
richer prototype. The extra semantic detail helps most exactly where data is
scarce — the tail.

Pipeline:
1. ``generate_descriptions`` — run a small open LLM on Kaggle to produce N sentences
   per class. Cached to JSON so it runs once (``save_descriptions`` / ``load_descriptions``).
2. ``src.experts.clip_expert.encode_text_prototypes`` — average their CLIP embeddings
   into ``(C, D)`` prototypes, used as a drop-in for the zero-shot ``text_features``
   and as LIFT's cosine-head initialisation.

The LLM step needs ``transformers`` (preinstalled on Kaggle) + internet to download
the model the first time. Everything else is text and runs in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

# Generic CuPL-style prompts; "{}" is replaced by the (underscore-free) class name.
DEFAULT_QUERY_TEMPLATES = [
    "Describe what a {} looks like in a single sentence.",
    "How can you identify a {} in a photo? Answer in one sentence.",
    "Describe the visual appearance of a {} (color, shape, texture).",
    "What does a {} look like? Give a short visual description.",
]


def generate_descriptions(class_names: list[str], device,
                          llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                          n_per_class: int = 8, max_new_tokens: int = 48,
                          templates: list[str] | None = None) -> dict[str, list[str]]:
    """Generate ``n_per_class`` short visual descriptions per class with a local LLM.

    Uses a small instruction-tuned model via ``transformers``. Cheap on a Kaggle GPU
    (a few minutes for 100 classes). Returns ``{class_name: [description, ...]}``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    templates = templates or DEFAULT_QUERY_TEMPLATES
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models must pad on the LEFT for correct batched generation
    # (right-padding shifts the generation start and corrupts the output).
    tokenizer.padding_side = "left"

    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name, torch_dtype=dtype, device_map=None).to(device).eval()

    descriptions: dict[str, list[str]] = {}
    for name in class_names:
        readable = name.replace("_", " ")
        # Round-robin the templates until we have n_per_class prompts.
        prompts = [templates[i % len(templates)].format(readable) for i in range(n_per_class)]
        chats = [tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in prompts]
        batch = tokenizer(chats, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.9, top_p=0.95, pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        # Keep the class name in each sentence so CLIP still grounds on it; clean whitespace.
        cleaned = [f"{readable}: {g.strip().splitlines()[0]}" if g.strip() else f"a photo of a {readable}"
                   for g in gen]
        descriptions[name] = cleaned

    # Free the LLM before the rest of the notebook loads CLIP / DINOv2 features.
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return descriptions


def save_descriptions(descriptions: dict[str, list[str]], path) -> None:
    """Persist generated descriptions so the LLM only has to run once."""
    Path(path).write_text(json.dumps(descriptions, indent=2, ensure_ascii=False), encoding="utf-8")


def load_descriptions(path) -> dict[str, list[str]]:
    """Load a cached descriptions JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prompts_in_label_order(descriptions: dict[str, list[str]],
                           class_names: list[str]) -> list[list[str]]:
    """Return descriptions as a list aligned to class id order (for ``encode_text_prototypes``).

    Falls back to the plain ``"a photo of a {name}"`` prompt for any class missing
    from the dict, so a partial generation never crashes the pipeline.
    """
    return [descriptions.get(name, [f"a photo of a {name.replace('_', ' ')}"])
            for name in class_names]
