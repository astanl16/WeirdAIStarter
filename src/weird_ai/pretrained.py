"""
Pretrained model helpers for Weird AI.

Lesson 9 introduces students to using an already-trained language model for
text generation. Earlier lessons built many of the pieces from scratch. In this
lesson, students use a pretrained model so they can focus on the complete
inference pipeline:

    text prompt -> tokenizer -> token IDs -> model -> generated token IDs -> text

The functions in this file are intentionally scaffolded. Complete the TODO
sections, then run:

    python -m pytest tests/test_pretrained.py

The unit tests use small fake model and tokenizer objects, so they do not need
to download a real Hugging Face model.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

from weird_ai.reasoning import build_reasoning_prompt

DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"


def get_device(preferred_device: str | None = None) -> torch.device:
    """
    Choose the device that should be used for model inference.

    Args:
        preferred_device: Optional device name such as "cpu" or "cuda".

    Returns:
        A torch.device object.
    """

    if preferred_device is not None:
        return torch.device(preferred_device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """
    Load a pretrained tokenizer.

    Args:
        model_name: Hugging Face model name or local model path.

    Returns:
        A tokenizer object loaded with AutoTokenizer.from_pretrained().
    """

    if AutoTokenizer is None:
        raise ImportError(
            "transformers is required for load_tokenizer(). "
            "Install it with: pip install transformers"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | torch.device | None = None,
) -> Any:
    """
    Load a pretrained causal language model for text generation.

    Args:
        model_name: Hugging Face model name or local model path.
        device: Optional device name or torch.device.

    Returns:
        A model object in evaluation mode.
    """

    if AutoModelForCausalLM is None:
        raise ImportError(
            "transformers is required for load_model(). "
            "Install it with: pip install transformers"
        )

    device_obj = get_device(device)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device_obj)
    model.eval()

    return model


def tokenize_prompt(
    tokenizer: Any,
    prompt: str,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """
    Convert a text prompt into model input tensors.

    Args:
        tokenizer: Tokenizer used to encode the prompt.
        prompt: User prompt text.
        device: Optional device where tensors should be placed.

    Returns:
        A dictionary of input tensors that can be passed to model.generate().
    """

    device_obj = get_device(device)

    inputs = tokenizer(prompt, return_tensors="pt")

    return {
        key: value.to(device_obj) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def decode_generated_tokens(
    tokenizer: Any,
    generated_token_ids: torch.Tensor,
    skip_special_tokens: bool = True,
) -> str:
    """
    Decode generated token IDs into text.

    Args:
        tokenizer: Tokenizer used to decode token IDs.
        generated_token_ids: Tensor containing generated token IDs.
        skip_special_tokens: Whether special tokens should be hidden.

    Returns:
        The decoded text as a string.
    """

    if generated_token_ids.dim() > 1:
        generated_token_ids = generated_token_ids[0]

    if isinstance(generated_token_ids, torch.Tensor):
        generated_token_ids = generated_token_ids.tolist()

    return tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=skip_special_tokens,
    )


def generate_text(
    prompt: str,
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 50,
    do_sample: bool = True,
    device: str | torch.device | None = None,
) -> str:
    """
    Generate text from a prompt using a pretrained causal language model.
    """

    device_obj = get_device(device)

    if tokenizer is None:
        tokenizer = load_tokenizer(model_name)

    if model is None:
        model = load_model(model_name, device=device_obj)

    inputs = tokenize_prompt(tokenizer, prompt, device=device_obj)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
        )

    return decode_generated_tokens(tokenizer, output_ids)


def generate_with_reasoning(
    user_prompt: str,
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str | torch.device | None = None,
) -> str:
    """
    Generate text after wrapping the original prompt in a reasoning prompt.
    """

    reasoning_prompt = build_reasoning_prompt(user_prompt)

    return generate_text(
        reasoning_prompt,
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )