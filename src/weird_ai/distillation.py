
"""
Distillation dataset utilities for Weird AI.

Chapter 8 introduces model distillation:

    teacher model -> synthetic training examples -> student model

For Weird AI, this means preparing teacher-generated parody lyrics and teacher
notes as reusable supervised training data. This module does not train a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import random
from pathlib import Path
from typing import Any, Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


@dataclass
class TeacherExample:
    prompt: str
    teacher_lyrics: str
    teacher_notes: str
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillationRecord:
    prompt: str
    target_text: str
    full_text: str
    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_teacher_example(example: dict) -> bool:
    """
    Validate that a raw teacher example contains usable fields.

    Required fields:
        prompt, teacher_lyrics, teacher_notes, quality_score
    """
    required = ["prompt", "teacher_lyrics", "teacher_notes", "quality_score"]
    for key in required:
        if key not in example:
            return False

    for key in ["prompt", "teacher_lyrics", "teacher_notes"]:
        val = example.get(key)
        if not isinstance(val, str) or not val.strip():
            return False

    try:
        float(example["quality_score"])
    except (ValueError, TypeError):
        return False

    return True


def teacher_example_from_dict(example: dict) -> TeacherExample:
    """Convert a valid dictionary into a TeacherExample."""
    if not validate_teacher_example(example):
        raise ValueError("Invalid teacher example dictionary")

    required = {"prompt", "teacher_lyrics", "teacher_notes", "quality_score"}
    metadata = {k: v for k, v in example.items() if k not in required}

    return TeacherExample(
        prompt=example["prompt"].strip(),
        teacher_lyrics=example["teacher_lyrics"].strip(),
        teacher_notes=example["teacher_notes"].strip(),
        quality_score=float(example["quality_score"]),
        metadata=metadata,
    )


def format_teacher_response(
    teacher_lyrics: str,
    teacher_notes: str,
    *,
    include_thinking: bool = True,
) -> str:
    """Format teacher notes and lyrics as a distillation target."""
    lyrics = teacher_lyrics.strip()
    notes = teacher_notes.strip()
    
    if include_thinking:
        return f"<think>{notes}</think>\n\n{lyrics}"
    return lyrics


def format_training_prompt(prompt: str) -> str:
    """Format a Weird AI prompt for distillation training."""
    return f"User: {prompt.strip()}\nAssistant:"


def build_full_text(prompt: str, target_text: str) -> str:
    """Combine formatted prompt and target text."""
    return f"{format_training_prompt(prompt)}\n{target_text}"


def tokenize_text(tokenizer: TokenizerLike, text: str) -> list[int]:
    """Tokenize text using an object with encode(text) -> list[int]."""
    ids = tokenizer.encode(text)
    if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
        raise ValueError("Tokenizer output must be a list of integers")
    return ids


def create_labels(
    input_ids: list[int],
    prompt_length: int,
    *,
    mask_prompt: bool = True,
    ignore_index: int = -100,
) -> list[int]:
    """
    Create training labels.

    If mask_prompt=True, prompt token labels should be replaced with ignore_index.
    """
    labels = list(input_ids)
    if mask_prompt:
        for i in range(prompt_length):
            if i < len(labels):
                labels[i] = ignore_index
    return labels


def build_distillation_record(
    example: TeacherExample,
    tokenizer: TokenizerLike,
    *,
    include_thinking: bool = True,
    mask_prompt: bool = True,
) -> DistillationRecord:
    """Build one processed distillation record."""
    target_text = format_teacher_response(
        example.teacher_lyrics, example.teacher_notes, include_thinking=include_thinking
    )
    prompt_text = format_training_prompt(example.prompt)
    full_text = build_full_text(example.prompt, target_text)

    input_ids = tokenize_text(tokenizer, full_text)
    prompt_ids = tokenize_text(tokenizer, prompt_text)
    prompt_length = len(prompt_ids)

    labels = create_labels(input_ids, prompt_length, mask_prompt=mask_prompt)

    return DistillationRecord(
        prompt=example.prompt,
        target_text=target_text,
        full_text=full_text,
        input_ids=input_ids,
        labels=labels,
        prompt_length=prompt_length,
        metadata=example.metadata,
    )


def filter_teacher_examples(
    examples: list[TeacherExample],
    *,
    min_quality_score: float = 0.75,
    min_lyric_lines: int = 4,
) -> list[TeacherExample]:
    """Keep only high-quality teacher examples."""
    filtered = []
    for ex in examples:
        if ex.quality_score < min_quality_score:
            continue
        lines = [line for line in ex.teacher_lyrics.replace("\\n", "\n").splitlines() if line.strip()]
        if len(lines) < min_lyric_lines:
            continue
        filtered.append(ex)
    return filtered


def split_records(
    records: list[DistillationRecord],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[DistillationRecord], list[DistillationRecord]]:
    """Split records into train and validation sets."""
    if not (0 <= validation_fraction <= 1):
        raise ValueError("validation_fraction must be between 0 and 1")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    val_size = int(len(shuffled) * validation_fraction)
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]

    return train_records, val_records


def record_to_json_dict(record: DistillationRecord) -> dict:
    """Convert a DistillationRecord into a JSON-serializable dictionary."""
    return asdict(record)


def export_jsonl(records: list[DistillationRecord], path: str | Path) -> None:
    """Export records to a JSONL file."""
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record_to_json_dict(record), f, ensure_ascii=False)
            f.write("\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into dictionaries."""
    p = Path(path)
    result = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                result.append(json.loads(line))
    return result


def summarize_distillation_records(records: list[DistillationRecord]) -> str:
    """Create a readable summary of distillation records."""
    if not records:
        return "No records to summarize."

    num_records = len(records)
    total_input = sum(len(r.input_ids) for r in records)
    total_prompt = sum(r.prompt_length for r in records)
    total_target = sum(len(r.input_ids) - r.prompt_length for r in records)

    avg_input = total_input / num_records
    avg_prompt = total_prompt / num_records
    avg_target = total_target / num_records

    return (
        f"Number of records: {num_records}\n"
        f"Average input length: {avg_input:.2f} tokens\n"
        f"Average prompt length: {avg_prompt:.2f} tokens\n"
        f"Average target length: {avg_target:.2f} tokens"
    )
