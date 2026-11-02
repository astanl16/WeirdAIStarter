"""
Best-of-N generation utilities for Weird AI.

This module connects the Chapter 4 idea of inference-time scaling to the
Weird AI project.

The core idea is:

    generate one candidate     -> maybe good, maybe not
    generate several candidates -> evaluate each one
    select the best candidate   -> better final output without retraining

Students complete this file as the applied assignment for this lesson.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional


GenerationFunction = Callable[..., str]
EvaluationFunction = Callable[[str], dict]


@dataclass
class GenerationCandidate:
    """
    Store one generated candidate and its evaluation results.

    Attributes:
        text: The generated text.
        score: Overall evaluation score for the text.
        evaluation: Full evaluation dictionary returned by the evaluator.
        index: Candidate number, starting at 1.
    """

    text: str
    score: float
    evaluation: dict
    index: int


def build_generation_prompt(
    topic: str, style: str = "emo parody", line_count: int = 8
) -> str:
    """
    Build a prompt for Weird AI parody generation.
    """

    return (
        f"Write a {line_count}-line {style} song about {topic}.\n\n"
        "Requirements:\n"
        f"- Exactly {line_count} lyric lines.\n"
        "- Keep each line short enough to be sung.\n"
        "- Aim for a consistent rhyme scheme (AABB if possible).\n"
        "- Make it funny but still sound dramatic.\n"
        "- Return ONLY the lyrics. Do not include a title, explanation, "
        "or commentary.\n"
    )


def generate_once(
    prompt: str,
    generation_func: GenerationFunction,
    *,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    max_new_tokens: int = 120,
    **kwargs: Any,
) -> str:
    """
    Generate a single candidate response.
    """

    text = generation_func(
        prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
    return text if isinstance(text, str) else str(text)


def generate_multiple(
    prompt: str,
    generation_func: GenerationFunction,
    *,
    num_candidates: int = 5,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    max_new_tokens: int = 120,
    **kwargs: Any,
) -> list[str]:
    """
    Generate several candidate responses for the same prompt.
    """

    if num_candidates < 1:
        raise ValueError(
            f"num_candidates must be at least 1, got {num_candidates}"
        )

    return [
        generate_once(
            prompt,
            generation_func,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        for _ in range(num_candidates)
    ]


def score_candidate(
    text: str,
    evaluation_func: EvaluationFunction,
    *,
    score_key: str = "overall_score",
) -> GenerationCandidate:
    """
    Evaluate a single generated candidate.
    """

    evaluation = evaluation_func(text)
    score = float(evaluation.get(score_key, 0.0))
    return GenerationCandidate(
        text=text,
        score=score,
        evaluation=evaluation,
        index=0,  # caller may overwrite
    )


def rank_candidates(
    texts: Iterable[str],
    evaluation_func: EvaluationFunction,
    *,
    score_key: str = "overall_score",
) -> list[GenerationCandidate]:
    """
    Evaluate and rank generated candidates from best to worst.
    """

    candidates: list[GenerationCandidate] = []
    for i, text in enumerate(texts, start=1):
        candidate = score_candidate(
            text, evaluation_func, score_key=score_key
        )
        candidate.index = i
        candidates.append(candidate)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def select_best_candidate(
    candidates: list[GenerationCandidate],
) -> GenerationCandidate:
    """
    Select the best candidate from a ranked or unranked list.
    """

    if not candidates:
        raise ValueError("Cannot select best candidate from an empty list.")
    return max(candidates, key=lambda c: c.score)


def generate_best(
    prompt: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    num_candidates: int = 5,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    max_new_tokens: int = 120,
    score_key: str = "overall_score",
    **kwargs: Any,
) -> dict:
    """
    Generate multiple candidates, evaluate them, and return the best result.
    """

    texts = generate_multiple(
        prompt,
        generation_func,
        num_candidates=num_candidates,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )

    ranked = rank_candidates(texts, evaluation_func, score_key=score_key)
    best = select_best_candidate(ranked)

    return {
        "best_text": best.text,
        "best_score": best.score,
        "best_candidate": best,
        "candidates": ranked,
    }


def format_candidate_report(candidates: list[GenerationCandidate]) -> str:
    """
    Format candidate scores as a readable report.
    """

    if not candidates:
        return "(no candidates to report)"

    sections = []
    for c in candidates:
        lines = [f"Candidate {c.index}: overall_score={c.score:.3f}"]

        for key in (
            "rhyme_score",
            "structure_score",
            "syllable_consistency_score",
            "syllable_score",   # fallback if used elsewhere
            "line_count",
            "passed",
        ):
            if key in c.evaluation:
                value = c.evaluation[key]
                if isinstance(value, float):
                    lines.append(f"  {key}={value:.3f}")
                else:
                    lines.append(f"  {key}={value}")

        lines.append("  text:")
        for lyric_line in c.text.strip().splitlines():
            lines.append(f"    {lyric_line}")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)