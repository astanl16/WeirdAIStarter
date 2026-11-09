"""
Self-refinement utilities for Weird AI.

This module connects Chapter 5 of Build a Reasoning Model from Scratch to the
Weird AI project.

Chapter 4 added best-of-N generation:

    generate several candidates -> evaluate each -> select the best

Chapter 5 adds self-refinement:

    generate one candidate -> evaluate -> ask for a revision -> evaluate again
    -> keep the better version -> repeat

The goal is to improve output quality during inference without retraining the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


GenerationFunction = Callable[..., str]
EvaluationFunction = Callable[[str], dict]


@dataclass
class RefinementStep:
    """Store information about one refinement attempt."""

    iteration: int
    prompt: str
    previous_text: str
    revised_text: str
    previous_score: float
    revised_score: float
    accepted: bool
    evaluation: dict = field(default_factory=dict)


def get_score(evaluation: dict, score_key: str = "overall_score") -> float:
    """Extract a numeric score from an evaluation dictionary."""

    val = evaluation.get(score_key, 0.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def build_refinement_prompt(
    original_prompt: str,
    current_text: str,
    evaluation: dict,
    *,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    score_key: str = "overall_score",
) -> str:
    """Build a prompt asking the model to improve a generated parody."""

    score = evaluation.get(score_key, 0.0)

    details = []
    for k in ("rhyme_score", "syllable_consistency_score", "structure_score"):
        if k in evaluation:
            val = evaluation[k]
            if isinstance(val, float):
                details.append(f"- {k}: {val:.3f}")
            else:
                details.append(f"- {k}: {val}")
    details_str = "\n".join(details) if details else "- No details available for component scores"

    focus_str = "\n".join(f"- {f}" for f in focus) if focus else "- Improve overall quality"
    preserve_str = "\n".join(f"- {p}" for p in preserve) if preserve else "- Preserve the original topic, tone, and length"
    
    return (
        f"Original task:\n{original_prompt}\n\n"
        f"Current lyrics:\n{current_text}\n\n"
        f"Current evaluation (overall score: {score}):\n{details_str}\n\n"
        f"Focus on these improvements:\n{focus_str}\n\n"
        f"Preserve these aspects:\n{preserve_str}\n\n"
        "Revise the lyrics to address the focus areas while preserving the key aspects.\n"
        "Return ONLY the revised lyrics. Do not include any explanation, title, or commentary."
    )

def refine_once(
    original_prompt: str,
    current_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> RefinementStep:
    """Run one self-refinement step."""

    current_eval = evaluation_func(current_text)
    previous_score = get_score(current_eval, score_key)
    
    prompt = build_refinement_prompt(
        original_prompt,
        current_text,
        current_eval,
        focus=focus,
        preserve=preserve,
        score_key=score_key,
    )
    
    revised_text = generation_func(
        prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
    
    revised_eval = evaluation_func(revised_text)
    revised_score = get_score(revised_eval, score_key)
    
    accepted = revised_score > previous_score
    return RefinementStep(
        iteration=0,
        prompt=prompt,
        previous_text=current_text,
        revised_text=revised_text,
        previous_score=previous_score,
        revised_score=revised_score,
        accepted=accepted,
        evaluation=revised_eval,
    )


def choose_better_text(
    current_text: str,
    revised_text: str,
    current_evaluation: dict,
    revised_evaluation: dict,
    *,
    score_key: str = "overall_score",
    require_improvement: bool = True,
) -> tuple[str, dict, bool]:
    """Decide whether to accept a revised text."""

    current_score = get_score(current_evaluation, score_key)
    revised_score = get_score(revised_evaluation, score_key)
    
    if require_improvement:
        accepted = revised_score > current_score
    else:
        accepted = revised_score >= current_score
        
    if accepted:
        return revised_text, revised_evaluation, True
    return current_text, current_evaluation, False


def should_continue(
    *,
    iteration: int,
    max_iterations: int,
    current_score: float,
    target_score: Optional[float] = None,
    last_improved: bool = True,
    stop_on_no_improvement: bool = True,
) -> bool:
    """Decide whether the refinement loop should continue."""

    if iteration >= max_iterations:
        return False
    if target_score is not None and current_score >= target_score:
        return False
    if stop_on_no_improvement and not last_improved:
        return False
    return True


def iterative_refinement(
    original_prompt: str,
    initial_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    max_iterations: int = 3,
    target_score: Optional[float] = None,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    stop_on_no_improvement: bool = True,
    require_improvement: bool = True,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> dict:
    """Run the full self-refinement loop."""

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    steps = []
    
    initial_eval = evaluation_func(initial_text)
    current_text = initial_text
    current_eval = initial_eval
    current_score = get_score(current_eval, score_key)
    initial_score = current_score
    
    iteration = 1
    last_improved = True
    
    while should_continue(
        iteration=iteration,
        max_iterations=max_iterations,
        current_score=current_score,
        target_score=target_score,
        last_improved=last_improved,
        stop_on_no_improvement=stop_on_no_improvement,
    ):
        step = refine_once(
            original_prompt=original_prompt,
            current_text=current_text,
            generation_func=generation_func,
            evaluation_func=evaluation_func,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            score_key=score_key,
            focus=focus,
            preserve=preserve,
            **kwargs,
        )
        step.iteration = iteration
        
        new_text, new_eval, accepted = choose_better_text(
            current_text=current_text,
            revised_text=step.revised_text,
            current_evaluation=current_eval,
            revised_evaluation=step.evaluation,
            score_key=score_key,
            require_improvement=require_improvement,
        )
        step.accepted = accepted
        
        if accepted:
            current_text = new_text
            current_eval = new_eval
            current_score = get_score(current_eval, score_key)
            last_improved = True
        else:
            last_improved = False
            
        steps.append(step)
        iteration += 1

    if target_score is not None and current_score >= target_score:
        stopped_reason = "target_score"
    elif stop_on_no_improvement and not last_improved:
        stopped_reason = "no_improvement"
    else:
        stopped_reason = "max_iterations"

    return {
        "initial_text": initial_text,
        "initial_score": initial_score,
        "final_text": current_text,
        "final_score": current_score,
        "iterations": steps,
        "num_iterations": len(steps),
        "steps": steps,
        "stopped_reason": stopped_reason,
    }


def summarize_refinement(result: dict) -> str:
    """Create a readable summary of a refinement run."""

    lines = []
    lines.append(f"Initial score: {result.get('initial_score', 0.0):.3f}")
    lines.append(f"Final score:   {result.get('final_score', 0.0):.3f}")
    lines.append(f"Iterations:    {len(result.get('iterations', []))}")
    lines.append(f"Stopped:       {result.get('stopped_reason', 'unknown')}")
    lines.append("")
    lines.append("Iteration details:")
    
    steps = result.get("iterations", result.get("steps", []))
    if not steps:
        lines.append("  (no refinement steps taken)")
    for step in steps:
        status = "ACCEPTED" if step.accepted else "REJECTED"
        lines.append(
            f"  Iter {step.iteration}: {step.previous_score:.3f} -> {step.revised_score:.3f} [{status}]"
        )
        
    return "\n".join(lines)
