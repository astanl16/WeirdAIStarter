"""
Reward utilities for Weird AI.

This module connects Chapter 6 of Build a Reasoning Model from Scratch to the
Weird AI project.

The book uses RLVR: reinforcement learning with verifiable rewards.

For math:
    generated answer -> deterministic verifier -> reward 0 or 1

For Weird AI:
    generated lyrics -> project evaluator -> heuristic reward between 0 and 1

This file does not train the model yet. Instead, it prepares the reward and
group-relative advantage data that a future GRPO-style training loop would need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Callable, Iterable, Optional


EvaluationFunction = Callable[[str], dict]


@dataclass
class RolloutReward:
    """
    Store reward information for one generated rollout.

    Attributes:
        prompt: The original prompt.
        text: The generated lyrics.
        evaluation: Full evaluation dictionary for the rollout.
        reward: Normalized reward value.
        advantage: Group-relative advantage.
        index: Rollout number, starting at 1.
    """

    prompt: str
    text: str
    evaluation: dict
    reward: float
    advantage: float
    index: int


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """
    Clamp a value to a numeric range.

    Args:
        value: Numeric value to clamp.
        minimum: Lowest allowed value.
        maximum: Highest allowed value.

    Returns:
        The clamped value.
    """

    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def extract_score(evaluation: dict, score_key: str = "overall_score") -> float:
    """
    Extract a numeric score from an evaluation dictionary.

    Args:
        evaluation: Evaluation dictionary, usually from weird_ai.evaluate.
        score_key: Key containing the main score.

    Returns:
        Score as a float. Missing or invalid values become 0.0.
    """

    try:
        return float(evaluation[score_key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def normalize_reward(
    score: float,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Normalize a score into the 0.0 to 1.0 reward range.

    Args:
        score: Raw score.
        minimum: Expected minimum score.
        maximum: Expected maximum score.

    Returns:
        Normalized reward in the range [0.0, 1.0].
    """

    if maximum == minimum:
        return 0.0
    normalized = (score - minimum) / (maximum - minimum)
    return clamp(normalized)


def reward_from_evaluation(
    evaluation: dict,
    *,
    score_key: str = "overall_score",
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Convert an evaluation dictionary into a reward.

    Args:
        evaluation: Evaluation dictionary.
        score_key: Key containing the main score.
        minimum: Expected minimum score.
        maximum: Expected maximum score.

    Returns:
        Normalized reward.
    """

    score = extract_score(evaluation, score_key)
    return normalize_reward(score, minimum=minimum, maximum=maximum)


def compute_group_advantages(
    rewards: Iterable[float],
    *,
    normalize: bool = False,
    epsilon: float = 1e-8,
) -> list[float]:
    """
    Compute group-relative advantages.

    In GRPO-style training, each rollout is compared with the other rollouts
    generated for the same prompt.

    Basic advantage:
        advantage = reward - mean(group_rewards)

    Optional normalized advantage:
        advantage = (reward - mean(group_rewards)) / std(group_rewards)

    Args:
        rewards: Reward values for a group of rollouts.
        normalize: Whether to divide by group standard deviation.
        epsilon: Small value to avoid division by zero.

    Returns:
        List of advantages in the same order as rewards.
    """

    rewards = list(rewards)
    if not rewards:
        return []
    group_mean = mean(rewards)
    advantages = [reward - group_mean for reward in rewards]
    if normalize:
        group_std = pstdev(rewards)
        advantages = [advantage / (group_std + epsilon) for advantage in advantages]
    return advantages


def build_rollout_reward(
    prompt: str,
    text: str,
    evaluation: dict,
    reward: float,
    advantage: float,
    index: int,
) -> RolloutReward:
    """
    Build a RolloutReward object.

    Args:
        prompt: Original prompt.
        text: Generated lyrics.
        evaluation: Evaluation dictionary.
        reward: Reward value.
        advantage: Group-relative advantage.
        index: Rollout number.

    Returns:
        RolloutReward instance.
    """

    return RolloutReward(
        prompt=prompt,
        text=text,
        evaluation=evaluation,
        reward=reward,
        advantage=advantage,
        index=index,
    )


def prepare_reward_batch(
    prompt: str,
    rollouts: list[str],
    evaluation_func: EvaluationFunction,
    *,
    score_key: str = "overall_score",
    reward_minimum: float = 0.0,
    reward_maximum: float = 1.0,
    normalize_advantages: bool = False,
) -> list[RolloutReward]:
    """
    Evaluate rollouts, compute rewards, and compute group-relative advantages.

    This is the main assignment function. It prepares the information a future
    GRPO-style training loop would need.

    Args:
        prompt: Original prompt.
        rollouts: Generated lyric candidates for the same prompt.
        evaluation_func: Function that evaluates one rollout.
        score_key: Key containing the main score in the evaluation dictionary.
        reward_minimum: Expected minimum evaluation score.
        reward_maximum: Expected maximum evaluation score.
        normalize_advantages: Whether to standardize advantages.

    Returns:
        List of RolloutReward objects.
    """

    if not rollouts:
        return []
    evaluations = [evaluation_func(rollout) for rollout in rollouts]
    rewards = [
        reward_from_evaluation(
            evaluation,
            score_key=score_key,
            minimum=reward_minimum,
            maximum=reward_maximum,
        )
        for evaluation in evaluations
    ]
    advantages = compute_group_advantages(rewards, normalize=normalize_advantages)
    return [
        build_rollout_reward(
            prompt,
            rollouts[i],
            evaluations[i],
            rewards[i],
            advantages[i],
            i + 1,
        )
        for i in range(len(rollouts))
    ]


def rank_by_reward(batch: list[RolloutReward], *, descending: bool = True) -> list[RolloutReward]:
    """
    Sort rollout records by reward.

    Args:
        batch: RolloutReward objects.
        descending: If True, highest reward first.

    Returns:
        Sorted RolloutReward list.
    """

    return sorted(batch, key=lambda rollout: rollout.reward, reverse=descending)


def get_best_rollout(batch: list[RolloutReward]) -> Optional[RolloutReward]:
    """
    Return the highest-reward rollout.

    Args:
        batch: RolloutReward objects.

    Returns:
        Best rollout, or None if batch is empty.
    """

    return max(batch, key=lambda rollout: rollout.reward, default=None)


def get_worst_rollout(batch: list[RolloutReward]) -> Optional[RolloutReward]:
    """
    Return the lowest-reward rollout.

    Args:
        batch: RolloutReward objects.

    Returns:
        Worst rollout, or None if batch is empty.
    """

    return min(batch, key=lambda rollout: rollout.reward, default=None)


def summarize_reward_batch(batch: list[RolloutReward]) -> str:
    """
    Create a readable summary of a reward batch.

    Args:
        batch: RolloutReward objects.

    Returns:
        Multi-line summary string.
    """

    if not batch:
        return "No rollouts"

    best = get_best_rollout(batch)
    worst = get_worst_rollout(batch)
    lines = [
        f"Rollouts: {len(batch)}",
        f"Average reward: {mean(rollout.reward for rollout in batch):.3f}",
        f"Best: rollout {best.index} (reward {best.reward:.3f})",
        f"Worst: rollout {worst.index} (reward {worst.reward:.3f})",
    ]
    for rollout in batch:
        lines.append(
            f"Rollout {rollout.index}: reward={rollout.reward:.3f}, "
            f"advantage={rollout.advantage:.3f}"
        )
    return "\n".join(lines) 