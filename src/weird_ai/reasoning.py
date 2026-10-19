"""
Reasoning prompt helpers for Weird AI.

This module introduces the first lightweight reasoning-related utilities for
Weird AI. These functions do not train a reasoning model yet. Instead, they
help students explore how prompt structure can encourage a model to produce
intermediate reasoning steps before generating a final answer.

Lesson 8 connects the Weird AI parody assistant to the reasoning model portion
of the course by focusing on inference-time reasoning prompts.
"""


def build_reasoning_prompt(user_prompt):
    """
    Convert a normal user prompt into a general reasoning prompt.

    The returned prompt should encourage the model to think through the task
    before giving its final answer.
    """

    return (
        "You are Weird AI, a thoughtful parody lyric assistant.\n\n"
        f"Task:\n{user_prompt}\n\n"
        "Think step-by-step. Break the task into chunks. Consider the constraints,"
        " plan your approach, and then answer.\n\n"
        "Final Answer:"
    )


def build_parody_reasoning_prompt(topic):
    """
    Build a reasoning prompt specifically for parody lyric generation.

    This prompt should guide Weird AI through a simple creative planning process
    before asking it to write the final parody lyrics.
    """

    return (
        "You are Weird AI, a parody lyric assistant.\n\n"
        f"Topic: {topic}\n\n"
        "Think step-by-step before writing the final parody lyrics:\n"
        "1. Identify the topic of the parody.\n"
        f"2. Identify funny ideas, exaggerations, or contrasts that fit {topic}.\n"
        "3. Identify possible rhyme words and a simple rhyme scheme.\n"
        "4. Write the final parody lyrics using the ideas and rhymes above.\n\n"
        "Final Parody:"
    )

def build_closed_world_prompt(premises, question):
    """
    Build a prompt for closed-world reasoning.

    Closed-world reasoning means the model should answer using only the facts
    explicitly provided in the prompt. It should not use outside knowledge.
    """

    premise_lines = "\n".join(f"- {p}" for p in premises)

    return (
        "You are Weird AI. Answer using only the provided premises below. "
        "Do not use outside knowledge or background facts. If a fact is not "
        "stated in the provided premises, do not use it.\n\n"
        f"Premises:\n{premise_lines}\n\n"
        f"Question: {question}\n\n"
        "Reasoning:\n"
        "Answer:"
    )


def build_open_world_prompt(premises, question):
    """
    Build a prompt for open-world reasoning.

    Open-world reasoning allows the model to consider both the provided premises
    and relevant background knowledge. This can reveal contradictions between
    the prompt and the model's background knowledge.
    """

    premise_lines = "\n".join(f"- {p}" for p in premises)

    return (
        "You are Weird AI. Answer using both the premises below and any relevant "
        "background knowledge you have. Carefully check whether the premises "
        "conflict with well-known facts, and point out any contradictions you find.\n\n"
        f"Premises:\n{premise_lines}\n\n"
        f"Question: {question}\n\n"
        "Reasoning:\n"
        "Answer:"
    )


