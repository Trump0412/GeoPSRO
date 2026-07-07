from __future__ import annotations

from geopsro4d.data.schema import GeoSample


def choices_text(choices: tuple[str, ...] | list[str] | None) -> str:
    if not choices:
        return ""
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rows = []
    for idx, choice in enumerate(choices):
        prefix = labels[idx] if idx < len(labels) else str(idx)
        rows.append(f"{prefix}. {choice}")
    return "\n".join(rows)


def sft_prompt(sample: GeoSample) -> str:
    choices = choices_text(sample.choices)
    suffix = f"\nChoices:\n{choices}" if choices else ""
    return f"Question: {sample.question}{suffix}\nAnswer:"


def psro_prompt(sample: GeoSample) -> str:
    choices = choices_text(sample.choices)
    choices_block = f"\nChoices:\n{choices}" if choices else ""
    return (
        "You should solve the problem using the following format:\n\n"
        "<think>\n"
        "Spatial Observation: write one concise sentence describing the relevant visual-spatial evidence.\n"
        "Spatial Transition: write one concise sentence describing the key spatial change, state continuity, or multi-frame relation.\n"
        "Answer Derivation: write one concise sentence explaining how the previous two parts determine the final answer.\n"
        "</think>\n"
        "<answer>\n"
        "Write only the final answer. For multiple-choice questions, write only the option letter.\n"
        "</answer>\n\n"
        f"Question: {sample.question}{choices_block}\n"
    )
