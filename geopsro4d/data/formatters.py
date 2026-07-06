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
        "Observation: describe relevant entities, spatial relations, views, and visible states.\n"
        "Transition: describe changes across frames/images, including motion, relation changes, occlusion, or viewpoint changes.\n"
        "Derivation: reason from the observation and transition to the answer.\n"
        "Answer: provide the final answer only.\n\n"
        f"Question: {sample.question}{choices_block}\n"
    )
