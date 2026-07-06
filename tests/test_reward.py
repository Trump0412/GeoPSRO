from __future__ import annotations

from geopsro4d.reward.answer_parser import parse_answer
from geopsro4d.reward.psro_parser import parse_psro
from geopsro4d.reward.reward_fn import compute_reward


def test_correct_answer_with_valid_psro() -> None:
    text = (
        "Observation: the cup is left of the box.\n"
        "Transition: across frames, the cup moves closer.\n"
        "Derivation: the observation and transition imply option B.\n"
        "Answer: B"
    )
    reward = compute_reward(text, "B", ["A text", "B text"], "dynamic")
    assert reward["r_acc"] == 1.0
    assert reward["r_psro_fmt"] == 1.0
    assert reward["reward"] > 1.0


def test_wrong_answer_with_valid_psro_gets_no_process_bonus() -> None:
    text = (
        "Observation: the cup is left of the box.\n"
        "Transition: across frames, the cup moves closer.\n"
        "Derivation: the observation and transition imply option B.\n"
        "Answer: B"
    )
    reward = compute_reward(text, "A", ["A text", "B text"], "dynamic")
    assert reward["r_acc"] == 0.0
    assert reward["reward"] < 0.5


def test_multiple_conflicting_answers_returns_none() -> None:
    assert parse_answer("Answer: A or B", ["x", "y"]) is None


def test_lowercase_sections_and_empty_transition() -> None:
    parsed = parse_psro("observation: objects are visible\ntransition:\nderivation: so\nanswer: a")
    assert parsed["present"]["transition"]
    assert parsed["empty"]["transition"]
