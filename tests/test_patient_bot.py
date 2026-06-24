from __future__ import annotations

import pytest

from scenarios.prompts import get_prompt
from scenarios.scenario_cards import get_all_scenarios
from src.patient_bot import get_opening_line, get_system_prompt, should_end_call


def test_get_system_prompt_matches_prompt_source():
    for scenario in get_all_scenarios():
        assert get_system_prompt(scenario.id) == get_prompt(scenario.id)


def test_get_opening_line_returns_scenario_opening():
    for scenario in get_all_scenarios():
        assert get_opening_line(scenario.id) == scenario.opening_line


def test_get_opening_line_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="Unknown scenario_id"):
        get_opening_line(99)


@pytest.mark.parametrize(
    "text",
    [
        "goodbye",
        "Okay, great. Thank you. Bye.",
        "Thanks, bye",
        "Have a good day, bye",
    ],
)
def test_should_end_call_detects_closing_phrases(text: str):
    assert should_end_call(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Hi there",
        "I would like to make an appointment",
        "Thank you for checking on that",
    ],
)
def test_should_end_call_ignores_non_closing_speech(text: str):
    assert should_end_call(text) is False
