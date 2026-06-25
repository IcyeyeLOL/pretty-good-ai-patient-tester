from __future__ import annotations

import re

from scenarios.prompts import BASE_PROMPT, SCENARIO_PROMPTS, get_prompt
from scenarios.scenario_cards import get_all_scenarios, get_scenario


def test_base_prompt_is_injected_into_every_prompt():
    base_start = BASE_PROMPT.strip().splitlines()[0]

    for scenario in get_all_scenarios():
        prompt = get_prompt(scenario.id)
        assert prompt.startswith(base_start)
        assert scenario.caller_name in prompt


def test_all_scenario_prompts_exist():
    assert set(SCENARIO_PROMPTS) == set(range(1, 11))


def test_scenario_four_contains_urgent_symptoms():
    prompt = get_prompt(4).lower()

    assert "back" in prompt
    assert "numb" in prompt
    assert "bladder" in prompt
    assert "emergency" in prompt or "911" in prompt or "er" in prompt


def test_scenario_six_contains_medication_side_effects():
    prompt = get_prompt(6).lower()

    assert "meloxicam" in prompt
    assert "calf" in prompt
    assert "swollen" in prompt


def test_base_prompt_hardens_voice_format_and_character_rules():
    base = BASE_PROMPT.lower()

    assert "never use markdown" in base
    assert "never use emojis" in base
    assert "do not say you are an ai" in base or "you are not an ai assistant" in base
    assert "maximum 2 sentences" in base
    assert "stay in character" in base
    assert "answer that direct question before adding any voluntary scenario detail" in base
    assert "say each voluntary scenario detail at most once" in base


def test_get_prompt_unknown_scenario_returns_base_prompt_only():
    prompt = get_prompt(99)

    assert prompt == BASE_PROMPT.strip()
    assert "Maria Johnson" not in prompt


def test_opening_lines_are_voice_safe():
    unsafe = re.compile(r"[*_`#>\U0001f300-\U0001faff]")

    for scenario in get_all_scenarios():
        assert not unsafe.search(scenario.opening_line), scenario.opening_line


def test_prompt_has_expected_caller_for_each_scenario():
    for scenario in get_all_scenarios():
        card = get_scenario(scenario.id)
        assert card is not None
        assert card.caller_name in get_prompt(scenario.id)
