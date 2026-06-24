from scenarios.prompts import get_prompt
from scenarios.scenario_cards import get_scenario


END_CALL_PHRASES = (
    "goodbye",
    "thank you, bye",
    "thanks, bye",
    "take care, bye",
    "have a good day, bye",
    "okay, great. thank you. bye",
)


def get_system_prompt(scenario_id: int) -> str:
    return get_prompt(scenario_id)


def should_end_call(response_text: str) -> bool:
    normalized = " ".join(response_text.lower().split())
    return any(phrase in normalized for phrase in END_CALL_PHRASES)


def get_opening_line(scenario_id: int) -> str:
    scenario = get_scenario(scenario_id)
    if scenario is None:
        raise ValueError(f"Unknown scenario_id: {scenario_id}")
    return scenario.opening_line

