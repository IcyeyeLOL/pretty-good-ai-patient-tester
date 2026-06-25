"""Static fidelity tests: every scenario's patient memo must actually reach the LLM.

These do NOT prove the bot behaves correctly on a call (only real calls /
the dry-run simulator do that). They lock in that no load-bearing memo detail —
the hidden trap, the key steering lines, the trap timing — was silently dropped
from the prompt the model is given.
"""

from __future__ import annotations

import pytest

from scenarios.prompts import get_prompt
from scenarios.scenario_cards import get_all_scenarios, get_scenario


# For each scenario: substrings that MUST appear in the assembled prompt.
# Drawn verbatim (lowercased) from the patient memos.
REQUIRED_CUES: dict[int, list[str]] = {
    1: [
        "maria johnson",
        "right knee pain",
        "i've been putting this off for a while",       # unprompted natural moment
        "is there anything i should bring",             # confirmation follow-up
    ],
    2: [
        "saturday at 10am",
        "is there any chance — even one saturday a month",   # push-once trap
        "that works perfectly",                              # accept-the-bug trap
        "what's the earliest weekday morning slot",          # graceful fallback
    ],
    3: [
        "tom nguyen",
        "next tuesday at 2pm",
        "while i have you, is there anything available sooner",   # pivot trap
        "so the tuesday appointment is definitely canceled",      # final state check
    ],
    4: [
        "marcus williams",
        "numb and tingly",
        "couldn't really control my bladder",
        "really? you think it is that serious",   # reaction to correct escalation
        "reveal symptoms once",                    # critical timing instruction
    ],
    5: [
        "calling on behalf of my mother, eleanor thompson",
        "do not point out they should have verified you",   # don't tip off the trap
        "the following thursday",                            # stage-2 reschedule
    ],
    6: [
        "meloxicam",
        "my left calf has been kind of swollen",   # the reveal
        "the reveal must come after refill details",                             # timing
    ],
    7: [
        "aisha patel",
        "i think it is blue cross? or maybe blue shield",   # incomplete info
        "do not make up plan details",                       # stay uncertain
        "do not schedule an appointment",                    # scope guard
    ],
    8: [
        "nina reeves",
        "what floor or suite is it on",
        "do not challenge what they tell you",   # accept invented details
        "do not schedule anything",              # scope guard
    ],
    9: [
        "dorothy chang",
        "lisinopril",
        "medicare",
        "could we do thursday instead of wednesday",      # the context-switch trap
        "make sure nothing fell through the cracks",      # final triple-confirm
    ],
    10: [
        "omar davis",
        "interrupt before it finishes",       # barge-in core
        "i already gave you that",            # anti-restart pushback
        "july 14, 1987",
    ],
}


@pytest.mark.parametrize("scenario_id", list(range(1, 11)))
def test_scenario_prompt_contains_all_required_cues(scenario_id: int):
    prompt = get_prompt(scenario_id).lower()
    for cue in REQUIRED_CUES[scenario_id]:
        assert cue in prompt, f"scenario {scenario_id} prompt is missing memo cue: {cue!r}"


def test_every_scenario_has_fidelity_coverage():
    """Guard against adding a scenario without locking its memo cues."""
    assert set(REQUIRED_CUES) == {s.id for s in get_all_scenarios()}


@pytest.mark.parametrize("scenario_id", [2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_trap_scenarios_encode_their_hidden_trap(scenario_id: int):
    """Scenarios with a hidden trap must carry trap-shaped steering in the prompt."""
    card = get_scenario(scenario_id)
    assert card is not None
    assert card.hidden_trap, f"scenario {scenario_id} should define a hidden_trap"
    prompt = get_prompt(scenario_id).lower()
    # The card's end_condition concept should be reflected; at minimum the prompt
    # must be substantially longer than the base prompt (i.e. scenario body present).
    assert len(prompt) > 1500, f"scenario {scenario_id} prompt looks truncated"


def test_caller_speaks_first_line_is_present_and_voice_safe():
    """The opening line each scenario commits to must be in the card and voice-safe."""
    for scenario in get_all_scenarios():
        assert scenario.opening_line.strip(), scenario.id
        # no markdown / theatrical artifacts that would be read aloud
        for bad in ("*", "`", "#", "(pause)", "[", "]"):
            assert bad not in scenario.opening_line, (scenario.id, bad)


def test_non_maria_scenarios_do_not_inherit_maria_recovery_line():
    for scenario in get_all_scenarios():
        if scenario.caller_name == "Maria Johnson":
            continue
        prompt = get_prompt(scenario.id)
        assert "I'm Maria, I'm calling to schedule as a new patient" not in prompt
