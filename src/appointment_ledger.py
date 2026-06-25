"""Lightweight appointment-fact ledger.

The patient bot is an LLM, so the live guard against accepting fabricated
appointment details is the system prompt (see scenarios/prompts.py). This module
is the *checkable* counterpart: pure functions that read a transcript and decide
whether a day/time was ever actually established, and whether an agent's claim
introduces a fact that was never confirmed. It is used by tests and can be used
post-call to flag a patient turn that accepted an invented date or time.
"""

from __future__ import annotations

import re

WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

_WEEKDAY_RE = re.compile(r"\b(" + "|".join(WEEKDAYS) + r")\b", re.IGNORECASE)
# Clock times: "12pm", "12 pm", "2:30 pm", "10am", or the word "noon"/"midnight".
_CLOCK_RE = re.compile(
    r"\b(?:(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)|(noon|midnight))\b",
    re.IGNORECASE,
)

# The canonical polite line a patient should use when the agent asserts an
# appointment fact that was never actually confirmed in the call.
MEMORY_CHALLENGE_LINE = (
    "Sorry, I don't think we actually confirmed that yet. "
    "Could you confirm the exact day and time?"
)


def find_weekdays(text: str) -> set[str]:
    return {m.lower() for m in _WEEKDAY_RE.findall(text or "")}


def find_clock_times(text: str) -> set[str]:
    times: set[str] = set()
    for hour, minute, meridiem, word in _CLOCK_RE.findall(text or ""):
        if word:
            times.add(word.lower())
            continue
        mer = meridiem.lower().replace(".", "")
        minute = minute or "00"
        times.add(f"{int(hour)}:{minute}{mer}")
    return times


def _turn_text(turns: list[dict]) -> str:
    return " ".join(turn.get("text", "") for turn in turns)


def confirmed_facts(turns: list[dict]) -> dict[str, set[str]]:
    """Days and clock times that were actually spoken anywhere in the call.

    A vague preference like "mornings next week" yields no weekday and no clock
    time, so it does not count as a confirmed date/time.
    """
    text = _turn_text(turns)
    return {
        "weekdays": find_weekdays(text),
        "times": find_clock_times(text),
    }


def claim_introduces_unconfirmed_fact(prior_turns: list[dict], claim_text: str) -> bool:
    """True if `claim_text` asserts a weekday or clock time not present earlier.

    This is the fabricated-appointment-fact check: the agent says "Friday 12PM"
    but neither "Friday" nor "12pm" ever appeared in the prior conversation.
    """
    prior = confirmed_facts(prior_turns)
    claim_days = find_weekdays(claim_text)
    claim_times = find_clock_times(claim_text)
    new_day = bool(claim_days - prior["weekdays"])
    new_time = bool(claim_times - prior["times"])
    return new_day or new_time


def patient_should_challenge(prior_turns: list[dict], claim_text: str) -> bool:
    """Whether the patient should push back instead of accepting the claim."""
    return claim_introduces_unconfirmed_fact(prior_turns, claim_text)


def missing_required_confirmations(
    turns: list[dict],
    required: tuple[str, ...] = ("weekday", "time"),
) -> list[str]:
    """Which required appointment facts were never established in the call.

    Used for scenario 1: if the agent tries to wrap up without a concrete day and
    time, the patient is expected to ask before finishing.
    """
    facts = confirmed_facts(turns)
    missing = []
    if "weekday" in required and not facts["weekdays"]:
        missing.append("weekday")
    if "time" in required and not facts["times"]:
        missing.append("time")
    return missing
