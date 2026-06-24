from __future__ import annotations

import pytest

from src.pipeline import ROLE_BREAK_RE, _initial_messages, _remove_emoji, _sanitize_voice_text


@pytest.mark.parametrize(
    "text",
    [
        "I am an AI assistant here to help.",
        "How can I help you today?",
        "As an AI, I cannot schedule that.",
        "I'm not a patient.",
        "The message got cut off.",
        "I am playing the role of Maria.",
    ],
)
def test_role_break_regex_matches_dangerous_phrases(text: str):
    assert ROLE_BREAK_RE.search(text)


@pytest.mark.parametrize(
    "text",
    [
        "I'm calling to make an appointment.",
        "Could you repeat that?",
        "My name is Maria Johnson.",
        "I need to reschedule my appointment.",
    ],
)
def test_role_break_regex_allows_normal_patient_speech(text: str):
    assert not ROLE_BREAK_RE.search(text)


def test_returns_fallback_on_role_break():
    text = "I am NOT a patient - I am an AI assistant here to help you."

    assert _sanitize_voice_text(text, scenario_id=1) == (
        "Sorry, I'm not sure I follow. I'm Maria, and I'm calling to schedule as a new patient."
    )


def test_returns_fallback_on_help_you_today():
    text = "Hello! How can I help you today?"

    assert _sanitize_voice_text(text, scenario_id=2) == (
        "Sorry, I'm not sure I follow. I'm just calling the office about this."
    )


def test_strips_bold_markdown():
    assert _sanitize_voice_text("I am a **new patient** looking to schedule.", 1) == (
        "I am a new patient looking to schedule."
    )


def test_strips_emoji():
    assert _sanitize_voice_text("Hi there 😊 I need an appointment ✅", 1) == (
        "Hi there I need an appointment"
    )


def test_strips_stage_directions_asterisk_form():
    assert _sanitize_voice_text("*rings phone* Hi, I need an appointment.", 1) == (
        "Hi, I need an appointment."
    )


def test_remove_emoji_preserves_ascii():
    assert _remove_emoji("Hello there! 123 😊🎉✅") == "Hello there! 123 "


def test_initial_messages_seed_system_and_opening_line():
    messages = _initial_messages("SYSTEM PROMPT", "Hello, I need an appointment.")

    assert messages[0] == {"role": "system", "content": "SYSTEM PROMPT"}
    assert messages[1]["role"] == "user"
    assert messages[2] == {"role": "assistant", "content": "Hello, I need an appointment."}
