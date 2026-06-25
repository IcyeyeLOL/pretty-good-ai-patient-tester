"""Tests for streaming TTS safety, latency instrumentation, and appointment memory."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.appointment_ledger import (
    MEMORY_CHALLENGE_LINE,
    claim_introduces_unconfirmed_fact,
    confirmed_facts,
    find_clock_times,
    find_weekdays,
    missing_required_confirmations,
    patient_should_challenge,
)
from src.pipeline import (
    LatencyTracker,
    StreamingResponseFilter,
    _fallback_patient_response,
    patient_response_decision,
)

# ── Streaming guard: safe text streams before the response ends ──


def test_streaming_filter_releases_sentence_before_finish():
    f = StreamingResponseFilter(scenario_id=1)
    # Mid-sentence chunk: nothing is released yet.
    assert f.push("Hi there, ") == ""
    # Completing the sentence releases it immediately — before finish() is called.
    released = f.push("I need an appointment. ")
    assert "I need an appointment." in released
    assert f.emitted_any is True
    trailing, fallback = f.finish()
    assert fallback is None


def test_streaming_filter_streams_multiple_sentences_incrementally():
    f = StreamingResponseFilter(scenario_id=1)
    first = f.push("First sentence. ")
    assert first.strip() == "First sentence."
    # The second sentence is delivered on a later chunk, not held to the end.
    second = f.push("Second one now? ")
    assert second.strip() == "Second one now?"


def test_streaming_filter_role_break_falls_back_without_leaking():
    f = StreamingResponseFilter(scenario_id=2)
    # Role-break phrase split across chunks; neither chunk may be spoken.
    assert f.push("How can I ") == ""
    assert f.push("help you today?") == ""
    assert f.unsafe is True
    trailing, fallback = f.finish()
    assert trailing == ""
    assert fallback == _fallback_patient_response(2)
    # Nothing safe was ever emitted, so the unsafe words never leaked.
    assert f.emitted_any is False


def test_streaming_filter_no_fallback_after_safe_text_already_spoken():
    f = StreamingResponseFilter(scenario_id=1)
    spoken = f.push("Sure, that works for me. ")
    assert spoken.strip() == "Sure, that works for me."
    # Now a role break appears late — we stop, but do NOT inject a fallback after
    # already speaking safe text (which would double up awkwardly).
    f.push("As an AI I must stop.")
    assert f.unsafe is True
    trailing, fallback = f.finish()
    assert fallback is None


def test_streaming_filter_strips_markdown_and_emoji_while_streaming():
    f = StreamingResponseFilter(scenario_id=1)
    out = f.push("I am a **new patient** here. ")
    assert out.strip() == "I am a new patient here."
    out2 = f.push("Thanks so much! 😊 ")
    assert "😊" not in out2


# ── Turn ownership (one response per agent turn / barge drops) ──


def _decide(**kw):
    base = dict(
        consecutive_patient=False,
        response_barge_generation=0,
        current_barge_generation=0,
        response_agent_generation=1,
        current_agent_generation=1,
        answered_agent_generation=-1,
    )
    base.update(kw)
    return patient_response_decision(**base)


def test_second_response_for_same_agent_turn_is_dropped():
    # Turn 1 already answered -> a second response for agent turn 1 drops.
    assert _decide(response_agent_generation=1, answered_agent_generation=1) == "drop_duplicate"


def test_response_dropped_when_barge_in_advances_during_generation():
    assert _decide(response_barge_generation=1, current_barge_generation=2) == "drop_barged"


# ── Latency instrumentation artifact ──


def test_latency_tracker_records_all_checkpoints(isolated_cwd):
    tracker = LatencyTracker(scenario_id=1, call_sid="CALLLAT", started_at=time.monotonic())
    for checkpoint in LatencyTracker.CHECKPOINTS:
        tracker.mark(checkpoint)
        time.sleep(0.001)
    tracker.finalize()

    path = Path("runs") / "scenario_01_CALLLAT" / "latency_debug.json"
    assert path.exists(), "latency_debug.json should be written on finalize"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["call_sid"] == "CALLLAT"
    assert len(data["turns"]) == 1
    record = data["turns"][0]
    for checkpoint in LatencyTracker.CHECKPOINTS:
        assert checkpoint in record, f"missing latency checkpoint: {checkpoint}"
    # Derived perceived-latency field is present.
    assert "stt_to_first_audio_ms" in record


def test_latency_tracker_records_call_level_events(isolated_cwd):
    tracker = LatencyTracker(scenario_id=1, call_sid="CALLEVT", started_at=time.monotonic())
    tracker.event("opener_scheduled_at")
    time.sleep(0.002)
    tracker.event("first_inbound_audio")
    # First write wins — a later duplicate event does not move the timestamp.
    first = tracker.events["opener_scheduled_at"]
    tracker.event("opener_scheduled_at")
    assert tracker.events["opener_scheduled_at"] == first

    data = json.loads(
        (Path("runs") / "scenario_01_CALLEVT" / "latency_debug.json").read_text(encoding="utf-8")
    )
    assert "opener_scheduled_at" in data["events"]
    assert "first_inbound_audio" in data["events"]


def test_latency_tracker_mark_is_first_write_wins():
    tracker = LatencyTracker(scenario_id=1, call_sid="X", started_at=time.monotonic())
    tracker.mark("first_llm_token")
    first = tracker._current["first_llm_token"]
    time.sleep(0.005)
    tracker.mark("first_llm_token")
    assert tracker._current["first_llm_token"] == first


def test_latency_tracker_waits_for_text_log_and_first_audio_before_auto_finalize(isolated_cwd):
    tracker = LatencyTracker(scenario_id=1, call_sid="CALLWAIT", started_at=time.monotonic())
    tracker.mark("patient_turn_logged")
    tracker.maybe_finalize()
    assert tracker.turns == []

    tracker.mark("first_audio_out")
    tracker.maybe_finalize()

    assert len(tracker.turns) == 1
    assert (Path("runs") / "scenario_01_CALLWAIT" / "latency_debug.json").exists()


def test_latency_tracker_ignores_trailing_audio_after_finalize():
    tracker = LatencyTracker(scenario_id=1, call_sid="CALLTRAIL", started_at=time.monotonic())
    tracker.mark("stt_committed")
    tracker.mark("patient_turn_logged")
    tracker.mark_first_audio_out()
    tracker.maybe_finalize()

    tracker.mark_first_audio_out()

    assert len(tracker.turns) == 1
    assert tracker._current == {}


# ── Appointment memory ledger ──

_PRIOR_VAGUE = [
    {"speaker": "patient", "text": "Mornings work best for me, any day next week is fine."},
    {"speaker": "agent", "text": "What days are you available?"},
]


def test_vague_preference_is_not_a_confirmed_date_or_time():
    facts = confirmed_facts(_PRIOR_VAGUE)
    assert facts["weekdays"] == set()
    assert facts["times"] == set()


def test_scenario1_patient_must_not_accept_fabricated_friday_12pm():
    # Nothing in the prior conversation established a weekday or clock time, yet the
    # agent asserts "Friday 12PM". The patient must challenge, not accept.
    assert patient_should_challenge(_PRIOR_VAGUE, "No. I said Friday 12PM, I'm pretty sure.") is True
    assert "confirm the exact day and time" in MEMORY_CHALLENGE_LINE.lower()


def test_claim_not_flagged_when_day_and_time_were_already_confirmed():
    prior = _PRIOR_VAGUE + [
        {"speaker": "agent", "text": "Okay, I can do Friday at 12pm."},
        {"speaker": "patient", "text": "Friday at noon works for me."},
    ]
    assert claim_introduces_unconfirmed_fact(prior, "Friday 12PM") is False


def test_missing_required_confirmations_for_scenario1_wrapup():
    assert missing_required_confirmations(_PRIOR_VAGUE) == ["weekday", "time"]
    settled = _PRIOR_VAGUE + [{"speaker": "agent", "text": "You're booked Tuesday at 9am."}]
    assert missing_required_confirmations(settled) == []


def test_find_weekdays_and_times_basic():
    assert find_weekdays("see you Friday and maybe Tuesday") == {"friday", "tuesday"}
    assert "12:00pm" in find_clock_times("how about 12pm")
    assert "noon" in find_clock_times("let's say noon")
