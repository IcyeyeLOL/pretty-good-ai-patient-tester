from __future__ import annotations

import pytest

from src.pipeline import (
    DEFAULT_OPENING_DELAY_SECONDS,
    ROLE_BREAK_RE,
    _env_float,
    _env_int,
    _initial_messages,
    _is_agent_fragment,
    _join_transcript_parts,
    _remove_emoji,
    _sanitize_voice_text,
    _is_greeting_filler,
    greeting_fastpath_action,
    patient_response_decision,
    should_send_opening,
)


def _decide(**overrides):
    """patient_response_decision with sensible 'fresh first response' defaults."""
    kwargs = dict(
        consecutive_patient=False,
        response_barge_generation=0,
        current_barge_generation=0,
        response_agent_generation=1,
        current_agent_generation=1,
        answered_agent_generation=-1,
    )
    kwargs.update(overrides)
    return patient_response_decision(**kwargs)


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
        "Sorry, I'm not sure I follow. Hi, I'd like to schedule an appointment. "
        "I'm a new patient looking to establish care."
    )


def test_returns_fallback_on_help_you_today():
    text = "Hello! How can I help you today?"

    assert _sanitize_voice_text(text, scenario_id=2) == (
        "Sorry, I'm not sure I follow. Hi, I need to make an appointment. "
        "Can I come in this Saturday around 10 in the morning?"
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


def test_initial_messages_seed_system_only_not_opening_line():
    # The opening line must NOT be pre-seeded into history; it is only added if it
    # is actually spoken. Otherwise an agent-first call would falsely show the
    # patient as having already spoken its opening.
    messages = _initial_messages("SYSTEM PROMPT")

    assert messages[0] == {"role": "system", "content": "SYSTEM PROMPT"}
    assert messages[1]["role"] == "user"
    assert all(m["role"] != "assistant" for m in messages)
    assert not any("appointment" in m["content"].lower() for m in messages[1:])


@pytest.mark.parametrize(
    "text",
    [
        "Okay. What about",
        "well,",
        "But",
        "I",
        "I can take you",
        "around 9PM for",
        "All of",
    ],
)
def test_agent_fragment_detector_holds_incomplete_transcripts(text: str):
    assert _is_agent_fragment(text)


@pytest.mark.parametrize(
    "text",
    [
        "What is your name?",
        "Tuesday, 9 PM?",
        "Yes.",
        "No.",
        "Thursday at 10 AM works.",
    ],
)
def test_agent_fragment_detector_allows_complete_transcripts(text: str):
    assert not _is_agent_fragment(text)


@pytest.mark.parametrize(
    "text",
    [
        # Complete questions that end in a preposition must NOT be dropped.
        "Where are you from?",
        "What are you calling about?",
        "Who is this for?",
        "What is this regarding?",
    ],
)
def test_complete_questions_ending_in_preposition_are_kept(text: str):
    assert not _is_agent_fragment(text)


@pytest.mark.parametrize(
    "text",
    [
        # Short but meaningful appointment facts must be kept.
        "eleven AM.",
        "11 AM.",
        "Thursday.",
        "Tuesday.",
        "Sure.",
        "Okay.",
        "Noon.",
        "9 PM.",
    ],
)
def test_short_appointment_facts_are_kept(text: str):
    assert not _is_agent_fragment(text)


@pytest.mark.parametrize(
    "text",
    [
        # Obvious fragments still drop, even the one that contains a time.
        "And",
        "But",
        "I",
        "All of",
        "around 9 PM for",
    ],
)
def test_obvious_fragments_still_dropped(text: str):
    assert _is_agent_fragment(text)


def test_regression_dropped_where_are_you_from_then_hello():
    """Exact failure shape: 'Where are you from?' was dropped as a fragment, then a
    later 'Hello?' made the bot guess the missed turn was the name. After the fix the
    question survives and the mid-call recovery is universal (never mentions name)."""
    from src.pipeline import MIDCALL_RECOVERY_LINE

    assert _is_agent_fragment("Where are you from?") is False
    assert (
        greeting_fastpath_action(
            "Hello?", in_handshake=False, opening_delivered=True, opener_was_gated=True
        )
        == "recover_midcall"
    )
    assert "name" not in MIDCALL_RECOVERY_LINE.lower()


def test_join_transcript_parts_normalizes_spacing():
    assert _join_transcript_parts(["Okay. What about", " Tuesday, 9 PM? "]) == (
        "Okay. What about Tuesday, 9 PM?"
    )


def test_env_float_uses_minimum(monkeypatch):
    monkeypatch.setenv("OPENING_DELAY_SECONDS", "3")

    assert _env_float("OPENING_DELAY_SECONDS", DEFAULT_OPENING_DELAY_SECONDS, minimum=4.0) == 4.0


def test_env_int_falls_back_on_bad_value(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_ENDPOINTING_MS", "nope")

    assert _env_int("DEEPGRAM_ENDPOINTING_MS", 200, minimum=100) == 200


def test_default_opening_delay_is_short():
    # Startup must be snappy and adaptive, not a 6s blocking wait.
    assert DEFAULT_OPENING_DELAY_SECONDS <= 2.0


# ── Turn-taking: one patient response per committed agent turn ──

def test_fresh_first_response_is_emitted():
    assert _decide() == "emit"


def test_consecutive_patient_response_is_suppressed():
    assert _decide(consecutive_patient=True) == "suppress_consecutive"


def test_response_dropped_when_barge_in_during_generation():
    # Caller interrupted (barge generation advanced) while the LLM was generating.
    assert _decide(response_barge_generation=2, current_barge_generation=3) == "drop_barged"


def test_response_dropped_when_newer_agent_turn_arrived():
    # A new agent turn committed (generation advanced) before this reply finished.
    assert _decide(response_agent_generation=1, current_agent_generation=2) == "drop_superseded"


def test_duplicate_response_for_same_agent_turn_is_dropped():
    # We already answered agent turn 1; a second rephrase for the same turn drops.
    assert _decide(response_agent_generation=1, answered_agent_generation=1) == "drop_duplicate"


def test_one_response_per_agent_turn_then_next_turn_allowed():
    # Turn 1: first response emits and marks turn 1 answered.
    assert _decide(response_agent_generation=1, answered_agent_generation=-1) == "emit"
    # Turn 1: duplicate dropped.
    assert _decide(response_agent_generation=1, answered_agent_generation=1) == "drop_duplicate"
    # Turn 2 (agent spoke again): a fresh response is allowed.
    assert (
        _decide(response_agent_generation=2, current_agent_generation=2, answered_agent_generation=1)
        == "emit"
    )


# ── Adaptive opening ──

def test_opening_sent_when_no_one_has_spoken():
    assert should_send_opening(turns=[], agent_turn_generation=0) is True


def test_opening_skipped_when_agent_already_committed_a_turn():
    assert should_send_opening(turns=[], agent_turn_generation=1) is False


def test_opening_skipped_when_caller_audio_started_before_transcript_commit():
    assert should_send_opening(turns=[], agent_turn_generation=0, barge_in_generation=1) is False


def test_opening_skipped_when_agent_spoke_first():
    turns = [{"speaker": "agent", "text": "Hello? Hello?"}]
    assert should_send_opening(turns, agent_turn_generation=0) is False


def test_opening_not_repeated_after_patient_already_spoke():
    turns = [{"speaker": "patient", "text": "Hi, I'd like to schedule."}]
    assert should_send_opening(turns, agent_turn_generation=0) is False


class _GuardSim:
    """Minimal re-implementation of the guard's generation bookkeeping, driven by
    the same pure helpers the real guard uses, so we can assert turn-taking over a
    transcript-shaped sequence without pipecat plumbing."""

    def __init__(self):
        self.barge_gen = 0
        self.agent_gen = 0
        self.answered = -1
        self.emitted: list[int] = []

    def agent_commits_turn(self):
        self.agent_gen += 1

    def barge_in(self):
        self.barge_gen += 1

    def llm_response(self, *, start_barge=None, start_agent=None, consecutive=False):
        # Captured at LLMFullResponseStart.
        rb = self.barge_gen if start_barge is None else start_barge
        ra = self.agent_gen if start_agent is None else start_agent
        decision = patient_response_decision(
            consecutive_patient=consecutive,
            response_barge_generation=rb,
            current_barge_generation=self.barge_gen,
            response_agent_generation=ra,
            current_agent_generation=self.agent_gen,
            answered_agent_generation=self.answered,
        )
        if decision == "emit":
            self.answered = ra
            self.emitted.append(ra)
        return decision


def test_regression_no_duplicate_responses_to_repeated_hellos():
    """Transcript shape: agent says 'Hello?' three times, plus a stray fragment LLM
    completion. The bot must answer exactly once per committed agent turn."""
    sim = _GuardSim()

    # Auto opening (agent turn 0) is spoken first.
    assert sim.llm_response() == "emit"
    assert sim.emitted == [0]

    # Agent: "Hello? Hello?" -> one committed turn -> one patient reply.
    sim.agent_commits_turn()
    assert sim.llm_response() == "emit"
    # A second LLM completion for the SAME agent turn (the duplicate catch-up) drops.
    assert sim.llm_response() == "drop_duplicate"

    # Agent: "Hello?" again -> new turn -> exactly one new reply.
    sim.agent_commits_turn()
    assert sim.llm_response() == "emit"

    # Exactly one emit per agent turn (opening + 2 agent turns = 3 total).
    assert sim.emitted == [0, 1, 2]


def test_regression_stale_response_dropped_when_agent_keeps_talking():
    """If the agent commits another turn while the LLM is still generating, the now
    stale reply is dropped and the next (fresh) one is emitted instead."""
    sim = _GuardSim()
    sim.agent_commits_turn()  # agent turn 1

    # LLM starts generating for turn 1...
    start_agent = sim.agent_gen
    # ...but the agent says more before it finishes -> turn 2 commits.
    sim.agent_commits_turn()

    # The reply tagged to turn 1 is now superseded.
    assert sim.llm_response(start_agent=start_agent) == "drop_superseded"
    # The fresh reply for turn 2 is emitted.
    assert sim.llm_response() == "emit"
    assert sim.emitted == [2]


# ── Opening handshake: deterministic greeting fast path ──


def test_default_opening_delay_targets_fast_audible_opener():
    # Real calls schedule the fallback after the media stream connects, so keep
    # this short enough that the opener lands near 1.5-2s in practice.
    assert 0.3 <= DEFAULT_OPENING_DELAY_SECONDS <= 0.6


@pytest.mark.parametrize(
    "text",
    [
        "Hello?",
        "Hello? Hello?",
        "Hi?",
        "Hello? Can you hear me?",
        "Are you there?",
        "Hi there",
        "Hey, you there?",
        "Anybody there?",
    ],
)
def test_greeting_filler_detected(text):
    assert _is_greeting_filler(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "What do you need?",
        "Yes, this is the office. What can I do?",
        "Thanks for calling, how can I help?",
        "We are closed on weekends.",
        "What days are you available?",
    ],
)
def test_substantive_agent_turn_is_not_greeting_filler(text):
    assert _is_greeting_filler(text) is False


def test_fastpath_delivers_opener_on_first_greeting():
    # Callee-gated: we heard their "Hello?" first, so deliver the opener now.
    assert (
        greeting_fastpath_action("Hello?", in_handshake=True, opening_delivered=False)
        == "deliver_opener"
    )


def test_fastpath_swallows_repeat_greeting_only_when_opener_was_gated():
    # We already responded to the callee's speech (gated) -> a repeat "hello" drops.
    assert (
        greeting_fastpath_action(
            "Hello? Hello?",
            in_handshake=True,
            opening_delivered=True,
            opener_was_gated=True,
        )
        == "swallow"
    )


def test_fastpath_recovers_when_blind_opener_may_have_been_missed():
    # The opener was sent blindly (silence fallback, NOT gated) and the callee is
    # still greeting -> they likely never heard it, so re-greet instead of dead air.
    assert (
        greeting_fastpath_action(
            "Hello? Are you there?",
            in_handshake=True,
            opening_delivered=True,
            opener_was_gated=False,
        )
        == "recover_opener"
    )


def test_fastpath_forwards_substantive_first_turn_to_llm():
    assert (
        greeting_fastpath_action("What days are you available?", in_handshake=True, opening_delivered=False)
        == "forward"
    )


def test_fastpath_passthrough_substantive_after_handshake():
    # A substantive turn after the handshake flows to the LLM as normal.
    assert (
        greeting_fastpath_action(
            "What is your date of birth?", in_handshake=False, opening_delivered=True
        )
        == "passthrough"
    )


def test_fastpath_midcall_hello_uses_universal_recovery_not_name():
    # A bare "Hello?" after the handshake means the agent missed a turn. We must NOT
    # assume it was about the name; respond with a universal, content-free line.
    assert (
        greeting_fastpath_action(
            "Hello?", in_handshake=False, opening_delivered=True, opener_was_gated=True
        )
        == "recover_midcall"
    )
    from src.pipeline import MIDCALL_RECOVERY_LINE

    lowered = MIDCALL_RECOVERY_LINE.lower()
    assert "name" not in lowered
    assert "maria" not in lowered
    assert "repeat" in lowered
