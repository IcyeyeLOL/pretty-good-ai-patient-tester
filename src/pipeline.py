from __future__ import annotations

import asyncio
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.live_audio_recorder import LiveAudioRecorder
from src.patient_bot import get_opening_line, get_system_prompt, should_end_call

load_dotenv()

CARTESIA_VOICE_ID = "a0e99841-438c-4a64-b679-ae501e7d6091"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_CARTESIA_MODEL = "sonic-2"
DEFAULT_OPENING_DELAY_SECONDS = 0.5  # legacy; the opener is now VAD/silence-driven
# Dynamic opener: the patient speaks its opening line only when the line is silent
# (no fixed timer). Confirm this much continuous silence before opening; if the
# agent never speaks at all, open after the dead-line window so the call won't stall.
DEFAULT_OPENING_SILENCE_SECONDS = 0.6
DEFAULT_OPENING_DEAD_LINE_SECONDS = 8.0
DEFAULT_DEEPGRAM_ENDPOINTING_MS = 200
DEFAULT_AGENT_COMPLETE_DEBOUNCE_SECONDS = 0.2
DEFAULT_AGENT_FRAGMENT_DEBOUNCE_SECONDS = 1.35
DEFAULT_AGENT_FRAGMENT_MAX_HOLD_SECONDS = 2.0
# Scenario 10 deliberately tests barge-in, so it keeps the snappy defaults above.
# Every other scenario uses these more patient values so the bot waits for the
# agent to finish instead of cutting in during pauses (sensible turn-taking).
BARGE_IN_SCENARIO_ID = 10
PATIENT_AGENT_COMPLETE_DEBOUNCE_SECONDS = 0.8
PATIENT_AGENT_FRAGMENT_MAX_HOLD_SECONDS = 4.5
# Voice-activity detection: how long of silence ends the agent's turn. A real VAD
# lets the pipeline detect when the agent starts talking and interrupt the
# patient's speech (so they stop talking over each other). Barge-in keeps it short.
DEFAULT_VAD_STOP_SECONDS = 1.8
BARGE_IN_VAD_STOP_SECONDS = 0.5
# After the VAD says the agent stopped, wait this long for any trailing transcript
# before committing the (now complete) agent turn as a single unit.
DEFAULT_AGENT_VAD_GRACE_SECONDS = 0.35
DEFAULT_VAD_START_SECONDS = 0.2
DEFAULT_VAD_CONFIDENCE = 0.7
DEFAULT_VAD_MIN_VOLUME = 0.6
DEFAULT_USER_AGGREGATION_TIMEOUT_SECONDS = 0.45

ROLE_BREAK_RE = re.compile(
    r"\b("
    r"ai assistant|as an ai|i am an ai|i'm an ai|not a patient|"
    r"help you today|assist you today|message got cut off|typing|"
    r"prompt|roleplay|play the role|playing the role"
    r")\b",
    re.IGNORECASE,
)
STAGE_DIRECTION_RE = re.compile(
    r"((?<!\*)\*[^*]{1,80}\*(?!\*)|\((?:rings?|pauses?|laughs?|sighs?|clears throat)[^)]{0,40}\))",
    re.IGNORECASE,
)
FRAGMENT_END_RE = re.compile(r"\b(about|and|but|for|from|in|of|on|or|to|with)\W*$", re.IGNORECASE)
FILLER_FRAGMENT_RE = re.compile(r"^(okay[,.]?\s*)?(but|i|uh|um|well|all of)\W*$", re.IGNORECASE)


def _initial_messages(system_prompt: str) -> list[dict[str, str]]:
    # The opening line is NOT seeded here. It is only added to history if/when it
    # is actually spoken (it flows through the pipeline's assistant aggregator).
    # If the agent greets first, we skip the canned opening and must not pretend
    # the patient already spoke it.
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "The live phone call has just connected. You are the patient caller, "
                "and the other speaker is the medical office agent. Stay in character "
                "and speak naturally as the patient."
            ),
        },
    ]


def _fallback_patient_response(scenario_id: int) -> str:
    try:
        return f"Sorry, I'm not sure I follow. {get_opening_line(scenario_id)}"
    except ValueError:
        return "Sorry, I'm not sure I follow. I'm just calling the office about this."


def _remove_emoji(text: str) -> str:
    return "".join(
        char
        for char in text
        if unicodedata.category(char) not in {"So", "Sk", "Cs", "Co"}
    )


def _strip_cosmetics(text: str) -> str:
    """Remove things that must never be spoken: stage directions, emoji, markdown.

    Does NOT check for role-break phrases — that is a separate safety gate.
    """
    cleaned = STAGE_DIRECTION_RE.sub("", text)
    cleaned = _remove_emoji(cleaned)
    cleaned = re.sub(r"[*_`#>]+", "", cleaned)
    cleaned = re.sub(r"^\s*[-+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _sanitize_voice_text(text: str, scenario_id: int) -> str:
    if ROLE_BREAK_RE.search(text):
        return _fallback_patient_response(scenario_id)

    cleaned = _strip_cosmetics(text)

    if ROLE_BREAK_RE.search(cleaned):
        return _fallback_patient_response(scenario_id)
    return cleaned


# A complete sentence/clause we can safely hand to TTS as soon as it lands, so the
# voice starts before the whole LLM response finishes.
_SENTENCE_RE = re.compile(r".*?[.!?]+[\"')\]]*(?:\s+|$)", re.DOTALL)


class StreamingResponseFilter:
    """Safely stream a patient LLM response to TTS sentence-by-sentence.

    Pass chunks in via `push()`; it returns text that is safe to speak NOW (empty
    if nothing is ready yet). It keeps a rolling guard over the full response so a
    role-break phrase is caught before any unsafe text is emitted: a sentence is
    only released after the cumulative text so far passes the role-break check.

    Cosmetic junk (markdown, emoji, stage directions) is stripped per segment.
    """

    def __init__(self, scenario_id: int):
        self.scenario_id = scenario_id
        self._full = ""        # everything seen, for cumulative role-break checks
        self._buffer = ""       # not-yet-released text
        self.unsafe = False
        self.emitted_any = False

    def push(self, chunk: str) -> str:
        if self.unsafe or not chunk:
            return ""
        self._full += chunk
        if ROLE_BREAK_RE.search(self._full):
            # Withhold everything pending; the dangerous phrase is unspoken so far.
            self.unsafe = True
            self._buffer = ""
            return ""
        self._buffer += chunk
        return self._release_complete_sentences()

    def _release_complete_sentences(self) -> str:
        out_parts: list[str] = []
        while True:
            match = _SENTENCE_RE.match(self._buffer)
            if not match or match.end() == 0:
                break
            segment = self._buffer[: match.end()]
            self._buffer = self._buffer[match.end() :]
            cleaned = _strip_cosmetics(segment)
            if cleaned:
                out_parts.append(cleaned)
        if not out_parts:
            return ""
        self.emitted_any = True
        return " ".join(out_parts)

    def finish(self) -> tuple[str, str | None]:
        """End of response. Returns (trailing_text, fallback).

        `fallback` is set only when the response was unsafe AND nothing safe was
        already spoken — so we never both leak partial text and inject a fallback.
        """
        if self.unsafe:
            fallback = (
                _fallback_patient_response(self.scenario_id)
                if not self.emitted_any
                else None
            )
            return "", fallback
        trailing = _strip_cosmetics(self._buffer)
        self._buffer = ""
        if trailing:
            self.emitted_any = True
        return trailing, None


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(value, minimum)
    return value


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(value, minimum)
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


# Short utterances that ARE meaningful in an appointment call and must not be
# discarded as fragments: affirmations and bare day/time confirmations.
_AFFIRMATIONS = {
    "yes", "yeah", "yep", "yup", "no", "nope", "okay", "ok", "sure",
    "alright", "all right", "great", "perfect", "got it", "sounds good",
    "of course", "correct", "right",
}
_DAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE
)
_CLOCK_TIME_RE = re.compile(
    r"\b(?:"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|o'clock)"
    r"|(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(?:am|pm|o'clock)"
    r"|noon|midnight"
    r")\b",
    re.IGNORECASE,
)


def _is_meaningful_short(bare: str) -> bool:
    """Short but useful: an affirmation, a weekday, or a clock time."""
    if bare in _AFFIRMATIONS:
        return True
    if _DAY_RE.search(bare):
        return True
    if _CLOCK_TIME_RE.search(bare):
        return True
    return False


def _is_agent_fragment(text: str) -> bool:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return True
    lower = cleaned.lower().strip()
    bare = lower.strip(" .?!,")
    # A completed question is never a fragment — even if it ends in a preposition
    # like "Where are you from?" or "What is this regarding?".
    if lower.endswith("?"):
        return False
    if FILLER_FRAGMENT_RE.search(lower):
        return True
    if bare in {"i can take you", "i can take"}:
        return True
    # Trailing comma or dangling preposition ("around 9 PM for") => incomplete.
    # Checked before fact-retention so "for"/"about" endings still drop.
    if lower.endswith(",") or FRAGMENT_END_RE.search(lower):
        return True
    # Keep short appointment facts: "Thursday.", "eleven AM.", "Sure.", "Okay.".
    if _is_meaningful_short(bare):
        return False
    if _word_count(lower) <= 2:
        return True
    if bare in {"what about", "okay what about"}:
        return True
    return False


def _join_transcript_parts(parts: list[str]) -> str:
    text = " ".join(part.strip() for part in parts if part.strip())
    return re.sub(r"\s+", " ", text).strip()


def patient_response_decision(
    *,
    consecutive_patient: bool,
    response_barge_generation: int,
    current_barge_generation: int,
    response_agent_generation: int,
    current_agent_generation: int,
    answered_agent_generation: int,
) -> str:
    """Decide whether a completed LLM patient response should be spoken.

    Enforces one patient response per committed agent turn. Returns "emit" or a
    reason string explaining why the response is dropped.
    """
    if consecutive_patient:
        return "suppress_consecutive"
    if response_barge_generation != current_barge_generation:
        return "drop_barged"  # caller interrupted while we were generating
    if response_agent_generation != current_agent_generation:
        return "drop_superseded"  # a newer agent turn arrived; this reply is stale
    if answered_agent_generation == response_agent_generation:
        return "drop_duplicate"  # already answered this exact agent turn
    return "emit"


def should_send_opening(
    turns: list[dict],
    agent_turn_generation: int,
    barge_in_generation: int = 0,
) -> bool:
    """Adaptive opening: only speak the canned opening if nobody has spoken yet."""
    if agent_turn_generation > 0:
        return False
    if barge_in_generation > 0:
        return False
    for turn in turns:
        if turn.get("speaker") in {"agent", "patient"}:
            return False
    return True


# Bare line-check phrases an agent uses before any real conversation: "Hello?",
# "Hi?", "Can you hear me?", "Are you there?". These never need an LLM round-trip.
_GREETING_PHRASES = (
    "can you hear me",
    "are you there",
    "is anyone there",
    "anybody there",
    "anyone there",
    "you there",
    "hello there",
    "hi there",
    "hey there",
    "hello",
    "hiya",
    "hi",
    "hey",
    "yo",
)


def _is_greeting_filler(text: str) -> bool:
    """True if `text` is nothing but greeting/line-check filler (no real content)."""
    stripped = re.sub(r"[^a-z\s]", " ", (text or "").lower())
    stripped = " ".join(stripped.split())
    if not stripped:
        return False
    changed = True
    while changed and stripped:
        changed = False
        for phrase in _GREETING_PHRASES:  # ordered longest-ish first
            if stripped == phrase or stripped.startswith(phrase + " "):
                stripped = stripped[len(phrase):].strip()
                changed = True
                break
    return stripped == ""


def greeting_fastpath_action(
    text: str,
    *,
    in_handshake: bool,
    opening_delivered: bool,
    opener_was_gated: bool = False,
) -> str:
    """Decide how the opening handshake handles an agent turn.

    - "passthrough": handshake is over; behave normally (let the LLM handle it).
    - "deliver_opener": agent only said a greeting and we have not opened yet —
      speak the deterministic opener now, no LLM. This is callee-gated (we heard
      them first), so it is the trusted primary path.
    - "recover_opener": we already sent an opener, but BLINDLY (silence fallback)
      and the callee is still saying "Hello?" — they likely never heard it. Re-greet
      with a recovery line instead of swallowing into dead air.
    - "swallow": agent repeated a bare greeting after we already responded to their
      speech (gated). Safe to ignore so we never loop on "hello".
    - "recover_midcall": a bare "Hello?" arrives AFTER the handshake (the agent did
      not hear the patient). Respond with a universal, content-free recovery line —
      never guess that the missed turn was about the name.
    - "forward": agent said something substantive — end the handshake and let the
      LLM answer the real content.
    """
    if _is_greeting_filler(text):
        if not in_handshake:
            return "recover_midcall"
        if not opening_delivered:
            return "deliver_opener"
        if not opener_was_gated:
            return "recover_opener"
        return "swallow"
    if not in_handshake:
        return "passthrough"
    return "forward"


def _recovery_opener_line(opening_line: str) -> str:
    """Spoken when a blind opener may have been missed and the callee re-greets."""
    return "Hi, can you hear me? " + opening_line


# Universal mid-call recovery: said when the agent re-greets ("Hello?") after the
# handshake. Deliberately content-free — it must NOT assume the missed turn was
# about the patient's name or any other specific fact.
MIDCALL_RECOVERY_LINE = "Sorry, I'm here. Could you repeat what you needed?"


# A name request and nothing else. Conservative: false negatives (missing a name
# question) are fine; false positives (answering "name" to a DOB/symptom/insurance
# question) are not, so any cue for other information disqualifies the fast path.
_NAME_CUE_RE = re.compile(
    r"\b("
    r"your (full |first |last )?name|"
    r"who am i speaking (with|to)|who'?s calling|may i ask who|"
    r"can i (get|have|ask) your name|what(?:'s| is) your name|"
    r"your name please|name for the (appointment|file|record)"
    r")\b",
    re.IGNORECASE,
)
# If any of these appear, the agent wants more than just the name — do not fast-path.
_NAME_DISQUALIFY_RE = re.compile(
    r"\b("
    r"date of birth|d\.?o\.?b\.?|birth|born|"
    r"symptom|pain|feeling|reason|visit|"
    r"insurance|member id|policy|"
    r"appointment|schedule|reschedule|cancel|time|day|when|"
    r"address|phone|number|email|spell|"
    r"medication|refill|prescription|emergency|"
    r"saturday|sunday|monday|tuesday|wednesday|thursday|friday"
    r")\b",
    re.IGNORECASE,
)


def name_fastpath_response(agent_text: str, caller_name: str) -> str | None:
    """A deterministic patient reply if the agent asked *only* for the name.

    Returns the spoken line (containing `caller_name`) or None if the turn is not
    a pure name request — in which case the LLM handles it as usual.
    """
    if not agent_text or not caller_name:
        return None
    if _NAME_DISQUALIFY_RE.search(agent_text):
        return None
    if not _NAME_CUE_RE.search(agent_text):
        return None
    return f"It's {caller_name}."


@dataclass
class CallPipeline:
    pipeline: Any
    task: Any
    runner: Any
    transport: Any
    turns: list[dict]
    live_recorder: LiveAudioRecorder | None = None
    telnyx_event_log: Any = None


class LatencyTracker:
    """Records per-turn latency checkpoints and writes latency_debug.json.

    A "turn" accumulates marks from agent-speech-start through patient-turn-logged.
    Calling `finalize()` snapshots the current turn and flushes to disk.
    """

    CHECKPOINTS = (
        "agent_speech_started",
        "stt_committed",
        "llm_response_start",
        "first_llm_token",
        "tts_started",
        "first_audio_out",
        "patient_turn_logged",
    )

    # Call-level (not per-turn) checkpoints, recorded once each.
    CALL_EVENTS = (
        "opener_scheduled_at",
        "opener_queued_at",
        "first_inbound_audio",
        "first_outbound_audio_before_transport",
    )

    # Call-level events mirrored to the cross-component wall-clock StartupTimeline.
    _TIMELINE_EVENT_MARKS = {
        "opener_scheduled_at": "opener_scheduled",
        "opener_queued_at": "opener_queued",
        "first_outbound_audio_before_transport": "first_outbound_audio_before_transport",
    }

    def __init__(
        self,
        scenario_id: int,
        call_sid: str,
        started_at: float | None = None,
        timeline: Any = None,
    ):
        self.scenario_id = scenario_id
        self.call_sid = call_sid
        self._t0 = started_at if started_at is not None else time.monotonic()
        self._current: dict[str, int] = {}
        self.turns: list[dict[str, int]] = []
        self.events: dict[str, int] = {}
        self._timeline = timeline

    def _elapsed_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def mark(self, name: str) -> None:
        # First write wins per turn (e.g. first LLM token, first audio frame).
        if name not in self._current:
            self._current[name] = self._elapsed_ms()

    def event(self, name: str) -> None:
        """Record a one-shot, call-level event timestamp (first write wins)."""
        if name not in self.events:
            self.events[name] = self._elapsed_ms()
            self._write()
            mark = self._TIMELINE_EVENT_MARKS.get(name)
            if mark and self._timeline is not None:
                self._timeline.mark(mark)

    def mark_first_audio_out(self) -> None:
        # Ignore trailing audio chunks that arrive after a turn has already been
        # finalized; otherwise they become the next turn's impossible "first" audio.
        if self._current:
            self.mark("first_audio_out")

    def finalize(self) -> None:
        if not self._current:
            return
        record = dict(self._current)
        # Convenience: time from final STT to first audio out (perceived latency).
        if "stt_committed" in record and "first_audio_out" in record:
            record["stt_to_first_audio_ms"] = record["first_audio_out"] - record["stt_committed"]
        self.turns.append(record)
        self._current = {}
        self._write()

    def maybe_finalize(self) -> None:
        """Flush once both the text turn and first audible audio are observed."""
        if "patient_turn_logged" in self._current and "first_audio_out" in self._current:
            self.finalize()

    def _write(self) -> None:
        try:
            run_dir = Path("runs") / f"scenario_{self.scenario_id:02d}_{self.call_sid}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "latency_debug.json").write_text(
                json.dumps(
                    {"call_sid": self.call_sid, "events": self.events, "turns": self.turns},
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # never let instrumentation break a call
            print(f"[DEBUG] latency write failed: {exc}")


@dataclass
class TurnState:
    # Incremented on each caller barge-in / user-started-speaking.
    barge_in_generation: int = 0
    # Incremented only when a debounced agent transcript is actually committed.
    # This is the unit a patient response "owns" — one reply per agent turn.
    agent_turn_generation: int = 0
    # The agent generation a patient response was already emitted for. Used to
    # drop duplicate rephrases that target the same agent turn.
    answered_agent_generation: int = -1
    last_agent_transcript_generation: int = 0
    # Opening handshake state. The opener is a deterministic, non-LLM utterance.
    opening_delivered: bool = False
    in_handshake: bool = True
    # True only if the opener was sent AFTER we actually heard inbound speech from
    # the callee (callee-gated). A blind silence-fallback opener is NOT gated, so a
    # later "Hello?" must be treated as recoverable rather than swallowed.
    opener_was_gated: bool = False
    # Live VAD state: True while the agent is actively speaking. The dynamic opener
    # uses this so the patient never starts talking over the agent — it speaks only
    # when the line is silent.
    agent_speaking: bool = False
    # True once we have ever detected the agent speaking on this call.
    agent_has_spoken: bool = False


class AgentTranscriptDebouncer:
    def __new__(cls, state: TurnState, tracker: "LatencyTracker | None" = None, scenario_id: int | None = None):
        from pipecat.frames.frames import (
            CancelFrame,
            EndFrame,
            StartInterruptionFrame,
            StopInterruptionFrame,
            TranscriptionFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
        )
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        class _AgentTranscriptDebouncer(FrameProcessor):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._parts: list[str] = []
                self._first_seen_at: float | None = None
                self._latest_frame: TranscriptionFrame | None = None
                self._flush_task: asyncio.Task | None = None
                self._flush_generation = 0
                self._in_interruption = False
                # VAD-gated aggregation: while the agent is actively speaking we
                # hold every transcript fragment and only commit one full turn once
                # the VAD reports the agent has stopped (plus a short grace).
                self._user_speaking = False
                self._vad_grace = _env_float(
                    "AGENT_VAD_GRACE_SECONDS",
                    DEFAULT_AGENT_VAD_GRACE_SECONDS,
                    minimum=0.0,
                )
                # Barge-in scenario keeps the snappy defaults; all others wait a
                # beat longer so the patient doesn't cut the agent off mid-turn.
                barge_in = scenario_id == BARGE_IN_SCENARIO_ID
                complete_default = (
                    DEFAULT_AGENT_COMPLETE_DEBOUNCE_SECONDS
                    if barge_in
                    else PATIENT_AGENT_COMPLETE_DEBOUNCE_SECONDS
                )
                max_hold_default = (
                    DEFAULT_AGENT_FRAGMENT_MAX_HOLD_SECONDS
                    if barge_in
                    else PATIENT_AGENT_FRAGMENT_MAX_HOLD_SECONDS
                )
                self._complete_delay = _env_float(
                    "AGENT_COMPLETE_DEBOUNCE_SECONDS",
                    complete_default,
                    minimum=0.0,
                )
                self._fragment_delay = _env_float(
                    "AGENT_FRAGMENT_DEBOUNCE_SECONDS",
                    DEFAULT_AGENT_FRAGMENT_DEBOUNCE_SECONDS,
                    minimum=0.0,
                )
                self._fragment_max_hold = _env_float(
                    "AGENT_FRAGMENT_MAX_HOLD_SECONDS",
                    max_hold_default,
                    minimum=self._fragment_delay,
                )

            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if direction != FrameDirection.DOWNSTREAM:
                    await self.push_frame(frame, direction)
                    return

                if isinstance(frame, StartInterruptionFrame):
                    if not self._in_interruption:
                        state.barge_in_generation += 1
                    self._in_interruption = True
                    await self.push_frame(frame, direction)
                    return

                if isinstance(frame, UserStartedSpeakingFrame):
                    if tracker is not None:
                        tracker.mark("agent_speech_started")
                    # Agent is talking now: hold any pending flush and accumulate
                    # everything they say into one turn until the VAD says stop.
                    self._user_speaking = True
                    state.agent_speaking = True
                    state.agent_has_spoken = True
                    await self._cancel_flush_task()
                    if not self._in_interruption:
                        state.barge_in_generation += 1
                        self._in_interruption = True
                    await self.push_frame(frame, direction)
                    return

                if isinstance(frame, UserStoppedSpeakingFrame):
                    # Agent finished its turn — commit the whole accumulated
                    # utterance as a single unit after a short grace for trailing
                    # transcript, instead of splitting it into part 1 / part 2.
                    self._user_speaking = False
                    state.agent_speaking = False
                    if self._parts:
                        # Agent paused. Commit only if the accumulated statement
                        # looks complete; if it's still mid-thought, keep holding so
                        # we don't answer half a sentence (PivotPoint pauses a lot).
                        await self._schedule_flush(force_delay=self._vad_grace)
                    await self.push_frame(frame, direction)
                    return

                if isinstance(frame, StopInterruptionFrame):
                    self._in_interruption = False
                    await self.push_frame(frame, direction)
                    return

                if isinstance(frame, TranscriptionFrame):
                    await self._buffer_transcription(frame)
                    return

                if isinstance(frame, (EndFrame, CancelFrame)):
                    await self._flush(force=True)
                    await self.push_frame(frame, direction)
                    return

                await self.push_frame(frame, direction)

            async def _buffer_transcription(self, frame: TranscriptionFrame) -> None:
                now = time.monotonic()
                if self._first_seen_at and now - self._first_seen_at > self._fragment_max_hold:
                    self._reset_buffer()
                if not self._parts:
                    self._first_seen_at = now
                self._parts.append(frame.text)
                self._latest_frame = frame
                if self._user_speaking:
                    # Still mid-utterance: keep accumulating, commit on VAD stop.
                    await self._cancel_flush_task()
                else:
                    # No active speech (VAD already stopped, or VAD unavailable):
                    # trailing text — re-evaluate completeness after a short window.
                    await self._schedule_flush(force_delay=self._vad_grace)

            async def _schedule_flush(self, force_delay: float | None = None, force: bool = False) -> None:
                await self._cancel_flush_task()
                if force_delay is not None:
                    delay = force_delay
                else:
                    text = _join_transcript_parts(self._parts)
                    delay = self._fragment_delay if _is_agent_fragment(text) else self._complete_delay
                self._flush_generation += 1
                generation = self._flush_generation
                self._flush_task = self.create_task(self._flush_after(delay, generation, force))

            async def _cancel_flush_task(self) -> None:
                if self._flush_task and self._flush_task is not asyncio.current_task():
                    await self.cancel_task(self._flush_task)
                self._flush_task = None

            async def _flush_after(self, delay: float, generation: int, force: bool = False) -> None:
                await time_sleep(delay)
                if generation != self._flush_generation:
                    return
                await self._flush(force=force)

            async def _flush(self, force: bool = False) -> None:
                if not self._parts or not self._latest_frame:
                    return
                text = _join_transcript_parts(self._parts)
                is_fragment = _is_agent_fragment(text)
                held_for = time.monotonic() - (self._first_seen_at or time.monotonic())
                if is_fragment and not force and held_for < self._fragment_max_hold:
                    # Still mid-thought and within the hold window: keep waiting for
                    # the agent to finish the sentence instead of answering a fragment.
                    await self._schedule_flush()
                    return
                # Commit: text looks complete, OR we held a fragment past the max
                # window (commit rather than drop so the agent's turn isn't lost).

                frame = TranscriptionFrame(
                    text,
                    self._latest_frame.user_id,
                    self._latest_frame.timestamp,
                    self._latest_frame.language,
                )
                self._reset_buffer()
                # A real agent turn is now committed — open a new generation that a
                # single patient response is allowed to answer.
                state.agent_turn_generation += 1
                state.last_agent_transcript_generation = state.barge_in_generation
                if tracker is not None:
                    tracker.mark("stt_committed")
                await self.push_frame(frame)

            def _reset_buffer(self) -> None:
                self._parts = []
                self._first_seen_at = None
                self._latest_frame = None
                self._flush_task = None

        return _AgentTranscriptDebouncer(name="AgentTranscriptDebouncer")


class GreetingFastPath:
    """Deterministic opening handshake — the first agent turn skips the LLM.

    A bare greeting ("Hello? Can you hear me?") triggers the canned opener
    immediately and is swallowed, so the bot never round-trips through Claude on
    "hello" (which caused the multi-second dead-air loop). A substantive first turn
    ends the handshake and flows to the LLM normally.

    Placed upstream of the user-context aggregator: swallowing a greeting here means
    the LLM is never triggered for it, so there is no duplicate response to suppress.
    """

    def __new__(
        cls,
        opening_line: str,
        state: TurnState,
        tracker: "LatencyTracker | None" = None,
    ):
        from pipecat.frames.frames import (
            LLMFullResponseEndFrame,
            LLMFullResponseStartFrame,
            TextFrame,
            TranscriptionFrame,
        )
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        class _GreetingFastPath(FrameProcessor):
            async def _emit(self, line: str):
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(TextFrame(line))
                await self.push_frame(LLMFullResponseEndFrame())

            async def _speak(self, line: str):
                state.opening_delivered = True
                # Reaching here means we heard the callee's speech first, so the
                # opener is now callee-gated and trustworthy.
                state.opener_was_gated = True
                if tracker is not None:
                    tracker.event("opener_queued_at")
                await self._emit(line)

            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if (
                    isinstance(frame, TranscriptionFrame)
                    and direction == FrameDirection.DOWNSTREAM
                    and frame.text.strip()
                ):
                    action = greeting_fastpath_action(
                        frame.text,
                        in_handshake=state.in_handshake,
                        opening_delivered=state.opening_delivered,
                        opener_was_gated=state.opener_was_gated,
                    )
                    if action == "deliver_opener":
                        print("[DEBUG] Greeting fast path: callee-gated opener (no LLM).")
                        await self._speak(opening_line)
                        return  # swallow the greeting; LLM never sees it
                    if action == "recover_opener":
                        print("[DEBUG] Greeting fast path: blind opener may have been missed; re-greeting.")
                        await self._speak(_recovery_opener_line(opening_line))
                        return
                    if action == "recover_midcall":
                        print("[DEBUG] Greeting fast path: mid-call re-greet; universal recovery (no LLM).")
                        await self._emit(MIDCALL_RECOVERY_LINE)
                        return  # swallow the bare "hello?"; do not guess content
                    if action == "swallow":
                        print("[DEBUG] Greeting fast path: swallowing redundant greeting.")
                        return
                    if action == "forward":
                        # Substantive agent turn — handshake is over, let the LLM answer.
                        state.in_handshake = False
                        state.opening_delivered = True
                    # passthrough / forward both fall through to normal routing.
                await self.push_frame(frame, direction)

        return _GreetingFastPath(name="GreetingFastPath")


class NameFastPath:
    """Deterministic, no-LLM reply when the agent asks only for the caller's name.

    Placed after GreetingFastPath and before the user-context aggregator (same as
    the greeting fast path): swallowing the agent transcription here means the LLM
    is never triggered for it, so there is no duplicate patient response to drop.

    Conservative gating keeps it safe:
    - only after the opening handshake is over (`in_handshake` is False),
    - only once the patient has already spoken at least one turn (so we never
      pre-empt the opener/intro),
    - only when `name_fastpath_response` matches a pure name request.
    """

    def __new__(
        cls,
        caller_name: str,
        turns: list[dict],
        state: "TurnState",
        enabled: bool = True,
    ):
        from pipecat.frames.frames import (
            LLMFullResponseEndFrame,
            LLMFullResponseStartFrame,
            TextFrame,
            TranscriptionFrame,
        )
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        class _NameFastPath(FrameProcessor):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if (
                    enabled
                    and isinstance(frame, TranscriptionFrame)
                    and direction == FrameDirection.DOWNSTREAM
                    and not state.in_handshake
                    and any(t.get("speaker") == "patient" for t in turns)
                ):
                    line = name_fastpath_response(frame.text, caller_name)
                    if line:
                        print(f"[DEBUG] Name fast path: answering name deterministically (no LLM).")
                        await self.push_frame(LLMFullResponseStartFrame())
                        await self.push_frame(TextFrame(line))
                        await self.push_frame(LLMFullResponseEndFrame())
                        return  # swallow the agent turn; LLM never sees it
                await self.push_frame(frame, direction)

        return _NameFastPath(name="NameFastPath")


class AgentTurnLogger:
    def __new__(cls, turns: list[dict], started_at: float):
        from pipecat.frames.frames import TranscriptionFrame
        from pipecat.processors.frame_processor import FrameProcessor

        class _AgentTurnLogger(FrameProcessor):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if isinstance(frame, TranscriptionFrame) and frame.text.strip():
                    elapsed_ms = int((time.monotonic() - started_at) * 1000)
                    turns.append(
                        {
                            "speaker": "agent",
                            "text": frame.text.strip(),
                            "start_ms": elapsed_ms,
                            "end_ms": elapsed_ms,
                        }
                    )
                await self.push_frame(frame, direction)

        return _AgentTurnLogger(name="AgentTurnLogger")


class PatientTurnLogger:
    def __new__(cls, turns: list[dict], started_at: float, tracker: "LatencyTracker | None" = None):
        from pipecat.frames.frames import (
            EndFrame,
            LLMFullResponseEndFrame,
            LLMFullResponseStartFrame,
            TextFrame,
        )
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        class _PatientTurnLogger(FrameProcessor):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._capturing = False
                self._parts: list[str] = []

            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if isinstance(frame, LLMFullResponseStartFrame):
                    self._capturing = True
                    self._parts = []
                elif isinstance(frame, TextFrame) and self._capturing:
                    self._parts.append(frame.text)
                elif isinstance(frame, LLMFullResponseEndFrame) and self._capturing:
                    text = "".join(self._parts).strip()
                    self._capturing = False
                    self._parts = []
                    if text:
                        elapsed_ms = int((time.monotonic() - started_at) * 1000)
                        turns.append(
                            {
                                "speaker": "patient",
                                "text": text,
                                "start_ms": elapsed_ms,
                                "end_ms": elapsed_ms,
                            }
                        )
                        if tracker is not None:
                            tracker.mark("patient_turn_logged")
                            tracker.maybe_finalize()
                        if should_end_call(text):
                            self.create_task(self._end_after_audio_delay())

                await self.push_frame(frame, direction)

            async def _end_after_audio_delay(self):
                await time_sleep(2.0)
                await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)

        return _PatientTurnLogger(name="PatientTurnLogger")


class VoiceResponseGuard:
    def __new__(
        cls,
        scenario_id: int,
        turns: list[dict],
        state: TurnState,
        tracker: "LatencyTracker | None" = None,
    ):
        from pipecat.frames.frames import (
            LLMFullResponseEndFrame,
            LLMFullResponseStartFrame,
            StartInterruptionFrame,
            TextFrame,
            UserStartedSpeakingFrame,
        )
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        class _VoiceResponseGuard(FrameProcessor):
            """Streams sanitized patient text to TTS as it arrives.

            - Releases complete sentences immediately (low latency) instead of
              buffering the whole response.
            - Rolling role-break guard: unsafe responses fall back without leaking
              partial unsafe text.
            - Preserves stale/duplicate suppression via barge_in_generation and
              agent_turn_generation.
            """

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._reset()

            def _reset(self):
                self._capturing = False
                self._dropping = False
                self._aborted = False
                self._started_out = False  # emitted a downstream Start yet?
                self._filter: StreamingResponseFilter | None = None
                self._response_barge_generation = 0
                self._response_agent_generation = -1

            async def _ensure_started(self):
                if not self._started_out:
                    self._started_out = True
                    await self.push_frame(LLMFullResponseStartFrame())

            async def _close(self):
                if self._started_out:
                    await self.push_frame(LLMFullResponseEndFrame())

            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if direction != FrameDirection.DOWNSTREAM:
                    await self.push_frame(frame, direction)
                    return

                if isinstance(frame, (StartInterruptionFrame, UserStartedSpeakingFrame)):
                    # Caller barged in mid-generation: abandon the in-flight response.
                    if self._capturing and not self._dropping:
                        self._aborted = True
                    await self.push_frame(frame, direction)
                    return

                if isinstance(frame, LLMFullResponseStartFrame):
                    self._capturing = True
                    self._aborted = False
                    self._started_out = False
                    self._response_barge_generation = state.barge_in_generation
                    self._response_agent_generation = state.agent_turn_generation
                    # Early drop decisions knowable at start: consecutive patient
                    # turn, or a duplicate for an already-answered agent turn.
                    consecutive = bool(turns and turns[-1].get("speaker") == "patient")
                    decision = patient_response_decision(
                        consecutive_patient=consecutive,
                        response_barge_generation=self._response_barge_generation,
                        current_barge_generation=state.barge_in_generation,
                        response_agent_generation=self._response_agent_generation,
                        current_agent_generation=state.agent_turn_generation,
                        answered_agent_generation=state.answered_agent_generation,
                    )
                    self._dropping = decision != "emit"
                    self._filter = StreamingResponseFilter(scenario_id)
                    if self._dropping:
                        print(f"[DEBUG] Dropping patient response ({decision}) for agent turn {self._response_agent_generation}.")
                    elif tracker is not None:
                        tracker.mark("llm_response_start")
                    return

                if isinstance(frame, TextFrame) and self._capturing:
                    if self._dropping or self._aborted:
                        return
                    if tracker is not None:
                        tracker.mark("first_llm_token")
                    # Became stale while generating? Drop the rest.
                    if (
                        self._response_barge_generation != state.barge_in_generation
                        or self._response_agent_generation != state.agent_turn_generation
                    ):
                        self._aborted = True
                        await self._close()
                        return
                    safe = self._filter.push(frame.text)
                    if safe:
                        await self._ensure_started()
                        await self.push_frame(TextFrame(safe + " "))
                    return

                if isinstance(frame, LLMFullResponseEndFrame) and self._capturing:
                    capturing_filter = self._filter
                    dropping = self._dropping
                    aborted = self._aborted
                    agent_gen = self._response_agent_generation
                    await self._finish_response(capturing_filter, dropping, aborted, agent_gen)
                    self._reset()
                    return

                await self.push_frame(frame, direction)

            async def _finish_response(self, response_filter, dropping, aborted, agent_gen):
                if dropping:
                    return
                if aborted:
                    print(f"[DEBUG] Dropped stale/barged patient response for agent turn {agent_gen}.")
                    await self._close()
                    return
                trailing, fallback = response_filter.finish()
                if fallback is not None:
                    await self._ensure_started()
                    await self.push_frame(TextFrame(fallback))
                elif trailing:
                    await self._ensure_started()
                    await self.push_frame(TextFrame(trailing))
                if self._started_out:
                    # This agent turn is now answered — block any later duplicate.
                    state.answered_agent_generation = agent_gen
                    await self._close()

        return _VoiceResponseGuard(name="VoiceResponseGuard")


class LatencyProbe:
    """Marks TTS-started and first-outbound-audio checkpoints. Place after TTS."""

    def __new__(cls, tracker: "LatencyTracker | None"):
        from pipecat.frames.frames import (
            OutputAudioRawFrame,
            TTSAudioRawFrame,
            TTSStartedFrame,
        )
        from pipecat.processors.frame_processor import FrameProcessor

        class _LatencyProbe(FrameProcessor):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if tracker is not None:
                    if isinstance(frame, TTSStartedFrame):
                        tracker.mark("tts_started")
                    elif isinstance(frame, (TTSAudioRawFrame, OutputAudioRawFrame)):
                        tracker.mark_first_audio_out()
                        tracker.maybe_finalize()
                await self.push_frame(frame, direction)

        return _LatencyProbe(name="LatencyProbe")


class LiveAudioCapture:
    """Tee live audio frames to the per-call recorder."""

    def __new__(
        cls,
        recorder: LiveAudioRecorder | None,
        speaker: str,
        tracker: "LatencyTracker | None" = None,
    ):
        from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
        from pipecat.processors.frame_processor import FrameProcessor

        class _LiveAudioCapture(FrameProcessor):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if speaker == "agent" and isinstance(frame, InputAudioRawFrame):
                    if recorder is not None:
                        recorder.write_agent(frame.audio, frame.sample_rate, frame.num_channels)
                    if tracker is not None:
                        tracker.event("first_inbound_audio")
                elif speaker == "patient" and isinstance(frame, OutputAudioRawFrame):
                    if recorder is not None:
                        recorder.write_patient(frame.audio, frame.sample_rate, frame.num_channels)
                    if tracker is not None:
                        tracker.event("first_outbound_audio_before_transport")
                await self.push_frame(frame, direction)

        return _LiveAudioCapture(name=f"LiveAudioCapture:{speaker}")


async def time_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def build_pipeline(
    scenario_id: int,
    websocket,
    call_sid: str,
    stream_sid: str,
    telephony_provider: str = "twilio",
    inbound_encoding: str = "PCMU",
    outbound_encoding: str = "PCMU",
    turns: list[dict] | None = None,
) -> CallPipeline:
    from deepgram import LiveOptions
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.frames.frames import (
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        TextFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.services.cartesia import CartesiaTTSService
    from pipecat.services.deepgram import DeepgramSTTService
    from pipecat.transports.network.fastapi_websocket import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    from src.startup_timeline import StartupTimeline

    turns = turns if turns is not None else []
    turn_state = TurnState()
    started_at = time.monotonic()
    timeline = StartupTimeline(scenario_id, call_sid)
    tracker = LatencyTracker(scenario_id, call_sid, started_at=started_at, timeline=timeline)
    live_recorder = LiveAudioRecorder(scenario_id, call_sid, started_at=started_at)
    system_prompt = get_system_prompt(scenario_id)
    opening_line = get_opening_line(scenario_id)
    from scenarios.scenario_cards import get_scenario

    _scenario_card = get_scenario(scenario_id)
    caller_name = _scenario_card.caller_name if _scenario_card else ""
    name_fastpath_enabled = _env_bool("NAME_FASTPATH_ENABLED", True)
    print(f"[DEBUG] System prompt loaded: {system_prompt[:100]}...")

    telnyx_event_log: "TelnyxWsEventLog | None" = None
    if telephony_provider == "telnyx":
        from src.telnyx_instrumentation import (
            InstrumentedTelnyxSerializer,
            TelnyxWsEventLog,
        )

        telnyx_event_log = TelnyxWsEventLog(scenario_id, call_sid, started_at=started_at)
        # Prime-suspect toggle: the stock serializer omits stream_id on outbound
        # media; some Telnyx bidirectional setups need it to route playback. On by
        # default so the fix ships; set TELNYX_INJECT_STREAM_ID=0 to A/B the bug.
        include_stream_id = _env_bool("TELNYX_INJECT_STREAM_ID", True)
        serializer = InstrumentedTelnyxSerializer(
            stream_id=stream_sid,
            inbound_encoding=inbound_encoding,
            outbound_encoding=outbound_encoding,
            event_log=telnyx_event_log,
            include_stream_id=include_stream_id,
            timeline=timeline,
        )
        print(f"[DEBUG] Telnyx serializer instrumented (inject_stream_id={include_stream_id}).")
    else:
        from pipecat.serializers.twilio import TwilioFrameSerializer

        serializer = TwilioFrameSerializer(stream_sid=stream_sid)

    # Real voice-activity detection so the pipeline knows when the agent is talking.
    # This is what lets allow_interruptions actually stop the patient's speech the
    # moment the agent speaks — the fix for the two sides talking over each other.
    vad_stop_secs = _env_float(
        "VAD_STOP_SECONDS",
        BARGE_IN_VAD_STOP_SECONDS if scenario_id == BARGE_IN_SCENARIO_ID else DEFAULT_VAD_STOP_SECONDS,
        minimum=0.1,
    )
    vad_analyzer = SileroVADAnalyzer(
        sample_rate=8000,
        params=VADParams(
            confidence=_env_float("VAD_CONFIDENCE", DEFAULT_VAD_CONFIDENCE, minimum=0.1),
            start_secs=_env_float("VAD_START_SECONDS", DEFAULT_VAD_START_SECONDS, minimum=0.0),
            stop_secs=vad_stop_secs,
            min_volume=_env_float("VAD_MIN_VOLUME", DEFAULT_VAD_MIN_VOLUME, minimum=0.0),
        ),
    )
    transport = FastAPIWebsocketTransport(
        websocket,
        FastAPIWebsocketParams(
            serializer=serializer,
            audio_in_enabled=True,
            audio_in_sample_rate=8000,
            audio_out_enabled=True,
            audio_out_sample_rate=8000,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            session_timeout=300,
        ),
    )

    live_options = LiveOptions(
        encoding="linear16",
        sample_rate=8000,
        channels=1,
        language="en-US",
        model="nova-2",
        interim_results=False,
        smart_format=True,
        punctuate=True,
        endpointing=_env_int(
            "DEEPGRAM_ENDPOINTING_MS",
            DEFAULT_DEEPGRAM_ENDPOINTING_MS,
            minimum=100,
        ),
    )
    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        sample_rate=8000,
        live_options=live_options,
    )
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        from pipecat.services.anthropic import AnthropicLLMService

        llm = AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
            params=AnthropicLLMService.InputParams(temperature=0.35, max_tokens=90),
        )
        context = OpenAILLMContext(messages=_initial_messages(system_prompt))
    elif provider == "openai":
        from pipecat.services.openai import OpenAILLMService

        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            params=OpenAILLMService.InputParams(temperature=0.35, max_tokens=90),
        )
        context = OpenAILLMContext(messages=_initial_messages(system_prompt))
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        voice_id=os.environ.get("CARTESIA_VOICE_ID", CARTESIA_VOICE_ID),
        model=os.environ.get("CARTESIA_MODEL", DEFAULT_CARTESIA_MODEL),
        sample_rate=8000,
        encoding="pcm_s16le",
    )

    context_aggregator = llm.create_context_aggregator(context)
    user_context_aggregator = context_aggregator.user()
    if hasattr(user_context_aggregator, "_aggregation_timeout"):
        user_context_aggregator._aggregation_timeout = _env_float(
            "USER_AGGREGATION_TIMEOUT_SECONDS",
            DEFAULT_USER_AGGREGATION_TIMEOUT_SECONDS,
            minimum=0.1,
        )

    pipeline = Pipeline(
        [
            transport.input(),
            LiveAudioCapture(live_recorder, "agent", tracker),
            stt,
            AgentTranscriptDebouncer(turn_state, tracker, scenario_id),
            AgentTurnLogger(turns, started_at),
            GreetingFastPath(opening_line, turn_state, tracker),
            NameFastPath(caller_name, turns, turn_state, name_fastpath_enabled),
            user_context_aggregator,
            llm,
            VoiceResponseGuard(scenario_id, turns, turn_state, tracker),
            PatientTurnLogger(turns, started_at, tracker),
            tts,
            LiveAudioCapture(live_recorder, "patient", tracker),
            LatencyProbe(tracker),
            transport.output(),
            context_aggregator.assistant(),
        ]
    )
    # Turn-taking contract: the patient only speaks once the agent is silent (the
    # VAD-gated debouncer holds the agent's whole utterance until VAD reports stop,
    # so we respond to a full statement, not part-by-part), and it STOPS talking the
    # moment the agent makes noise (allow_interruptions). The earlier "no audio /
    # are-you-still-there" loop came from premature starts during the agent's
    # mid-turn pauses — VAD aggregation fixes that root cause, so yielding to the
    # agent is now safe and gives clean patient/agent/patient alternation.
    allow_interruptions = _env_bool("ALLOW_INTERRUPTIONS", True)
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=allow_interruptions,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    runner = PipelineRunner(handle_sigint=False)

    # Hold strong references so the background opening task is not garbage collected.
    startup_tasks: list = []

    async def _open_when_silent() -> None:
        # DYNAMIC OPENER — no fixed start time. The patient speaks its opening line
        # only when the line is silent (the agent isn't talking), driven by the VAD:
        #   - If the agent greets first, the callee-gated GreetingFastPath delivers
        #     the opener the instant their greeting ends; this loop stands down.
        #   - While the agent is talking, we wait (never talk over them).
        #   - The moment the agent has been silent long enough, the patient opens.
        #   - If the agent never speaks at all (dead/silent line), open after the
        #     dead-line window so the call doesn't stall.
        silence_needed = _env_float(
            "OPENING_SILENCE_SECONDS", DEFAULT_OPENING_SILENCE_SECONDS, minimum=0.2
        )
        dead_line_after = _env_float(
            "OPENING_DEAD_LINE_SECONDS", DEFAULT_OPENING_DEAD_LINE_SECONDS, minimum=silence_needed
        )
        step = 0.1
        waited = 0.0
        silent_for = 0.0
        while True:
            await time_sleep(step)
            waited += step
            if turn_state.opening_delivered:
                return
            # Agent spoke and the fast path already owns the opener -> stand down.
            if turn_state.agent_has_spoken and not should_send_opening(
                turns, turn_state.agent_turn_generation, turn_state.barge_in_generation
            ):
                return
            if turn_state.agent_speaking:
                silent_for = 0.0
                continue
            silent_for += step
            # Open once the line is quiet: immediately after the agent's greeting
            # ends, or — if the agent never spoke — after the dead-line window.
            if silent_for >= silence_needed and (
                turn_state.agent_has_spoken or waited >= dead_line_after
            ):
                break

        if turn_state.opening_delivered or not should_send_opening(
            turns, turn_state.agent_turn_generation, turn_state.barge_in_generation
        ):
            return
        print("[DEBUG] Dynamic opener: line is silent, sending opening line.")
        turn_state.opening_delivered = True
        turn_state.opener_was_gated = turn_state.agent_has_spoken
        tracker.event("opener_queued_at")
        await task.queue_frame(LLMFullResponseStartFrame())
        await task.queue_frame(TextFrame(opening_line))
        await task.queue_frame(LLMFullResponseEndFrame())

    async def _run_audio_probe() -> None:
        # AUDIO-PATH PROBE. Bypasses LLM/TTS entirely: pushes a known PCMU test
        # tone straight to the output transport (-> serializer -> Telnyx). If the
        # caller hears the tone, playback works and the bug is in the TTS/timing
        # path. If the caller hears silence, the bug is the Telnyx payload/codec/
        # target-leg/stream_id setup. Bypassing the opener makes the test clean.
        from pipecat.frames.frames import OutputAudioRawFrame

        from src.telnyx_instrumentation import chunk_pcm, generate_probe_tone

        seconds = _env_float("TELNYX_PROBE_TONE_SECONDS", 2.0, minimum=0.2)
        repeats = _env_int("TELNYX_PROBE_REPEATS", 3, minimum=1)
        gap = _env_float("TELNYX_PROBE_GAP_SECONDS", 1.5, minimum=0.0)
        tone = generate_probe_tone(seconds=seconds, sample_rate=8000)
        frame_bytes = 320  # 20ms @ 8kHz mono 16-bit
        print(f"[DEBUG] AUDIO PROBE active: {seconds}s tone x{repeats}. LLM/TTS bypassed.")
        for _ in range(repeats):
            for chunk in chunk_pcm(tone, frame_bytes):
                await task.queue_frame(
                    OutputAudioRawFrame(audio=chunk, sample_rate=8000, num_channels=1)
                )
            await time_sleep(gap)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _websocket):
        # Non-blocking: schedule the silence fallback and return immediately so the
        # receive loop starts capturing early caller audio (e.g. "Hello?") right away
        # — that inbound speech is what drives the primary, callee-gated opener.
        tracker.event("opener_scheduled_at")
        if _env_bool("TELNYX_AUDIO_PROBE", False):
            startup_tasks.append(asyncio.create_task(_run_audio_probe()))
            return
        startup_tasks.append(asyncio.create_task(_open_when_silent()))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _websocket):
        for startup_task in startup_tasks:
            startup_task.cancel()
        if startup_tasks:
            await asyncio.gather(*startup_tasks, return_exceptions=True)
        await task.cancel()

    @transport.event_handler("on_session_timeout")
    async def on_session_timeout(_transport, _websocket):
        await task.queue_frame(LLMFullResponseStartFrame())
        await task.queue_frame(TextFrame("I should go now. Thank you, bye."))
        await task.queue_frame(LLMFullResponseEndFrame())

    timeline.mark("pipeline_built")

    return CallPipeline(
        pipeline=pipeline,
        task=task,
        runner=runner,
        transport=transport,
        turns=turns,
        live_recorder=live_recorder,
        telnyx_event_log=telnyx_event_log,
    )
