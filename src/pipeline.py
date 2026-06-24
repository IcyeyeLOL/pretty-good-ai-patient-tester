from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from src.patient_bot import get_opening_line, get_system_prompt, should_end_call

load_dotenv()

CARTESIA_VOICE_ID = "a0e99841-438c-4a64-b679-ae501e7d6091"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_CARTESIA_MODEL = "sonic-2"
DEFAULT_OPENING_DELAY_SECONDS = 3.0

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


def _initial_messages(system_prompt: str, opening_line: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "The live phone call has just connected. You are the patient caller, "
                "and the other speaker is the medical office agent. Your first spoken "
                "line has already been sent to the phone audio."
            ),
        },
        {"role": "assistant", "content": opening_line},
    ]


def _fallback_patient_response(scenario_id: int) -> str:
    if scenario_id == 1:
        return "Sorry, I'm not sure I follow. I'm Maria, and I'm calling to schedule as a new patient."
    return "Sorry, I'm not sure I follow. I'm just calling the office about this."


def _remove_emoji(text: str) -> str:
    return "".join(
        char
        for char in text
        if unicodedata.category(char) not in {"So", "Sk", "Cs", "Co"}
    )


def _sanitize_voice_text(text: str, scenario_id: int) -> str:
    if ROLE_BREAK_RE.search(text):
        return _fallback_patient_response(scenario_id)

    cleaned = STAGE_DIRECTION_RE.sub("", text)
    cleaned = _remove_emoji(cleaned)
    cleaned = re.sub(r"[*_`#>]+", "", cleaned)
    cleaned = re.sub(r"^\s*[-+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if ROLE_BREAK_RE.search(cleaned):
        return _fallback_patient_response(scenario_id)
    return cleaned


@dataclass
class CallPipeline:
    pipeline: Any
    task: Any
    runner: Any
    transport: Any
    turns: list[dict]


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
    def __new__(cls, turns: list[dict], started_at: float):
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
                        if should_end_call(text):
                            self.create_task(self._end_after_audio_delay())

                await self.push_frame(frame, direction)

            async def _end_after_audio_delay(self):
                await time_sleep(2.0)
                await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)

        return _PatientTurnLogger(name="PatientTurnLogger")


class VoiceResponseGuard:
    def __new__(cls, scenario_id: int, turns: list[dict]):
        from pipecat.frames.frames import (
            LLMFullResponseEndFrame,
            LLMFullResponseStartFrame,
            TextFrame,
        )
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

        class _VoiceResponseGuard(FrameProcessor):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._capturing = False
                self._suppressing = False
                self._parts: list[str] = []

            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if direction != FrameDirection.DOWNSTREAM:
                    await self.push_frame(frame, direction)
                    return

                if isinstance(frame, LLMFullResponseStartFrame):
                    self._capturing = True
                    self._suppressing = bool(turns and turns[-1].get("speaker") == "patient")
                    self._parts = []
                    if self._suppressing:
                        print("[DEBUG] Suppressing consecutive patient response.")
                    return

                if isinstance(frame, TextFrame) and self._capturing:
                    if not self._suppressing:
                        self._parts.append(frame.text)
                    return

                if isinstance(frame, LLMFullResponseEndFrame) and self._capturing:
                    text = "".join(self._parts)
                    self._capturing = False
                    suppressing = self._suppressing
                    self._suppressing = False
                    self._parts = []
                    if suppressing:
                        return
                    cleaned = _sanitize_voice_text(text, scenario_id)
                    if not cleaned:
                        return
                    await self.push_frame(LLMFullResponseStartFrame())
                    await self.push_frame(TextFrame(cleaned))
                    await self.push_frame(LLMFullResponseEndFrame())
                    return

                await self.push_frame(frame, direction)

        return _VoiceResponseGuard(name="VoiceResponseGuard")


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

    turns = turns if turns is not None else []
    started_at = time.monotonic()
    system_prompt = get_system_prompt(scenario_id)
    opening_line = get_opening_line(scenario_id)
    print(f"[DEBUG] System prompt loaded: {system_prompt[:100]}...")

    if telephony_provider == "telnyx":
        from pipecat.serializers.telnyx import TelnyxFrameSerializer

        serializer = TelnyxFrameSerializer(
            stream_id=stream_sid,
            inbound_encoding=inbound_encoding,
            outbound_encoding=outbound_encoding,
        )
    else:
        from pipecat.serializers.twilio import TwilioFrameSerializer

        serializer = TwilioFrameSerializer(stream_sid=stream_sid)
    transport = FastAPIWebsocketTransport(
        websocket,
        FastAPIWebsocketParams(
            serializer=serializer,
            audio_in_enabled=True,
            audio_in_sample_rate=8000,
            audio_out_enabled=True,
            audio_out_sample_rate=8000,
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
        endpointing=500,
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
        context = OpenAILLMContext(messages=_initial_messages(system_prompt, opening_line))
    elif provider == "openai":
        from pipecat.services.openai import OpenAILLMService

        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            params=OpenAILLMService.InputParams(temperature=0.35, max_tokens=90),
        )
        context = OpenAILLMContext(messages=_initial_messages(system_prompt, opening_line))
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

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            AgentTurnLogger(turns, started_at),
            context_aggregator.user(),
            llm,
            VoiceResponseGuard(scenario_id, turns),
            PatientTurnLogger(turns, started_at),
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    runner = PipelineRunner(handle_sigint=False)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _websocket):
        await time_sleep(float(os.environ.get("OPENING_DELAY_SECONDS", DEFAULT_OPENING_DELAY_SECONDS)))
        await task.queue_frame(LLMFullResponseStartFrame())
        await task.queue_frame(TextFrame(opening_line))
        await task.queue_frame(LLMFullResponseEndFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _websocket):
        await task.cancel()

    @transport.event_handler("on_session_timeout")
    async def on_session_timeout(_transport, _websocket):
        await task.queue_frame(LLMFullResponseStartFrame())
        await task.queue_frame(TextFrame("I should go now. Thank you, bye."))
        await task.queue_frame(LLMFullResponseEndFrame())

    return CallPipeline(
        pipeline=pipeline,
        task=task,
        runner=runner,
        transport=transport,
        turns=turns,
    )
