"""Telnyx websocket instrumentation.

The local `recording.mp3` / `live_patient.wav` only prove our app *generated*
patient audio. They do NOT prove Telnyx accepted, queued, or played that audio
into the phone call. This module closes that evidence gap by wrapping the
Pipecat `TelnyxFrameSerializer` — the single choke point every outbound frame
(`serialize`) and every inbound message (`deserialize`) passes through.

It does three things:

1. Logs outbound media frames (count, byte length, codec, stream_id presence)
   so we can prove frames were actually serialized and sent over the socket.
2. Logs inbound non-media events — crucially Telnyx `error` and `mark` frames —
   which the stock serializer silently drops (it returns None for anything that
   is not `media` or `dtmf`). An `invalid_media`/playback error is otherwise
   invisible.
3. Optionally injects `stream_id` into the outbound media JSON. The stock
   serializer emits `{"event": "media", "media": {"payload": ...}}` with no
   `stream_id`; some Telnyx bidirectional configurations require it to route
   playback to the call leg. This is the prime suspect for "audio generated but
   caller hears silence", so it is toggleable for A/B testing on a live call.

All output goes to `runs/scenario_XX_<call_sid>/telnyx_ws_events.jsonl`.
"""

from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path
from typing import Any

from pipecat.serializers.telnyx import TelnyxFrameSerializer


def generate_probe_tone(
    *,
    seconds: float = 2.0,
    freq_hz: float = 440.0,
    sample_rate: int = 8000,
    amplitude: float = 0.6,
) -> bytes:
    """A mono 16-bit PCM sine tone, for the Telnyx audio-path probe.

    Bypasses LLM/TTS entirely: if the caller hears this tone, the playback path
    (serializer + Telnyx codec/target-leg/stream_id) works and the bug is in the
    TTS/timing path. If the caller hears silence, the bug is in the Telnyx
    payload/codec/target-leg setup itself.
    """
    total = max(1, int(seconds * sample_rate))
    peak = int(max(0.0, min(1.0, amplitude)) * 32767)
    samples = bytearray()
    for n in range(total):
        value = int(peak * math.sin(2.0 * math.pi * freq_hz * (n / sample_rate)))
        samples += struct.pack("<h", value)
    return bytes(samples)


def chunk_pcm(audio: bytes, frame_bytes: int) -> list[bytes]:
    """Split PCM bytes into fixed-size frames (last frame may be short)."""
    if frame_bytes <= 0:
        return [audio] if audio else []
    return [audio[i : i + frame_bytes] for i in range(0, len(audio), frame_bytes)]


def _b64_decoded_len(payload_b64: str) -> int:
    """Decoded byte length of a base64 payload without allocating the bytes."""
    if not payload_b64:
        return 0
    padding = payload_b64.count("=", -2)
    return (len(payload_b64) * 3) // 4 - padding


class TelnyxWsEventLog:
    """Append-only JSONL log of Telnyx websocket activity for one call.

    High-frequency media frames are aggregated into periodic summary lines
    (every `media_summary_every` frames) instead of one line per ~20ms frame,
    so a normal call produces a readable log rather than thousands of lines.
    Low-frequency control events (errors, marks, start/stop) are logged verbatim.
    """

    def __init__(
        self,
        scenario_id: int,
        call_sid: str,
        *,
        started_at: float | None = None,
        media_summary_every: int = 100,
    ):
        self.scenario_id = scenario_id
        self.call_sid = call_sid
        self._t0 = started_at if started_at is not None else time.monotonic()
        self._media_summary_every = max(1, media_summary_every)
        run_dir = Path("runs") / f"scenario_{scenario_id:02d}_{call_sid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._path = run_dir / "telnyx_ws_events.jsonl"

        # Outbound media aggregation.
        self.outbound_media_frames = 0
        self.outbound_media_bytes = 0
        self._since_summary = 0
        # Inbound media aggregation (we don't log every inbound frame either).
        self.inbound_media_frames = 0
        self._finalized = False

    def _elapsed_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def _append(self, record: dict[str, Any]) -> None:
        record = {"t_ms": self._elapsed_ms(), **record}
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception as exc:  # never let instrumentation break a call
            print(f"[DEBUG] telnyx ws event log write failed: {exc}")

    def record(self, direction: str, event: str, **fields: Any) -> None:
        self._append({"direction": direction, "event": event, **fields})

    def record_outbound_media(self, *, byte_len: int, codec: str, has_stream_id: bool) -> None:
        self.outbound_media_frames += 1
        self.outbound_media_bytes += byte_len
        self._since_summary += 1
        # Always log the very first frame so we can confirm sending started.
        if self.outbound_media_frames == 1:
            self._append(
                {
                    "direction": "out",
                    "event": "media_first",
                    "byte_len": byte_len,
                    "codec": codec,
                    "has_stream_id": has_stream_id,
                    "stream_id_status": "present" if has_stream_id else "MISSING",
                }
            )
            return
        if self._since_summary >= self._media_summary_every:
            self._flush_media_summary(codec=codec, has_stream_id=has_stream_id)

    def _flush_media_summary(self, *, codec: str, has_stream_id: bool) -> None:
        self._append(
            {
                "direction": "out",
                "event": "media_summary",
                "frames_total": self.outbound_media_frames,
                "bytes_total": self.outbound_media_bytes,
                "codec": codec,
                "has_stream_id": has_stream_id,
            }
        )
        self._since_summary = 0

    def record_inbound_media(self) -> None:
        self.inbound_media_frames += 1

    def finalize(self) -> None:
        """Write a closing tally so totals are visible without summing lines."""
        if self._finalized:
            return
        self._finalized = True
        self._append(
            {
                "direction": "summary",
                "event": "final",
                "outbound_media_frames": self.outbound_media_frames,
                "outbound_media_bytes": self.outbound_media_bytes,
                "inbound_media_frames": self.inbound_media_frames,
            }
        )


# Telnyx control events that are interesting and low-frequency enough to log
# verbatim. (Inbound `media` is aggregated separately; `dtmf` is rare but small.)
_LOGGED_INBOUND_EVENTS = {
    "error",
    "mark",
    "start",
    "stop",
    "connected",
    "stream_started",
    "streaming.started",
    "dtmf",
}


def inject_stream_id(serialized: str, stream_id: str) -> tuple[str, bool]:
    """Add `stream_id` to a serialized Telnyx media/clear message.

    Returns (possibly-rewritten payload, whether a stream_id is now present).
    Non-JSON or non-dict payloads are returned untouched.
    """
    try:
        message = json.loads(serialized)
    except (ValueError, TypeError):
        return serialized, False
    if not isinstance(message, dict):
        return serialized, False
    if "stream_id" not in message:
        message["stream_id"] = stream_id
    return json.dumps(message), bool(message.get("stream_id"))


def classify_outbound(serialized: str | bytes | None) -> dict[str, Any] | None:
    """Inspect a serialized outbound payload for logging.

    Returns a dict with `event`, and for media: `byte_len`, `has_stream_id`.
    Returns None if there is nothing meaningful to record.
    """
    if not serialized or isinstance(serialized, bytes):
        return None
    try:
        message = json.loads(serialized)
    except (ValueError, TypeError):
        return None
    if not isinstance(message, dict):
        return None
    event = message.get("event")
    if event == "media":
        payload = (message.get("media") or {}).get("payload", "")
        return {
            "event": "media",
            "byte_len": _b64_decoded_len(payload),
            "has_stream_id": bool(message.get("stream_id")),
        }
    return {"event": event or "unknown"}


class InstrumentedTelnyxSerializer(TelnyxFrameSerializer):
    """Telnyx serializer that logs traffic and can inject a missing stream_id."""

    def __init__(
        self,
        *,
        stream_id: str,
        outbound_encoding: str,
        inbound_encoding: str,
        event_log: TelnyxWsEventLog,
        include_stream_id: bool,
        timeline: Any = None,
    ):
        super().__init__(
            stream_id=stream_id,
            outbound_encoding=outbound_encoding,
            inbound_encoding=inbound_encoding,
        )
        self._event_log = event_log
        self._include_stream_id = include_stream_id
        self._codec = outbound_encoding
        self._timeline = timeline

    async def serialize(self, frame):
        payload = await super().serialize(frame)
        if payload is None or isinstance(payload, bytes):
            return payload

        if self._include_stream_id:
            payload, _ = inject_stream_id(payload, self._stream_id)

        info = classify_outbound(payload)
        if info is not None:
            if info["event"] == "media":
                if self._timeline is not None and self._event_log.outbound_media_frames == 0:
                    self._timeline.mark("first_serialized_media")
                self._event_log.record_outbound_media(
                    byte_len=info["byte_len"],
                    codec=self._codec,
                    has_stream_id=info["has_stream_id"],
                )
            else:
                self._event_log.record("out", info["event"])
        return payload

    async def deserialize(self, data):
        try:
            message = json.loads(data)
            event = message.get("event")
            if event == "media":
                self._event_log.record_inbound_media()
            elif event in _LOGGED_INBOUND_EVENTS:
                # Log control frames verbatim — this is where Telnyx error and
                # mark frames surface, which the stock serializer drops.
                self._event_log.record("in", event, payload=message.get(event) or message)
                if self._timeline is not None and event in {"error", "mark", "stop"}:
                    self._timeline.mark(f"telnyx_{event}")
        except Exception as exc:
            print(f"[DEBUG] telnyx deserialize peek failed: {exc}")
        return await super().deserialize(data)
