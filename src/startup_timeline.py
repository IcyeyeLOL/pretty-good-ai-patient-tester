"""Cross-component call-startup timeline.

The existing `LatencyTracker` (pipeline.py) measures *in-app* per-turn latency
using `time.monotonic()` anchored to when the pipeline was built. That cannot
answer the question that matters here: of the ~10s before the caller hears the
bot, how much is Telnyx/PSTN connection, how much is websocket/media-stream
establishment, and how much is our app?

To answer that we need timestamps from components that live in different threads
(the CLI that POSTs the call, the FastAPI webhook handler, the websocket handler,
the pipeline) on a *shared, comparable* clock. So this module records named
wall-clock epoch-millisecond marks into a single per-call file,
`runs/scenario_XX_<call_sid>/startup_timeline.json`, and recomputes the delta
report on every write so the file is ready the moment the call ends.

All marks are keyed by (scenario_id, call_sid); Telnyx's `call_session_id` is the
stable id used everywhere (see `_telnyx_call_id` in server.py and `_make_telnyx_call`
in caller.py), so marks written by different components land in the same file.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

# Writes are infrequent (a handful per call) and span threads; one lock is fine.
_WRITE_LOCK = threading.Lock()


def _now_ms() -> int:
    return int(time.time() * 1000)


# Ordered phases of the startup sequence. Used for both display order and to
# decide which adjacent deltas to compute.
PHASE_ORDER = (
    "call_initiated",
    "call_answered",
    "streaming_started",
    "ws_accepted",
    "ws_start_event",
    "pipeline_build_started",
    "pipeline_built",
    "opener_scheduled",
    "opener_queued",
    "first_outbound_audio_before_transport",
    "first_serialized_media",
)


def compute_startup_deltas(marks: dict[str, int]) -> dict[str, int]:
    """Derive the human-meaningful deltas from raw marks. Pure function.

    Missing endpoints simply omit that delta (no guessing). `first_bot_audio` is
    the first serialized Telnyx media frame if present, else the first outbound
    audio captured before the transport.
    """
    deltas: dict[str, int] = {}

    def diff(key: str, start: str, end: str) -> None:
        if start in marks and end in marks:
            deltas[key] = marks[end] - marks[start]

    diff("initiated_to_answered_ms", "call_initiated", "call_answered")
    diff("answered_to_streaming_started_ms", "call_answered", "streaming_started")
    diff("streaming_started_to_ws_start_ms", "streaming_started", "ws_start_event")
    diff("ws_accepted_to_ws_start_ms", "ws_accepted", "ws_start_event")
    diff("ws_start_to_opener_queued_ms", "ws_start_event", "opener_queued")

    # Caller-relevant timing: what the person on the phone actually experiences,
    # measured from pickup (call_answered) rather than from dial (call_initiated,
    # which includes uncontrollable ring/answer time).
    diff("call_answered_to_opener_scheduled_ms", "call_answered", "opener_scheduled")
    diff(
        "opener_scheduled_to_first_serialized_media_ms",
        "opener_scheduled",
        "first_serialized_media",
    )

    # App cold-start / build cost.
    diff("ws_start_to_pipeline_build_started_ms", "ws_start_event", "pipeline_build_started")
    diff("pipeline_build_duration_ms", "pipeline_build_started", "pipeline_built")

    first_bot_audio = marks.get("first_serialized_media") or marks.get(
        "first_outbound_audio_before_transport"
    )
    if "ws_start_event" in marks and first_bot_audio is not None:
        deltas["ws_start_to_first_outbound_media_ms"] = first_bot_audio - marks["ws_start_event"]
    if "call_answered" in marks and first_bot_audio is not None:
        # The number that matches what the caller hears, post-pickup.
        deltas["call_answered_to_first_bot_audio_ms"] = first_bot_audio - marks["call_answered"]
    if "call_initiated" in marks and first_bot_audio is not None:
        # NOTE: includes ring + answer time, so it is larger than what the caller
        # perceives once they pick up. Use call_answered_to_first_bot_audio_ms for
        # the caller-perceived figure.
        deltas["total_initiated_to_first_bot_audio_ms_incl_ring"] = (
            first_bot_audio - marks["call_initiated"]
        )

    return deltas


class StartupTimeline:
    """Append-only (first-write-wins) timeline persisted to the run dir."""

    def __init__(self, scenario_id: int, call_sid: str):
        self.scenario_id = scenario_id
        self.call_sid = call_sid
        run_dir = Path("runs") / f"scenario_{scenario_id:02d}_{call_sid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._path = run_dir / "startup_timeline.json"

    def mark(self, name: str, when_ms: int | None = None) -> None:
        """Record `name` at `when_ms` (default now). First write wins per name."""
        when_ms = when_ms if when_ms is not None else _now_ms()
        with _WRITE_LOCK:
            data = self._read()
            marks = data.setdefault("marks", {})
            if name in marks:
                return  # first write wins; ignore later duplicates
            marks[name] = when_ms
            data["scenario_id"] = self.scenario_id
            data["call_sid"] = self.call_sid
            data["deltas"] = compute_startup_deltas(marks)
            self._write(data)

    def _read(self) -> dict[str, Any]:
        try:
            if self._path.exists():
                return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[DEBUG] startup timeline read failed: {exc}")
        return {}

    def _write(self, data: dict[str, Any]) -> None:
        try:
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:  # never let instrumentation break a call
            print(f"[DEBUG] startup timeline write failed: {exc}")
