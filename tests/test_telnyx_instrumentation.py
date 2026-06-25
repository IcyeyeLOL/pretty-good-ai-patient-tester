import asyncio
import json

import pytest

from src.caller import build_telnyx_stream_params
from src.telnyx_instrumentation import (
    InstrumentedTelnyxSerializer,
    TelnyxWsEventLog,
    chunk_pcm,
    classify_outbound,
    generate_probe_tone,
    inject_stream_id,
)


# ── inject_stream_id ──────────────────────────────────────────────────────────

def test_inject_stream_id_adds_when_missing():
    payload = json.dumps({"event": "media", "media": {"payload": "abc"}})
    out, present = inject_stream_id(payload, "stream-123")
    assert present
    assert json.loads(out)["stream_id"] == "stream-123"


def test_inject_stream_id_preserves_existing():
    payload = json.dumps({"event": "media", "stream_id": "orig", "media": {}})
    out, present = inject_stream_id(payload, "stream-123")
    assert present
    assert json.loads(out)["stream_id"] == "orig"


def test_inject_stream_id_ignores_non_json():
    out, present = inject_stream_id("not json", "stream-123")
    assert out == "not json"
    assert present is False


# ── classify_outbound ─────────────────────────────────────────────────────────

def test_classify_outbound_media_reports_bytes_and_stream_id():
    payload = json.dumps({"event": "media", "stream_id": "s", "media": {"payload": "AAAA"}})
    info = classify_outbound(payload)
    assert info["event"] == "media"
    assert info["has_stream_id"] is True
    assert info["byte_len"] == 3  # "AAAA" -> 3 decoded bytes


def test_classify_outbound_media_missing_stream_id():
    payload = json.dumps({"event": "media", "media": {"payload": ""}})
    info = classify_outbound(payload)
    assert info["has_stream_id"] is False
    assert info["byte_len"] == 0


def test_classify_outbound_clear_event():
    assert classify_outbound(json.dumps({"event": "clear"}))["event"] == "clear"


def test_classify_outbound_none_for_empty_or_bytes():
    assert classify_outbound(None) is None
    assert classify_outbound(b"binary") is None


# ── probe tone ────────────────────────────────────────────────────────────────

def test_generate_probe_tone_length_matches_duration():
    pcm = generate_probe_tone(seconds=1.0, sample_rate=8000)
    assert len(pcm) == 8000 * 2  # 16-bit mono


def test_generate_probe_tone_is_not_silence():
    assert any(b != 0 for b in generate_probe_tone(seconds=0.5))


def test_chunk_pcm_splits_evenly_with_short_tail():
    chunks = chunk_pcm(b"x" * 970, 320)
    assert [len(c) for c in chunks] == [320, 320, 320, 10]


def test_chunk_pcm_empty():
    assert chunk_pcm(b"", 320) == []


# ── build_telnyx_stream_params ────────────────────────────────────────────────

def test_stream_params_defaults():
    # Minimal known-good baseline: the extra bidirectional knobs caused Telnyx
    # 90046 when sent together, so by default we send only these three.
    params = build_telnyx_stream_params(env={})
    assert params["stream_bidirectional_mode"] == "rtp"
    assert params["stream_bidirectional_codec"] == "PCMU"
    assert "stream_bidirectional_target_legs" not in params
    assert "stream_bidirectional_sampling_rate" not in params
    assert "stream_establish_before_call_originate" not in params
    assert "send_silence_when_idle" not in params


def test_stream_params_env_overrides():
    env = {
        "TELNYX_STREAM_BIDIRECTIONAL_TARGET_LEGS": "opposite",
        "TELNYX_STREAM_BIDIRECTIONAL_SAMPLING_RATE": "16000",
        "TELNYX_SEND_SILENCE_WHEN_IDLE": "false",
        "TELNYX_STREAM_CODEC": "pcma",
    }
    params = build_telnyx_stream_params(env=env)
    assert params["stream_bidirectional_target_legs"] == "opposite"
    assert params["stream_bidirectional_sampling_rate"] == 16000
    assert params["send_silence_when_idle"] is False
    assert params["stream_bidirectional_codec"] == "PCMA"


# ── TelnyxWsEventLog ──────────────────────────────────────────────────────────

def _read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_event_log_logs_first_media_and_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = TelnyxWsEventLog(1, "callX", started_at=0.0, media_summary_every=3)
    for _ in range(7):
        log.record_outbound_media(byte_len=160, codec="PCMU", has_stream_id=True)
    log.finalize()

    events = _read_lines(log._path)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "media_first"
    assert events[0]["stream_id_status"] == "present"
    assert "media_summary" in kinds
    final = events[-1]
    assert final["event"] == "final"
    assert final["outbound_media_frames"] == 7
    assert final["outbound_media_bytes"] == 7 * 160


def test_event_log_finalize_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = TelnyxWsEventLog(2, "callY", started_at=0.0)
    log.finalize()
    log.finalize()
    finals = [e for e in _read_lines(log._path) if e["event"] == "final"]
    assert len(finals) == 1


def test_event_log_records_control_event(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = TelnyxWsEventLog(3, "callZ", started_at=0.0)
    log.record("in", "error", payload={"code": "invalid_media"})
    events = _read_lines(log._path)
    assert events[0]["direction"] == "in"
    assert events[0]["event"] == "error"
    assert events[0]["payload"]["code"] == "invalid_media"


# ── InstrumentedTelnyxSerializer ──────────────────────────────────────────────

class _Frame:
    """Minimal stand-in; serializer.serialize only checks isinstance(AudioRawFrame)."""


def test_serializer_injects_stream_id_and_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = TelnyxWsEventLog(4, "callS", started_at=0.0)
    ser = InstrumentedTelnyxSerializer(
        stream_id="STREAM-9",
        outbound_encoding="PCMU",
        inbound_encoding="PCMU",
        event_log=log,
        include_stream_id=True,
    )

    async def fake_serialize(self, frame):
        return json.dumps({"event": "media", "media": {"payload": "AAAA"}})

    monkeypatch.setattr(
        "src.telnyx_instrumentation.TelnyxFrameSerializer.serialize", fake_serialize
    )

    out = asyncio.run(ser.serialize(_Frame()))
    assert json.loads(out)["stream_id"] == "STREAM-9"
    assert log.outbound_media_frames == 1


def test_serializer_logs_inbound_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = TelnyxWsEventLog(5, "callD", started_at=0.0)
    ser = InstrumentedTelnyxSerializer(
        stream_id="S",
        outbound_encoding="PCMU",
        inbound_encoding="PCMU",
        event_log=log,
        include_stream_id=False,
    )

    async def fake_deserialize(self, data):
        return None

    monkeypatch.setattr(
        "src.telnyx_instrumentation.TelnyxFrameSerializer.deserialize", fake_deserialize
    )

    asyncio.run(ser.deserialize(json.dumps({"event": "error", "payload": {"code": "x"}})))
    events = _read_lines(log._path)
    assert any(e["event"] == "error" and e["direction"] == "in" for e in events)
