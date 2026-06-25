from __future__ import annotations

import json
import shutil
import time
import types
from pathlib import Path

import pytest

import src.recorder as recorder
from src.live_audio_recorder import LiveAudioRecorder
from src.recorder import get_run_directory, update_call_meta


def test_recording_download_url_swaps_json_for_mp3():
    url = recorder._recording_download_url("/2010-04-01/Accounts/AC/Recordings/RE1.json")
    assert url == "https://api.twilio.com/2010-04-01/Accounts/AC/Recordings/RE1.mp3"


class _FakeRecordingCtx:
    def __init__(self, store):
        self._store = store

    def fetch(self):
        return types.SimpleNamespace(uri="/2010-04-01/Accounts/AC/Recordings/RE1.json")

    def delete(self):
        self._store["deleted"] = True


class _FakeTwilio:
    def __init__(self, store):
        self._store = store

    def recordings(self, sid):
        self._store["fetched_sid"] = sid
        return _FakeRecordingCtx(self._store)


def _fake_httpx_client(content=b"MP3DATA", raise_exc=None):
    class _Resp:
        def __init__(self):
            self.content = content

        def raise_for_status(self):
            if raise_exc:
                raise raise_exc

    class _Client:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, auth=None):
            return _Resp()

    return _Client


def test_download_recording_success(fake_env, isolated_cwd, monkeypatch):
    store: dict = {}
    monkeypatch.setattr(recorder, "_twilio_client", lambda: _FakeTwilio(store))
    monkeypatch.setattr(recorder.httpx, "Client", _fake_httpx_client())

    path = recorder.download_recording("RE1", "CA1", 1)

    assert path is not None
    written = Path(path)
    assert written.read_bytes() == b"MP3DATA"
    assert store["fetched_sid"] == "RE1"
    assert store["deleted"] is True


def test_download_recording_failure_retries_and_writes_error(fake_env, isolated_cwd, monkeypatch):
    monkeypatch.setattr(recorder, "_twilio_client", lambda: _FakeTwilio({}))
    monkeypatch.setattr(
        recorder.httpx, "Client", _fake_httpx_client(raise_exc=RuntimeError("download 500"))
    )
    sleeps: list = []
    monkeypatch.setattr(recorder.time, "sleep", lambda s: sleeps.append(s))

    result = recorder.download_recording("RE1", "CA1", 1)

    assert result is None
    assert len(sleeps) == 2  # retried 3 times, slept between the first two failures
    err = isolated_cwd / "runs" / "scenario_01_CA1" / "recording_error.txt"
    assert "download 500" in err.read_text(encoding="utf-8")


def test_get_run_directory_creates_expected_path(isolated_cwd):
    run_dir = Path(get_run_directory(1, "CA123"))

    assert run_dir == Path("runs") / "scenario_01_CA123"
    assert run_dir.exists()


def test_update_call_meta_creates_metadata_file(isolated_cwd):
    meta_path = Path(
        update_call_meta(
            1,
            "CA123",
            started_at="2026-06-23T00:00:00+00:00",
            recording_sid="RE123",
            extra={"provider": "test"},
        )
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["scenario_id"] == 1
    assert meta["scenario_name"] == "Basic Appointment Scheduling"
    assert meta["call_sid"] == "CA123"
    assert meta["started_at"] == "2026-06-23T00:00:00+00:00"
    assert meta["recording_sid"] == "RE123"
    assert meta["provider"] == "test"


def test_update_call_meta_merges_existing_file(isolated_cwd):
    update_call_meta(1, "CA123", recording_sid="RE123", extra={"first": True})
    meta_path = Path(
        update_call_meta(
            1,
            "CA123",
            ended_at="2026-06-23T00:01:00+00:00",
            duration_seconds=60,
            extra={"second": True},
        )
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["recording_sid"] == "RE123"
    assert meta["ended_at"] == "2026-06-23T00:01:00+00:00"
    assert meta["duration_seconds"] == 60
    assert meta["first"] is True
    assert meta["second"] is True


def test_live_audio_recorder_exports_mp3_and_metadata(isolated_cwd):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for MP3 export")

    live = LiveAudioRecorder(1, "LIVE123", started_at=time.monotonic())
    silence = b"\x00\x00" * 800

    live.write_agent(silence, sample_rate=8000, num_channels=1)
    live.write_patient(silence, sample_rate=8000, num_channels=1)
    mp3_path = live.close()

    assert mp3_path is not None
    mp3 = Path(mp3_path)
    assert mp3.name == "recording.mp3"
    assert mp3.exists()
    assert mp3.stat().st_size > 0

    run_dir = isolated_cwd / "runs" / "scenario_01_LIVE123"
    assert (run_dir / "live_agent.wav").exists()
    assert (run_dir / "live_patient.wav").exists()
    meta = json.loads((run_dir / "call_meta.json").read_text(encoding="utf-8"))
    assert meta["recording_source"] == "live_websocket"
    assert meta["recording_path"].endswith("recording.mp3")


def test_live_audio_recorder_writes_error_when_no_audio(isolated_cwd):
    live = LiveAudioRecorder(1, "EMPTY123", started_at=time.monotonic())

    assert live.close() is None

    err = isolated_cwd / "runs" / "scenario_01_EMPTY123" / "live_recording_error.txt"
    assert "No live audio frames" in err.read_text(encoding="utf-8")
