import json

from src.startup_timeline import StartupTimeline, compute_startup_deltas


def test_compute_deltas_full_sequence():
    marks = {
        "call_initiated": 1000,
        "call_answered": 4000,
        "streaming_started": 4500,
        "ws_accepted": 4600,
        "ws_start_event": 4700,
        "opener_queued": 5000,
        "first_outbound_audio_before_transport": 5100,
        "first_serialized_media": 5150,
    }
    deltas = compute_startup_deltas(marks)
    assert deltas["initiated_to_answered_ms"] == 3000
    assert deltas["answered_to_streaming_started_ms"] == 500
    assert deltas["streaming_started_to_ws_start_ms"] == 200
    assert deltas["ws_accepted_to_ws_start_ms"] == 100
    # first_bot_audio prefers the serialized media frame.
    assert deltas["ws_start_to_first_outbound_media_ms"] == 5150 - 4700
    assert deltas["total_initiated_to_first_bot_audio_ms_incl_ring"] == 5150 - 1000


def test_compute_deltas_falls_back_to_before_transport_audio():
    marks = {
        "call_initiated": 0,
        "ws_start_event": 100,
        "first_outbound_audio_before_transport": 800,
    }
    deltas = compute_startup_deltas(marks)
    assert deltas["total_initiated_to_first_bot_audio_ms_incl_ring"] == 800
    assert deltas["ws_start_to_first_outbound_media_ms"] == 700


def test_compute_deltas_omits_missing_endpoints():
    deltas = compute_startup_deltas({"call_initiated": 0})
    assert "initiated_to_answered_ms" not in deltas
    assert "total_initiated_to_first_bot_audio_ms_incl_ring" not in deltas


def test_timeline_first_write_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tl = StartupTimeline(1, "abc")
    tl.mark("call_initiated", 1000)
    tl.mark("call_initiated", 9999)  # ignored
    data = json.loads(tl._path.read_text())
    assert data["marks"]["call_initiated"] == 1000


def test_timeline_merges_across_instances_and_recomputes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Simulate different components (threads) writing to the same call file.
    StartupTimeline(1, "abc").mark("call_initiated", 1000)
    StartupTimeline(1, "abc").mark("call_answered", 4000)
    StartupTimeline(1, "abc").mark("first_serialized_media", 5000)

    data = json.loads((tmp_path / "runs" / "scenario_01_abc" / "startup_timeline.json").read_text())
    assert data["marks"] == {
        "call_initiated": 1000,
        "call_answered": 4000,
        "first_serialized_media": 5000,
    }
    assert data["deltas"]["initiated_to_answered_ms"] == 3000
    assert data["deltas"]["total_initiated_to_first_bot_audio_ms_incl_ring"] == 4000
