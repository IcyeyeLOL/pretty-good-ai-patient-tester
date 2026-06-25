from __future__ import annotations

import json


def test_transcribe_recording_maps_utterances_to_turns(fake_env, isolated_cwd, monkeypatch):
    import src.transcriber as transcriber

    monkeypatch.setattr(transcriber, "_upload_audio", lambda path: "https://upload/x")
    monkeypatch.setattr(transcriber, "_submit_transcript", lambda url: "tid-1")
    monkeypatch.setattr(
        transcriber,
        "_poll_transcript",
        lambda tid: {
            "status": "completed",
            "audio_duration": 73,
            "text": "Hello. Hi there.",
            "utterances": [
                {"speaker": "A", "text": "Hello.", "start": 0, "end": 900},
                {"speaker": "B", "text": "Hi there.", "start": 1000, "end": 1800},
            ],
        },
    )

    rec = isolated_cwd / "rec.mp3"
    rec.write_bytes(b"audio")

    transcript = transcriber.transcribe_recording(str(rec), 1, "CA1")

    assert transcript is not None
    assert transcript["duration_seconds"] == 73
    assert transcript["speaker_A_is_patient"] is True
    assert [t["speaker"] for t in transcript["turns"]] == ["A", "B"]
    assert transcript["turns"][0]["text"] == "Hello."

    # File is persisted for the run
    out = isolated_cwd / "runs" / "scenario_01_CA1" / "transcript_final.json"
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["call_sid"] == "CA1"


def test_transcribe_recording_writes_error_file_on_failure(fake_env, isolated_cwd, monkeypatch):
    import src.transcriber as transcriber

    def boom(path):
        raise RuntimeError("upload exploded")

    monkeypatch.setattr(transcriber, "_upload_audio", boom)

    rec = isolated_cwd / "rec.mp3"
    rec.write_bytes(b"audio")

    result = transcriber.transcribe_recording(str(rec), 2, "CA2")

    assert result is None
    err = isolated_cwd / "runs" / "scenario_02_CA2" / "transcription_error.txt"
    assert err.exists()
    assert "upload exploded" in err.read_text(encoding="utf-8")


def test_save_live_transcript_persists_turns(fake_env, isolated_cwd):
    from src.transcriber import save_live_transcript

    turns = [{"speaker": "patient", "text": "Hi", "start_ms": 0, "end_ms": 0}]
    save_live_transcript(turns, 3, "CA3")

    out = isolated_cwd / "runs" / "scenario_03_CA3" / "transcript_live.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["turns"] == turns
    assert data["scenario_id"] == 3
