from __future__ import annotations

import json

from fastapi.testclient import TestClient


def test_ping(fake_env, isolated_cwd):
    from src.server import app

    with TestClient(app) as client:
        response = client.get("/api/ping")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_scenarios_endpoint_returns_all_scenarios(fake_env, isolated_cwd):
    from src.server import app

    with TestClient(app) as client:
        response = client.get("/api/scenarios")

    scenarios = response.json()
    assert response.status_code == 200
    assert len(scenarios) == 10
    assert {"id", "name", "opening_line", "severity"}.issubset(scenarios[0])


def test_twiml_endpoint_returns_stream_xml(fake_env, isolated_cwd):
    from src.server import app

    with TestClient(app) as client:
        response = client.post(
            "/twiml/1",
            content="CallSid=CA123",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )

    assert response.status_code == 200
    assert "<Stream" in response.text
    assert "wss://test.ngrok-free.app/ws/1" in response.text
    assert 'Parameter name="call_sid" value="CA123"' in response.text


def test_twiml_unknown_scenario_hangs_up(fake_env, isolated_cwd):
    from src.server import app

    with TestClient(app) as client:
        response = client.post("/twiml/99")

    assert response.status_code == 404
    assert "<Hangup" in response.text


def test_all_twiml_scenarios_return_200(fake_env, isolated_cwd):
    from src.server import app

    with TestClient(app) as client:
        for scenario_id in range(1, 11):
            response = client.post(
                f"/twiml/{scenario_id}",
                content=f"CallSid=CA{scenario_id}",
                headers={"content-type": "application/x-www-form-urlencoded"},
            )
            assert response.status_code == 200


def test_recording_complete_missing_fields_returns_400(fake_env, isolated_cwd):
    from src.server import app

    with TestClient(app) as client:
        response = client.post(
            "/recording-complete?scenario_id=1",
            content="CallSid=CA123",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )

    assert response.status_code == 400


def test_call_status_updates_status_endpoint(fake_env, isolated_cwd):
    from src import server

    server.ACTIVE_CALLS.clear()
    server.CALL_STATUS.clear()
    with TestClient(server.app) as client:
        response = client.post(
            "/call-status?scenario_id=1",
            content="CallSid=CA123&CallStatus=completed&CallDuration=42",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        status_response = client.get("/api/status/CA123")

    assert response.status_code == 200
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["call_sid"] == "CA123"
    assert body["status"] == "completed"
    assert body["twilio_status"] == "completed"


def _write_run(cwd, scenario_id, call_sid, *, judge=None, transcript=None, meta=None):
    run_dir = cwd / "runs" / f"scenario_{scenario_id:02d}_{call_sid}"
    run_dir.mkdir(parents=True, exist_ok=True)
    base_meta = {
        "scenario_id": scenario_id,
        "scenario_name": f"Scenario {scenario_id}",
        "call_sid": call_sid,
        "started_at": "2026-06-23T10:00:00+00:00",
    }
    base_meta.update(meta or {})
    (run_dir / "call_meta.json").write_text(json.dumps(base_meta), encoding="utf-8")
    if judge is not None:
        (run_dir / "judge_output.json").write_text(json.dumps(judge), encoding="utf-8")
    if transcript is not None:
        (run_dir / "transcript_live.json").write_text(json.dumps(transcript), encoding="utf-8")
    return run_dir


def test_status_under_telnyx_uses_generic_status_field(fake_env, isolated_cwd, monkeypatch):
    """The dev UI keys off `status`; under Telnyx, twilio_status must stay null."""
    from src import server

    monkeypatch.setenv("TELEPHONY_PROVIDER", "telnyx")
    server.ACTIVE_CALLS.clear()
    server.CALL_STATUS.clear()
    server.CALL_STATUS["sess-1"] = "completed"
    server.ACTIVE_CALLS["sess-1"] = 2

    with TestClient(server.app) as client:
        body = client.get("/api/status/sess-1").json()

    assert body["status"] == "completed"
    assert body["telnyx_status"] == "completed"
    assert body["twilio_status"] is None


def test_status_includes_judge_and_transcript_when_present(fake_env, isolated_cwd):
    from src import server

    server.ACTIVE_CALLS.clear()
    server.CALL_STATUS.clear()
    server.ACTIVE_CALLS["CA9"] = 1
    server.CALL_STATUS["CA9"] = "completed"
    _write_run(
        isolated_cwd, 1, "CA9",
        judge={"verdict": "BUG", "scenario_id": 1},
        transcript={"turns": [{"speaker": "patient", "text": "hi"}]},
    )

    with TestClient(server.app) as client:
        body = client.get("/api/status/CA9").json()

    assert body["judge"]["verdict"] == "BUG"
    assert body["transcript"]["turns"][0]["text"] == "hi"


def test_scenarios_endpoint_surfaces_last_verdict(fake_env, isolated_cwd):
    from src.server import app

    _write_run(isolated_cwd, 4, "CAlast", judge={"verdict": "WEAKNESS", "scenario_id": 4})

    with TestClient(app) as client:
        scenarios = client.get("/api/scenarios").json()

    by_id = {s["id"]: s for s in scenarios}
    assert by_id[4]["last_verdict"] == "WEAKNESS"
    assert by_id[1]["last_verdict"] is None


def test_runs_endpoint_lists_and_filters(fake_env, isolated_cwd):
    from src.server import app

    _write_run(isolated_cwd, 1, "CAa", judge={"verdict": "PASS"})
    _write_run(isolated_cwd, 2, "CAb", judge={"verdict": "BUG"})

    with TestClient(app) as client:
        all_runs = client.get("/api/runs").json()
        filtered = client.get("/api/runs?scenario_id=2").json()

    assert len(all_runs) == 2
    assert len(filtered) == 1
    assert filtered[0]["scenario_id"] == 2
    assert filtered[0]["judge"]["verdict"] == "BUG"


def test_report_get_404_then_generate(fake_env, isolated_cwd, monkeypatch):
    from src import server

    # No report yet
    with TestClient(server.app) as client:
        assert client.get("/api/report").status_code == 404

    _write_run(
        isolated_cwd, 2, "CAbug",
        judge={
            "verdict": "BUG",
            "scenario_id": 2,
            "scenario_name": "Weekend Hours Hallucination",
            "severity": "high",
            "summary": "Booked a Saturday slot.",
            "evidence": "Sure, Saturday at 10 works.",
            "expected_behavior": "Decline weekend booking.",
            "call_sid": "CAbug",
        },
    )

    with TestClient(server.app) as client:
        gen = client.post("/api/report").json()
        saved = client.get("/api/report").json()

    assert "Weekend Hours Hallucination" in gen["content"]
    assert gen["content"] == saved["content"]


def test_run_endpoint_places_call(fake_env, isolated_cwd, monkeypatch):
    import src.caller as caller
    from src import server

    monkeypatch.setattr(
        caller, "make_call", lambda scenario_id, use_test_target=False: "CAplaced"
    )

    with TestClient(server.app) as client:
        body = client.post("/api/run/1").json()

    assert body["call_sid"] == "CAplaced"
    assert body["scenario_id"] == 1


def test_run_endpoint_passes_test_target(fake_env, isolated_cwd, monkeypatch):
    import src.caller as caller
    from src import server

    captured = {}

    def fake_make_call(scenario_id, use_test_target=False):
        captured["use_test_target"] = use_test_target
        return "CAtest"

    monkeypatch.setattr(caller, "make_call", fake_make_call)

    with TestClient(server.app) as client:
        body = client.post("/api/run/1?test_target=true").json()

    assert body["call_sid"] == "CAtest"
    assert captured["use_test_target"] is True


def test_reset_archives_runs_and_clears_report(fake_env, isolated_cwd):
    from pathlib import Path

    from src import server

    run_dir = Path("runs") / "scenario_02_RESETME"
    run_dir.mkdir(parents=True)
    (run_dir / "judge_output.json").write_text('{"verdict": "BUG"}', encoding="utf-8")
    report = Path("reports")
    report.mkdir(exist_ok=True)
    (report / "bug_report.md").write_text("# Bug Report\n", encoding="utf-8")
    server.ACTIVE_CALLS["RESETME"] = 2
    server.CALL_STATUS["RESETME"] = "completed"

    with TestClient(server.app) as client:
        body = client.post("/api/reset").json()

    assert body["ok"] is True
    assert body["archived_runs"] == 1
    assert not run_dir.exists()
    assert not (report / "bug_report.md").exists()
    assert list(Path("runs_archive").glob("*/scenario_02_RESETME"))
    assert "RESETME" not in server.ACTIVE_CALLS
    assert "RESETME" not in server.CALL_STATUS


def test_rejudge_reruns_judge_on_existing_transcript(fake_env, isolated_cwd, monkeypatch):
    import json as _json
    from pathlib import Path

    from src import judge_bot, server

    monkeypatch.setattr(judge_bot, "_call_judge_llm", lambda _p: '{"verdict": "PASS"}')

    run_dir = Path("runs") / "scenario_01_CArejudge"
    run_dir.mkdir(parents=True)
    (run_dir / "transcript_live.json").write_text(
        _json.dumps(
            {
                "scenario_id": 1,
                "call_sid": "CArejudge",
                "turns": [
                    {"speaker": "agent", "text": "May I have your name?", "start_ms": 0},
                    {"speaker": "patient", "text": "Maria Johnson.", "start_ms": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    with TestClient(server.app) as client:
        resp = client.post("/api/runs/CArejudge/rejudge")

    assert resp.status_code == 200
    assert resp.json()["judge"]["verdict"] == "PASS"
    assert (run_dir / "judge_output.json").exists()


def test_rejudge_unknown_call_404(fake_env, isolated_cwd):
    from src.server import app

    with TestClient(app) as client:
        assert client.post("/api/runs/NOPE/rejudge").status_code == 404


def test_run_endpoint_unknown_scenario_404(fake_env, isolated_cwd):
    from src.server import app

    with TestClient(app) as client:
        assert client.post("/api/run/99").status_code == 404


def test_process_recording_falls_back_to_live_transcript(fake_env, isolated_cwd, monkeypatch):
    """If the recording/AssemblyAI path yields nothing, the live transcript is judged."""
    from src import server

    monkeypatch.setattr(server, "download_recording", lambda *a, **k: None)
    monkeypatch.setattr(server, "transcribe_recording", lambda *a, **k: None)

    _write_run(
        isolated_cwd, 1, "CAfb",
        transcript={"turns": [{"speaker": "patient", "text": "Hi"}], "call_sid": "CAfb"},
    )

    judged = {}
    monkeypatch.setattr(server, "evaluate", lambda transcript, scenario: judged.update(
        {"turns": transcript["turns"], "scenario_id": scenario.id}
    ))

    server.process_recording("RE1", "CAfb", 1)

    assert judged["scenario_id"] == 1
    assert judged["turns"][0]["text"] == "Hi"


def test_process_recording_prefers_final_transcript(fake_env, isolated_cwd, monkeypatch):
    from src import server

    monkeypatch.setattr(server, "download_recording", lambda *a, **k: "rec.mp3")
    monkeypatch.setattr(
        server, "transcribe_recording",
        lambda *a, **k: {"turns": [{"speaker": "agent", "text": "Final"}], "call_sid": "CAf2"},
    )

    judged = {}
    monkeypatch.setattr(server, "evaluate", lambda transcript, scenario: judged.update(
        {"text": transcript["turns"][0]["text"]}
    ))

    server.process_recording("RE2", "CAf2", 1)

    assert judged["text"] == "Final"


def test_telnyx_events_marks_completed_on_hangup(fake_env, isolated_cwd, monkeypatch):
    from src import server

    server.ACTIVE_CALLS.clear()
    server.CALL_STATUS.clear()
    payload = {
        "data": {
            "event_type": "call.hangup",
            "payload": {"call_session_id": "sess-7", "hangup_cause": "normal_clearing"},
        }
    }

    with TestClient(server.app) as client:
        resp = client.post("/telnyx/events?scenario_id=2", json=payload)

    assert resp.status_code == 200
    assert server.CALL_STATUS["sess-7"] == "completed"
    assert server.ACTIVE_CALLS["sess-7"] == 2
