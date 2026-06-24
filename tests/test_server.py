from __future__ import annotations

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
