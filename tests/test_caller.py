from __future__ import annotations

import base64
import json

import httpx
import pytest


# ─── Phone helpers ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("+1 (805) 439-8008", "+18054398008"),
        ("+1-805-439-8008", "+18054398008"),
        ("  +18054398008  ", "+18054398008"),
    ],
)
def test_normalize_phone_strips_formatting(raw, expected):
    from src.caller import _normalize_phone

    assert _normalize_phone(raw) == expected


@pytest.mark.parametrize("good", ["+18054398008", "+447911123456", "+15551234567"])
def test_require_e164_accepts_valid(good):
    from src.caller import _require_e164

    assert _require_e164("X", good) == good


@pytest.mark.parametrize("bad", ["8054398008", "+0123", "not-a-number", "", "+1abc5551234"])
def test_require_e164_rejects_invalid(bad):
    from src.caller import _require_e164

    with pytest.raises(ValueError):
        _require_e164("VERIFIED_TEST_PHONE", bad)


def test_mask_phone_hides_middle_digits():
    from src.caller import _mask_phone

    masked = _mask_phone("+18054398008")
    assert masked.startswith("+18")
    assert masked.endswith("8008")
    assert "43989" not in masked


def test_mask_phone_short_value():
    from src.caller import _mask_phone

    assert _mask_phone("12345") == "***"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("call-abc_123.def", "call-abc_123.def"),
        ("weird/value with spaces", "weird_value_with_spaces"),
        ("a@b#c", "a_b_c"),
    ],
)
def test_safe_identifier(raw, expected):
    from src.caller import _safe_identifier

    assert _safe_identifier(raw) == expected


# ─── Provider selection ───────────────────────────────────────────────────────

def test_telephony_provider_defaults_to_twilio(fake_env, monkeypatch):
    from src.caller import telephony_provider

    monkeypatch.delenv("TELEPHONY_PROVIDER", raising=False)
    assert telephony_provider() == "twilio"


def test_telephony_provider_rejects_unknown(fake_env, monkeypatch):
    from src.caller import telephony_provider

    monkeypatch.setenv("TELEPHONY_PROVIDER", "vonage")
    with pytest.raises(ValueError):
        telephony_provider()


# ─── Target phone lock ────────────────────────────────────────────────────────

def test_target_phone_returns_locked_assessment_number(fake_env):
    from src.caller import TARGET_NUMBER, _target_phone

    assert _target_phone() == TARGET_NUMBER


def test_target_phone_rejects_tampered_lock(fake_env, monkeypatch):
    from src.caller import _target_phone

    monkeypatch.setenv("TARGET_PHONE", "+15550009999")
    with pytest.raises(ValueError):
        _target_phone()


def test_target_phone_test_target_uses_verified_number(fake_env):
    from src.caller import _target_phone

    assert _target_phone(use_test_target=True) == "+15550000003"


def test_target_phone_test_target_must_differ_from_assessment(fake_env, monkeypatch):
    from src.caller import TARGET_NUMBER, _target_phone

    monkeypatch.setenv("VERIFIED_TEST_PHONE", TARGET_NUMBER)
    with pytest.raises(ValueError):
        _target_phone(use_test_target=True)


# ─── Telnyx client_state round-trips with the server decoder ──────────────────

def test_telnyx_client_state_round_trip():
    from src.caller import _telnyx_client_state
    from src.server import _scenario_from_client_state

    encoded = _telnyx_client_state(7)
    # Sanity: it is real base64 of JSON
    assert json.loads(base64.b64decode(encoded))["scenario_id"] == 7
    # The server must be able to decode what the caller encodes
    assert _scenario_from_client_state(encoded) == 7


def test_scenario_from_client_state_handles_garbage():
    from src.server import _scenario_from_client_state

    assert _scenario_from_client_state(None) is None
    assert _scenario_from_client_state("!!!not-base64!!!") is None
    assert _scenario_from_client_state(base64.b64encode(b"not json").decode()) is None


# ─── Telnyx outbound call construction ────────────────────────────────────────

def _telnyx_env(monkeypatch):
    monkeypatch.setenv("TELEPHONY_PROVIDER", "telnyx")
    monkeypatch.setenv("TELNYX_CONNECTION_ID", "conn-123")
    monkeypatch.setenv("TELNYX_PHONE_NUMBER", "+15550000002")
    monkeypatch.setenv("TELNYX_API_KEY", "fake-telnyx-key")


def test_make_telnyx_call_posts_expected_payload(fake_env, isolated_cwd, monkeypatch):
    import src.caller as caller

    _telnyx_env(monkeypatch)
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return httpx.Response(
            200,
            json={"data": {"call_session_id": "sess-abc", "call_control_id": "ctrl-1"}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(caller.httpx, "post", fake_post)

    call_id = caller._make_telnyx_call(3, "+18054398008")

    assert call_id == "sess-abc"
    assert captured["url"] == "https://api.telnyx.com/v2/calls"
    assert captured["headers"]["Authorization"] == "Bearer fake-telnyx-key"
    payload = captured["json"]
    assert payload["connection_id"] == "conn-123"
    assert payload["to"] == "+18054398008"
    assert payload["from"] == "+15550000002"
    assert "/telnyx/ws/3" in payload["stream_url"]
    assert payload["stream_url"].startswith("wss://")
    assert "scenario_id=3" in payload["webhook_url"]
    # client_state must decode back to the scenario
    assert json.loads(base64.b64decode(payload["client_state"]))["scenario_id"] == 3
    # And a meta file should have been written for the new call id
    meta = list((isolated_cwd / "runs").glob("scenario_03_*/call_meta.json"))
    assert meta, "expected call_meta.json to be written for the telnyx call"


def test_make_telnyx_call_raises_on_api_error(fake_env, isolated_cwd, monkeypatch):
    import src.caller as caller

    _telnyx_env(monkeypatch)

    def fake_post(url, headers=None, json=None, timeout=None):
        return httpx.Response(422, text="bad request", request=httpx.Request("POST", url))

    monkeypatch.setattr(caller.httpx, "post", fake_post)

    with pytest.raises(RuntimeError, match="Telnyx call failed"):
        caller._make_telnyx_call(3, "+18054398008")


def test_make_call_dispatches_to_telnyx(fake_env, isolated_cwd, monkeypatch):
    import src.caller as caller

    _telnyx_env(monkeypatch)
    monkeypatch.setattr(caller, "_make_telnyx_call", lambda sid, target: "telnyx-call-id")
    monkeypatch.setattr(caller, "_make_twilio_call", lambda sid, target: pytest.fail("twilio used"))

    assert caller.make_call(1) == "telnyx-call-id"


def test_make_call_dispatches_to_twilio(fake_env, isolated_cwd, monkeypatch):
    import src.caller as caller

    monkeypatch.setenv("TELEPHONY_PROVIDER", "twilio")
    monkeypatch.setattr(caller, "_make_twilio_call", lambda sid, target: "twilio-call-id")
    monkeypatch.setattr(caller, "_make_telnyx_call", lambda sid, target: pytest.fail("telnyx used"))

    assert caller.make_call(1) == "twilio-call-id"


# ─── Telnyx status derived from call_meta ─────────────────────────────────────

@pytest.mark.parametrize(
    "telnyx_status, expected",
    [
        ("call.hangup", "completed"),
        ("call.initiated", "initiated"),
        ("call.answered", "in-progress"),
        ("streaming.started", "in-progress"),
    ],
)
def test_get_telnyx_status_from_meta(fake_env, isolated_cwd, monkeypatch, telnyx_status, expected):
    import src.caller as caller
    from src.recorder import update_call_meta

    monkeypatch.setenv("TELEPHONY_PROVIDER", "telnyx")
    update_call_meta(4, "sess-xyz", extra={"telnyx_status": telnyx_status})

    assert caller.get_call_status("sess-xyz") == expected


def test_get_telnyx_status_defaults_to_initiated_when_missing(fake_env, isolated_cwd, monkeypatch):
    import src.caller as caller

    monkeypatch.setenv("TELEPHONY_PROVIDER", "telnyx")
    assert caller.get_call_status("nonexistent") == "initiated"
