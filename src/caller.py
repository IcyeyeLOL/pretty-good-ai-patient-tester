from __future__ import annotations

import base64
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

TARGET_NUMBER = "+18054398008"
E164_RE = re.compile(r"^\+[1-9]\d{1,14}$")
SUPPORTED_TELEPHONY_PROVIDERS = {"twilio", "telnyx"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _client() -> Client:
    return Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])


def _base_url() -> str:
    return os.environ["BASE_URL"].rstrip("/")


def _websocket_base_url() -> str:
    base_url = _base_url()
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://") :]
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://") :]
    return "wss://" + base_url


def telephony_provider() -> str:
    provider = os.environ.get("TELEPHONY_PROVIDER", "twilio").lower()
    if provider not in SUPPORTED_TELEPHONY_PROVIDERS:
        raise ValueError("TELEPHONY_PROVIDER must be 'twilio' or 'telnyx'.")
    return provider


def _normalize_phone(value: str) -> str:
    return re.sub(r"[\s().-]", "", value.strip())


def _require_e164(name: str, value: str) -> str:
    phone = _normalize_phone(value)
    if not E164_RE.match(phone):
        raise ValueError(f"{name} must be in E.164 format, for example +15551234567.")
    return phone


def _mask_phone(value: str) -> str:
    phone = _normalize_phone(value)
    if len(phone) <= 6:
        return "***"
    return f"{phone[:3]}***{phone[-4:]}"


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _target_phone(use_test_target: bool = False) -> str:
    if use_test_target:
        target = _require_e164("VERIFIED_TEST_PHONE", os.environ.get("VERIFIED_TEST_PHONE", ""))
        if target == TARGET_NUMBER:
            raise ValueError("VERIFIED_TEST_PHONE must be separate from the final assessment TARGET_PHONE.")
        return target

    target = os.environ.get("TARGET_PHONE", TARGET_NUMBER)
    if target != TARGET_NUMBER:
        raise ValueError(f"Refusing to call {target}; TARGET_PHONE must be {TARGET_NUMBER}")
    return target


def _make_twilio_call(scenario_id: int, target: str) -> str:
    callback_query = urlencode({"scenario_id": scenario_id})
    base_url = _base_url()
    call = _client().calls.create(
        to=target,
        from_=os.environ["TWILIO_PHONE_NUMBER"],
        url=f"{base_url}/twiml/{scenario_id}",
        method="POST",
        record=True,
        recording_channels="dual",
        recording_status_callback=f"{base_url}/recording-complete?{callback_query}",
        recording_status_callback_method="POST",
        status_callback=f"{base_url}/call-status?{callback_query}",
        status_callback_method="POST",
        status_callback_event=["initiated", "ringing", "answered", "completed"],
    )
    print(f"Call initiated: SID={call.sid}, Scenario={scenario_id}, Target={_mask_phone(target)}")
    return call.sid


def _telnyx_client_state(scenario_id: int) -> str:
    payload = json.dumps({"scenario_id": scenario_id}, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def _env_bool(name: str, default: bool, env: dict | None = None) -> bool:
    env = env if env is not None else os.environ
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_telnyx_stream_params(env: dict | None = None) -> dict:
    """Telnyx bidirectional-streaming params.

    Defaults to the known-good minimal payload (stream_track + bidirectional
    rtp/codec) that successfully connects and streams. The extra diagnostic knobs
    (target_legs, sampling_rate, establish-before-originate, send_silence) caused
    Telnyx 90046 "Failed to connect to destination" when sent together, so they
    are now *opt-in*: each is only included when its env var is explicitly set.
    """
    env = env if env is not None else os.environ
    codec = env.get("TELNYX_STREAM_BIDIRECTIONAL_CODEC", env.get("TELNYX_STREAM_CODEC", "PCMU")).upper()
    params = {
        "stream_track": env.get("TELNYX_STREAM_TRACK", "inbound_track"),
        "stream_bidirectional_mode": env.get("TELNYX_STREAM_BIDIRECTIONAL_MODE", "rtp"),
        "stream_bidirectional_codec": codec,
    }
    # Opt-in extras — only sent when the operator explicitly sets them.
    if "TELNYX_STREAM_BIDIRECTIONAL_TARGET_LEGS" in env:
        params["stream_bidirectional_target_legs"] = env["TELNYX_STREAM_BIDIRECTIONAL_TARGET_LEGS"]
    if "TELNYX_STREAM_BIDIRECTIONAL_SAMPLING_RATE" in env:
        params["stream_bidirectional_sampling_rate"] = int(
            env["TELNYX_STREAM_BIDIRECTIONAL_SAMPLING_RATE"]
        )
    if "TELNYX_STREAM_ESTABLISH_BEFORE_CALL_ORIGINATE" in env:
        params["stream_establish_before_call_originate"] = _env_bool(
            "TELNYX_STREAM_ESTABLISH_BEFORE_CALL_ORIGINATE", False, env
        )
    if "TELNYX_SEND_SILENCE_WHEN_IDLE" in env:
        params["send_silence_when_idle"] = _env_bool("TELNYX_SEND_SILENCE_WHEN_IDLE", False, env)
    return params


def _make_telnyx_call(scenario_id: int, target: str) -> str:
    base_url = _base_url()
    payload = {
        "connection_id": os.environ["TELNYX_CONNECTION_ID"],
        "to": target,
        "from": _require_e164("TELNYX_PHONE_NUMBER", os.environ["TELNYX_PHONE_NUMBER"]),
        "webhook_url": f"{base_url}/telnyx/events?{urlencode({'scenario_id': scenario_id})}",
        "stream_url": f"{_websocket_base_url()}/telnyx/ws/{scenario_id}",
        "client_state": _telnyx_client_state(scenario_id),
        **build_telnyx_stream_params(),
    }
    headers = {
        "Authorization": f"Bearer {os.environ['TELNYX_API_KEY']}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    initiated_ms = int(time.time() * 1000)
    response = httpx.post("https://api.telnyx.com/v2/calls", headers=headers, json=payload, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"Telnyx call failed: {response.status_code} {response.text}")
    data = response.json().get("data", {})
    call_id = data.get("call_session_id") or data.get("call_leg_id") or data.get("call_control_id")
    if not call_id:
        raise RuntimeError(f"Telnyx call response did not include a call id: {response.text}")
    call_id = _safe_identifier(call_id)

    from src.startup_timeline import StartupTimeline

    StartupTimeline(scenario_id, call_id).mark("call_initiated", initiated_ms)

    from src.recorder import update_call_meta

    update_call_meta(
        scenario_id,
        call_id,
        started_at=_utc_now(),
        extra={
            "telephony_provider": "telnyx",
            "telnyx_status": "initiated",
            "call_control_id": data.get("call_control_id"),
            "call_leg_id": data.get("call_leg_id"),
            "call_session_id": data.get("call_session_id"),
        },
    )
    print(f"Call initiated: ID={call_id}, Provider=telnyx, Scenario={scenario_id}, Target={_mask_phone(target)}")
    return call_id


def make_call(scenario_id: int, use_test_target: bool = False) -> str:
    target = _target_phone(use_test_target=use_test_target)
    if telephony_provider() == "telnyx":
        return _make_telnyx_call(scenario_id, target)
    return _make_twilio_call(scenario_id, target)


def end_call(call_sid: str) -> None:
    if telephony_provider() == "telnyx":
        raise NotImplementedError("Telnyx hangup is not wired to the CLI yet.")
    _client().calls(call_sid).update(status="completed")


def get_call_status(call_sid: str) -> str:
    if telephony_provider() == "telnyx":
        return _get_telnyx_status_from_meta(call_sid)
    return _client().calls(call_sid).fetch().status


def _get_telnyx_status_from_meta(call_sid: str) -> str:
    for meta_path in Path("runs").glob(f"scenario_*_{call_sid}/call_meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = meta.get("telnyx_status") or meta.get("status")
        if status in {"call.hangup", "hangup", "completed"}:
            return "completed"
        if status in {"call.initiated", "initiated"}:
            return "initiated"
        if status in {"call.answered", "answered", "streaming.started"}:
            return "in-progress"
        if status:
            return str(status)
    return "initiated"
