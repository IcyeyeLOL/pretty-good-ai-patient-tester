from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from twilio.rest import Client

from scenarios.scenario_cards import get_scenario

load_dotenv()


def get_run_directory(scenario_id: int, call_sid: str) -> str:
    path = Path("runs") / f"scenario_{scenario_id:02d}_{call_sid}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _twilio_client() -> Client:
    return Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])


def _recording_download_url(recording_uri: str) -> str:
    return f"https://api.twilio.com{recording_uri.replace('.json', '.mp3')}"


def download_recording(recording_sid: str, call_sid: str, scenario_id: int) -> str | None:
    client = _twilio_client()
    run_dir = Path(get_run_directory(scenario_id, call_sid))
    target = run_dir / "recording.mp3"

    for attempt in range(1, 4):
        try:
            recording = client.recordings(recording_sid).fetch()
            url = _recording_download_url(recording.uri)
            with httpx.Client(timeout=60.0) as http:
                response = http.get(
                    url,
                    auth=(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"]),
                )
                response.raise_for_status()
            target.write_bytes(response.content)
            client.recordings(recording_sid).delete()
            return str(target)
        except Exception as exc:
            error_path = run_dir / "recording_error.txt"
            error_path.write_text(
                f"Attempt {attempt}/3 failed for {recording_sid}: {exc}\n",
                encoding="utf-8",
            )
            if attempt < 3:
                time.sleep(10)

    return None


def update_call_meta(
    scenario_id: int,
    call_sid: str,
    *,
    recording_sid: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    duration_seconds: int | float | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    run_dir = Path(get_run_directory(scenario_id, call_sid))
    meta_path = run_dir / "call_meta.json"
    scenario = get_scenario(scenario_id)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {
            "scenario_id": scenario_id,
            "scenario_name": scenario.name if scenario else None,
            "call_sid": call_sid,
            "started_at": started_at or datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
            "duration_seconds": None,
            "recording_sid": None,
        }

    if recording_sid:
        meta["recording_sid"] = recording_sid
    if started_at:
        meta["started_at"] = started_at
    if ended_at:
        meta["ended_at"] = ended_at
    if duration_seconds is not None:
        meta["duration_seconds"] = duration_seconds
    if extra:
        meta.update(extra)

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(meta_path)

