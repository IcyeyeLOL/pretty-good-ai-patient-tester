from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from src.recorder import get_run_directory

load_dotenv()

ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2"


def _headers() -> dict[str, str]:
    return {"authorization": os.environ["ASSEMBLYAI_API_KEY"]}


def _upload_audio(recording_path: str) -> str:
    with httpx.Client(timeout=120.0) as client:
        with open(recording_path, "rb") as audio_file:
            response = client.post(
                f"{ASSEMBLYAI_BASE_URL}/upload",
                headers=_headers(),
                content=audio_file,
            )
        response.raise_for_status()
        return response.json()["upload_url"]


def _submit_transcript(upload_url: str) -> str:
    payload = {
        "audio_url": upload_url,
        "speaker_labels": True,
        "speakers_expected": 2,
        "language_code": "en",
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{ASSEMBLYAI_BASE_URL}/transcript",
            headers={**_headers(), "content-type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        return response.json()["id"]


def _poll_transcript(transcript_id: str, timeout_seconds: int = 300) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    with httpx.Client(timeout=30.0) as client:
        while time.time() < deadline:
            response = client.get(
                f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}",
                headers=_headers(),
            )
            response.raise_for_status()
            data = response.json()
            if data["status"] == "completed":
                return data
            if data["status"] == "error":
                raise RuntimeError(data.get("error", "AssemblyAI transcription failed"))
            time.sleep(3)

    raise TimeoutError(f"AssemblyAI transcription timed out after {timeout_seconds}s")


def transcribe_recording(recording_path: str, scenario_id: int, call_sid: str) -> dict | None:
    run_dir = Path(get_run_directory(scenario_id, call_sid))
    try:
        upload_url = _upload_audio(recording_path)
        transcript_id = _submit_transcript(upload_url)
        result = _poll_transcript(transcript_id)
        turns = [
            {
                "speaker": utterance.get("speaker"),
                "text": utterance.get("text", ""),
                "start_ms": utterance.get("start"),
                "end_ms": utterance.get("end"),
            }
            for utterance in result.get("utterances", []) or []
        ]
        transcript = {
            "scenario_id": scenario_id,
            "call_sid": call_sid,
            "duration_seconds": result.get("audio_duration"),
            "speaker_A_is_patient": True,
            "turns": turns,
            "full_text": result.get("text", ""),
        }
        output_path = run_dir / "transcript_final.json"
        output_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
        return transcript
    except Exception as exc:
        (run_dir / "transcription_error.txt").write_text(str(exc), encoding="utf-8")
        return None


def save_live_transcript(turns: list[dict], scenario_id: int, call_sid: str) -> None:
    run_dir = Path(get_run_directory(scenario_id, call_sid))
    payload = {
        "scenario_id": scenario_id,
        "call_sid": call_sid,
        "turns": turns,
        "speaker_labels": "patient/agent",
    }
    (run_dir / "transcript_live.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

