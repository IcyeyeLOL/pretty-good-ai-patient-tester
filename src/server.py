from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import parse_qs

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from scenarios.scenario_cards import get_all_scenarios, get_scenario
from src.judge_bot import evaluate, generate_bug_report
from src.pipeline import build_pipeline
from src.recorder import download_recording, update_call_meta
from src.transcriber import save_live_transcript, transcribe_recording

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ACTIVE_CALLS: dict[str, int] = {}
CALL_STATUS: dict[str, str] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Path("runs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    try:
        from src.warmup import warm_runtime

        report = warm_runtime()
        print(f"[DEBUG] warmup ready={report.get('ready')} failed={list(report.get('failed', {}))}")
    except Exception as exc:
        print(f"[DEBUG] warmup failed: {exc}")
    yield


app = FastAPI(lifespan=lifespan)

_STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _websocket_base_url() -> str:
    base_url = os.environ["BASE_URL"].rstrip("/")
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://") :]
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://") :]
    return "wss://" + base_url


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _scenario_from_client_state(client_state: str | None) -> int | None:
    if not client_state:
        return None
    try:
        decoded = base64.b64decode(client_state).decode("utf-8")
        payload = json.loads(decoded)
        scenario_id = payload.get("scenario_id")
        return int(scenario_id) if scenario_id else None
    except (ValueError, TypeError, json.JSONDecodeError, binascii.Error):
        return None


def _telnyx_call_id(payload: dict) -> str:
    raw = (
        payload.get("call_session_id")
        or payload.get("call_leg_id")
        or payload.get("call_control_id")
        or "unknown"
    )
    return _safe_identifier(str(raw))


def _evaluate_live_transcript_if_ready(scenario_id: int, call_sid: str, turns: list[dict]) -> None:
    scenario = get_scenario(scenario_id)
    if not scenario:
        return
    if not turns:
        # The call produced no conversation at all (e.g. the TTS provider failed /
        # ran out of credits, so the patient never spoke). Write an ERROR verdict
        # so the UI stops polling "judging..." forever and the failure is visible.
        from src.judge_bot import _save_judge_output

        _save_judge_output(
            {
                "scenario_id": scenario_id,
                "scenario_name": scenario.name,
                "call_sid": call_sid,
                "verdict": "ERROR",
                "severity": None,
                "summary": "No conversation was captured — the call produced no audio. "
                "Check the TTS provider (e.g. Cartesia credits) and try again.",
                "evidence": None,
                "bug_description": None,
                "expected_behavior": None,
                "call_reference": None,
                "judge_source": "none",
                "rule_verdict": None,
                "llm_verdict": None,
                "validation_warnings": ["Empty transcript; the call failed before any turns were exchanged."],
            },
            scenario_id,
            call_sid,
        )
        return
    transcript = {
        "scenario_id": scenario_id,
        "call_sid": call_sid,
        "turns": turns,
        "speaker_labels": "patient/agent",
    }
    evaluate(transcript, scenario)


# Maps Telnyx webhook event_type -> startup timeline mark name.
_TELNYX_EVENT_MARKS = {
    "call.answered": "call_answered",
    "streaming.started": "streaming_started",
    "call.streaming.started": "streaming_started",
    "call.hangup": "telnyx_hangup",
}


def _mark_ws_timeline(scenario_id: int, call_sid: str, ws_accepted_ms: int, ws_start_ms: int) -> None:
    try:
        from src.startup_timeline import StartupTimeline

        timeline = StartupTimeline(scenario_id, call_sid)
        timeline.mark("ws_accepted", ws_accepted_ms)
        timeline.mark("ws_start_event", ws_start_ms)
    except Exception as exc:
        print(f"[DEBUG] startup timeline ws mark failed: {exc}")


def _mark_pipeline_build_started(scenario_id: int, call_sid: str) -> None:
    try:
        from src.startup_timeline import StartupTimeline

        StartupTimeline(scenario_id, call_sid).mark("pipeline_build_started")
    except Exception as exc:
        print(f"[DEBUG] startup timeline build-start mark failed: {exc}")


def _mark_timeline_from_event(scenario_id: int, call_sid: str, event_type: str) -> None:
    mark = _TELNYX_EVENT_MARKS.get(event_type)
    if not mark:
        return
    try:
        from src.startup_timeline import StartupTimeline

        StartupTimeline(scenario_id, call_sid).mark(mark)
    except Exception as exc:
        print(f"[DEBUG] startup timeline mark failed: {exc}")


def _finalize_telnyx_event_log(bundle) -> None:
    log = getattr(bundle, "telnyx_event_log", None) if bundle else None
    if log is None:
        return
    try:
        log.finalize()
    except Exception as exc:
        print(f"[DEBUG] telnyx event log finalize failed: {exc}")


def _close_live_recorder(bundle) -> str | None:
    if not bundle or not getattr(bundle, "live_recorder", None):
        return None
    try:
        return bundle.live_recorder.close()
    except Exception as exc:
        print(f"[DEBUG] live recorder close failed: {exc}")
        return None


async def _form_data(request: Request) -> dict[str, str]:
    body = (await request.body()).decode("utf-8")
    parsed = parse_qs(body)
    return {key: values[0] for key, values in parsed.items() if values}


@app.post("/twiml/{scenario_id}")
async def twiml(scenario_id: int, request: Request) -> HTMLResponse:
    scenario = get_scenario(scenario_id)
    if scenario is None:
        return HTMLResponse("<Response><Hangup /></Response>", status_code=404)

    form = await _form_data(request)
    call_sid = form.get("CallSid")
    if call_sid:
        ACTIVE_CALLS[call_sid] = scenario_id
        update_call_meta(scenario_id, call_sid, started_at=_utc_now())

    stream_url = f"{_websocket_base_url()}/ws/{scenario_id}"
    call_sid_param = f'<Parameter name="call_sid" value="{escape(call_sid)}" />' if call_sid else ""
    response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{escape(stream_url)}">
      <Parameter name="scenario_id" value="{scenario_id}" />
      {call_sid_param}
    </Stream>
  </Connect>
</Response>"""
    return HTMLResponse(content=response, media_type="application/xml")


@app.websocket("/ws/{scenario_id}")
async def websocket_endpoint(websocket: WebSocket, scenario_id: int):
    await websocket.accept()
    call_sid = "unknown"
    stream_sid = None
    turns: list[dict] = []
    bundle = None

    try:
        while True:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)
            if message.get("event") != "start":
                continue

            start = message.get("start", {})
            stream_sid = message.get("streamSid") or start.get("streamSid")
            call_sid = start.get("callSid") or call_sid
            if call_sid != "unknown":
                ACTIVE_CALLS[call_sid] = scenario_id
                update_call_meta(scenario_id, call_sid, started_at=_utc_now())
            break

        if not stream_sid:
            await websocket.close(code=1011)
            return

        bundle = build_pipeline(
            scenario_id=scenario_id,
            websocket=websocket,
            call_sid=call_sid,
            stream_sid=stream_sid,
            telephony_provider="twilio",
            turns=turns,
        )
        await bundle.runner.run(bundle.task)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        if call_sid != "unknown":
            run_dir = Path("runs") / f"scenario_{scenario_id:02d}_{call_sid}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "websocket_error.txt").write_text(str(exc), encoding="utf-8")
    finally:
        if call_sid != "unknown":
            _close_live_recorder(bundle)
            save_live_transcript(turns, scenario_id, call_sid)
            update_call_meta(scenario_id, call_sid, ended_at=_utc_now())


@app.websocket("/telnyx/ws/{scenario_id}")
async def telnyx_websocket_endpoint(websocket: WebSocket, scenario_id: int):
    await websocket.accept()
    ws_accepted_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    call_sid = "unknown"
    stream_id = None
    inbound_encoding = "PCMU"
    outbound_encoding = "PCMU"
    turns: list[dict] = []
    bundle = None

    try:
        while True:
            raw_message = await websocket.receive_text()
            ws_start_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            message = json.loads(raw_message)
            if message.get("event") != "start":
                continue

            start = message.get("start", {})
            stream_id = message.get("stream_id") or start.get("stream_id")
            call_sid = _telnyx_call_id(start)
            scenario_from_state = _scenario_from_client_state(start.get("client_state"))
            if scenario_from_state:
                scenario_id = scenario_from_state

            media_format = start.get("media_format") or {}
            encoding = (media_format.get("encoding") or "PCMU").upper()
            inbound_encoding = encoding
            outbound_encoding = encoding

            if call_sid != "unknown":
                ACTIVE_CALLS[call_sid] = scenario_id
                _mark_ws_timeline(scenario_id, call_sid, ws_accepted_ms, ws_start_ms)
                update_call_meta(
                    scenario_id,
                    call_sid,
                    started_at=_utc_now(),
                    extra={
                        "telephony_provider": "telnyx",
                        "telnyx_status": "streaming.started",
                        "call_control_id": start.get("call_control_id"),
                        "call_session_id": start.get("call_session_id"),
                        "call_leg_id": start.get("call_leg_id"),
                    },
                )
            break

        if not stream_id:
            await websocket.close(code=1011)
            return

        if call_sid != "unknown":
            _mark_pipeline_build_started(scenario_id, call_sid)
        bundle = build_pipeline(
            scenario_id=scenario_id,
            websocket=websocket,
            call_sid=call_sid,
            stream_sid=stream_id,
            telephony_provider="telnyx",
            inbound_encoding=inbound_encoding,
            outbound_encoding=outbound_encoding,
            turns=turns,
        )
        await bundle.runner.run(bundle.task)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        if call_sid != "unknown":
            run_dir = Path("runs") / f"scenario_{scenario_id:02d}_{call_sid}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "websocket_error.txt").write_text(str(exc), encoding="utf-8")
    finally:
        if call_sid != "unknown":
            _finalize_telnyx_event_log(bundle)
            _close_live_recorder(bundle)
            save_live_transcript(turns, scenario_id, call_sid)
            update_call_meta(
                scenario_id,
                call_sid,
                ended_at=_utc_now(),
                extra={"telephony_provider": "telnyx"},
            )
            _evaluate_live_transcript_if_ready(scenario_id, call_sid, turns)


@app.post("/telnyx/events")
async def telnyx_events(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return PlainTextResponse("invalid json", status_code=400)

    data = payload.get("data") or {}
    event_type = data.get("event_type") or data.get("record_type") or "unknown"
    event_payload = data.get("payload") or {}
    scenario_id = int(
        request.query_params.get("scenario_id")
        or _scenario_from_client_state(event_payload.get("client_state"))
        or 0
    )
    call_sid = _telnyx_call_id(event_payload)

    if call_sid != "unknown":
        CALL_STATUS[call_sid] = "completed" if event_type == "call.hangup" else event_type
        if scenario_id:
            ACTIVE_CALLS[call_sid] = scenario_id
            _mark_timeline_from_event(scenario_id, call_sid, event_type)
            update_call_meta(
                scenario_id,
                call_sid,
                ended_at=_utc_now() if event_type == "call.hangup" else None,
                extra={
                    "telephony_provider": "telnyx",
                    "telnyx_status": event_type,
                    # Timestamp each distinct event_type so call_meta carries a Telnyx
                    # event timeline (e.g. telnyx_event_call.answered_at) — useful for
                    # correlating the opener against call.answered / media start.
                    f"telnyx_event_{event_type}_at": _utc_now(),
                    "call_control_id": event_payload.get("call_control_id"),
                    "call_session_id": event_payload.get("call_session_id"),
                    "call_leg_id": event_payload.get("call_leg_id"),
                    "telnyx_hangup_cause": event_payload.get("hangup_cause"),
                },
            )

    return PlainTextResponse("ok")


@app.post("/recording-complete")
async def recording_complete(request: Request, background_tasks: BackgroundTasks):
    form = await _form_data(request)
    call_sid = form.get("CallSid")
    recording_sid = form.get("RecordingSid")
    status = form.get("RecordingStatus")
    scenario_id = int(request.query_params.get("scenario_id") or ACTIVE_CALLS.get(call_sid, 0))

    if not call_sid or not recording_sid or not scenario_id:
        return PlainTextResponse("missing call, recording, or scenario", status_code=400)

    update_call_meta(scenario_id, call_sid, recording_sid=recording_sid)
    if status == "completed":
        background_tasks.add_task(process_recording, recording_sid, call_sid, scenario_id)

    return PlainTextResponse("ok")


@app.post("/call-status")
async def call_status(request: Request):
    form = await _form_data(request)
    call_sid = form.get("CallSid")
    status = form.get("CallStatus")
    scenario_id = int(request.query_params.get("scenario_id") or ACTIVE_CALLS.get(call_sid, 0))
    if call_sid and status:
        CALL_STATUS[call_sid] = status
        if scenario_id:
            ACTIVE_CALLS[call_sid] = scenario_id
            duration = form.get("CallDuration")
            update_call_meta(
                scenario_id,
                call_sid,
                ended_at=_utc_now() if status == "completed" else None,
                duration_seconds=int(duration) if duration and duration.isdigit() else None,
                extra={"twilio_status": status},
            )
    return PlainTextResponse("ok")


def process_recording(recording_sid: str, call_sid: str, scenario_id: int) -> None:
    scenario = get_scenario(scenario_id)
    if scenario is None:
        return

    recording_path = download_recording(recording_sid, call_sid, scenario_id)
    transcript = None
    if recording_path:
        transcript = transcribe_recording(recording_path, scenario_id, call_sid)

    if transcript is None:
        live_path = Path("runs") / f"scenario_{scenario_id:02d}_{call_sid}" / "transcript_live.json"
        if live_path.exists():
            transcript = json.loads(live_path.read_text(encoding="utf-8"))

    if transcript:
        evaluate(transcript, scenario)


# ─── Dev UI routes ────────────────────────────────────────────────────────────

@app.get("/")
async def ui_root() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/api/ping")
async def api_ping():
    return {"ok": True}


@app.get("/api/warmup")
@app.post("/api/warmup")
async def api_warmup():
    from src.warmup import warm_runtime

    report = warm_runtime()
    return JSONResponse({"ready": bool(report.get("ready")), "report": report})


@app.get("/api/scenarios")
async def api_scenarios():
    result = []
    for s in get_all_scenarios():
        last_verdict = _last_verdict_for_scenario(s.id)
        result.append({
            "id": s.id,
            "name": s.name,
            "caller_name": s.caller_name,
            "goal": s.goal,
            "hidden_trap": s.hidden_trap,
            "expected_agent_behavior": s.expected_agent_behavior,
            "bug_conditions": s.bug_conditions,
            "severity": s.severity,
            "end_condition": s.end_condition,
            "opening_line": s.opening_line,
            "last_verdict": last_verdict,
        })
    return JSONResponse(result)


def _last_verdict_for_scenario(scenario_id: int) -> str | None:
    pattern = f"scenario_{scenario_id:02d}_*"
    dirs = sorted(Path("runs").glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    for run_dir in reversed(dirs):
        judge_path = run_dir / "judge_output.json"
        if judge_path.exists():
            try:
                return json.loads(judge_path.read_text(encoding="utf-8")).get("verdict")
            except Exception:
                pass
    return None


@app.post("/api/run/{scenario_id}")
async def api_run_scenario(scenario_id: int, test_target: bool = False):
    from src.caller import make_call

    scenario = get_scenario(scenario_id)
    if scenario is None:
        return JSONResponse({"error": f"Unknown scenario {scenario_id}"}, status_code=404)
    try:
        call_sid = make_call(scenario_id, use_test_target=test_target)
        return JSONResponse({"call_sid": call_sid, "scenario_id": scenario_id})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/runs/{call_sid}/artifacts")
async def api_run_artifacts(call_sid: str):
    """Return latency_debug.json and startup_timeline.json for a specific call."""
    for run_dir in Path("runs").glob(f"scenario_*_{call_sid}"):
        return JSONResponse({
            "latency_debug": _load_json(run_dir / "latency_debug.json"),
            "startup_timeline": _load_json(run_dir / "startup_timeline.json"),
        })
    # Try without scenario prefix (sim runs)
    for run_dir in Path("runs").glob(f"*{call_sid}*"):
        return JSONResponse({
            "latency_debug": _load_json(run_dir / "latency_debug.json"),
            "startup_timeline": _load_json(run_dir / "startup_timeline.json"),
        })
    return JSONResponse({"latency_debug": None, "startup_timeline": None})


def _find_run_dir(call_sid: str) -> Path | None:
    for run_dir in Path("runs").glob(f"scenario_*_{call_sid}"):
        return run_dir
    for run_dir in Path("runs").glob(f"*{call_sid}*"):
        return run_dir
    return None


@app.get("/api/runs/{call_sid}/recording")
async def api_run_recording(call_sid: str, kind: str = "mp3"):
    """Serve the call audio for download.

    kind: "mp3" (mixed recording, default), "agent", or "patient" (live WAVs).
    """
    run_dir = _find_run_dir(call_sid)
    if run_dir is None:
        return JSONResponse({"error": "run not found"}, status_code=404)

    candidates = {
        "mp3": ("recording.mp3", "audio/mpeg"),
        "agent": ("live_agent.wav", "audio/wav"),
        "patient": ("live_patient.wav", "audio/wav"),
    }
    filename, media_type = candidates.get(kind, candidates["mp3"])
    audio_path = run_dir / filename
    # Fall back to the agent WAV if the mixed mp3 was never produced.
    if not audio_path.exists() and kind == "mp3":
        audio_path = run_dir / "live_agent.wav"
        filename, media_type = "live_agent.wav", "audio/wav"
    if not audio_path.exists():
        return JSONResponse({"error": "no recording for this call"}, status_code=404)

    return FileResponse(str(audio_path), media_type=media_type, filename=f"{run_dir.name}_{filename}")


@app.get("/api/runs/{call_sid}/has-recording")
async def api_run_has_recording(call_sid: str):
    run_dir = _find_run_dir(call_sid)
    if run_dir is None:
        return JSONResponse({"mp3": False, "agent": False, "patient": False})
    return JSONResponse({
        "mp3": (run_dir / "recording.mp3").exists(),
        "agent": (run_dir / "live_agent.wav").exists(),
        "patient": (run_dir / "live_patient.wav").exists(),
    })


@app.post("/api/runs/{call_sid}/rejudge")
async def api_rejudge(call_sid: str):
    """Re-run the judge on a call's existing transcript (e.g. after an ERROR)."""
    run_dir = _find_run_dir(call_sid)
    if run_dir is None:
        return JSONResponse({"error": "run not found"}, status_code=404)
    transcript = _load_json(run_dir / "transcript_final.json") or _load_json(
        run_dir / "transcript_live.json"
    )
    if not transcript or not transcript.get("turns"):
        return JSONResponse(
            {"error": "No transcript with conversation turns to judge for this call."},
            status_code=400,
        )
    scenario_id = transcript.get("scenario_id")
    if scenario_id is None:
        meta = _load_json(run_dir / "call_meta.json") or {}
        scenario_id = meta.get("scenario_id")
    scenario = get_scenario(int(scenario_id)) if scenario_id is not None else None
    if scenario is None:
        return JSONResponse({"error": "Unknown scenario for this call."}, status_code=400)
    try:
        result = evaluate(transcript, scenario)  # save=True writes judge_output.json
        return JSONResponse({"judge": result})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/status/{call_sid}")
async def api_call_status(call_sid: str):
    provider_status = CALL_STATUS.get(call_sid)
    scenario_id = ACTIVE_CALLS.get(call_sid)
    provider = os.environ.get("TELEPHONY_PROVIDER", "twilio").lower()
    result: dict = {
        "call_sid": call_sid,
        "status": provider_status,
        "twilio_status": provider_status if provider == "twilio" else None,
        "telnyx_status": provider_status if provider == "telnyx" else None,
    }

    if scenario_id:
        run_dir = Path("runs") / f"scenario_{scenario_id:02d}_{call_sid}"
        transcript = _load_json(run_dir / "transcript_final.json") or _load_json(run_dir / "transcript_live.json")
        judge = _load_json(run_dir / "judge_output.json")
        result["transcript"] = transcript
        result["judge"] = judge

    return JSONResponse(result)


@app.get("/api/runs")
async def api_list_runs(scenario_id: int | None = None, call_sid: str | None = None):
    runs = []
    pattern = f"scenario_{scenario_id:02d}_*" if scenario_id else "scenario_*"
    for run_dir in sorted(Path("runs").glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0):
        meta = _load_json(run_dir / "call_meta.json") or {}
        sid = meta.get("call_sid") or run_dir.name.split("_", 2)[-1]
        if call_sid and sid != call_sid:
            continue
        transcript = _load_json(run_dir / "transcript_final.json") or _load_json(run_dir / "transcript_live.json")
        judge = _load_json(run_dir / "judge_output.json")
        runs.append({
            "call_sid": sid,
            "scenario_id": meta.get("scenario_id"),
            "scenario_name": meta.get("scenario_name"),
            "started_at": meta.get("started_at"),
            "ended_at": meta.get("ended_at"),
            "duration_seconds": meta.get("duration_seconds"),
            "transcript": transcript,
            "judge": judge,
        })
    return JSONResponse(runs)


@app.post("/api/reset")
async def api_reset_session():
    """Archive existing runs and clear the bug report so the console is brand new.

    Non-destructive: runs are moved into runs_archive/<timestamp>/ rather than
    deleted, so captured transcripts and diagnostics are preserved.
    """
    import shutil

    archived = 0
    runs_dir = Path("runs")
    run_dirs = [p for p in runs_dir.glob("*") if p.is_dir()] if runs_dir.exists() else []
    if run_dirs:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_dir = Path("runs_archive") / stamp
        archive_dir.mkdir(parents=True, exist_ok=True)
        for run_dir in run_dirs:
            try:
                shutil.move(str(run_dir), str(archive_dir / run_dir.name))
                archived += 1
            except Exception as exc:
                print(f"[DEBUG] reset: could not archive {run_dir.name}: {exc}")

    report_path = Path("reports") / "bug_report.md"
    if report_path.exists():
        try:
            report_path.unlink()
        except Exception as exc:
            print(f"[DEBUG] reset: could not remove report: {exc}")

    ACTIVE_CALLS.clear()
    CALL_STATUS.clear()
    return JSONResponse({"ok": True, "archived_runs": archived})


@app.post("/api/report")
async def api_generate_report():
    outputs = []
    for path in sorted(Path("runs").glob("scenario_*/judge_output.json")):
        try:
            outputs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    report_path = generate_bug_report(outputs)
    content = Path(report_path).read_text(encoding="utf-8")
    return JSONResponse({"content": content, "path": report_path})


@app.get("/api/report")
async def api_get_report():
    report_path = Path("reports") / "bug_report.md"
    if not report_path.exists():
        return JSONResponse({"error": "No report yet"}, status_code=404)
    return JSONResponse({"content": report_path.read_text(encoding="utf-8"), "path": str(report_path)})


def _load_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None
