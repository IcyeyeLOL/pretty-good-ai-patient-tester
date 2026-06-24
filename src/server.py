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
    if not scenario or not turns:
        return
    transcript = {
        "scenario_id": scenario_id,
        "call_sid": call_sid,
        "turns": turns,
        "speaker_labels": "patient/agent",
    }
    evaluate(transcript, scenario)


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
            save_live_transcript(turns, scenario_id, call_sid)
            update_call_meta(scenario_id, call_sid, ended_at=_utc_now())


@app.websocket("/telnyx/ws/{scenario_id}")
async def telnyx_websocket_endpoint(websocket: WebSocket, scenario_id: int):
    await websocket.accept()
    call_sid = "unknown"
    stream_id = None
    inbound_encoding = "PCMU"
    outbound_encoding = "PCMU"
    turns: list[dict] = []

    try:
        while True:
            raw_message = await websocket.receive_text()
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
            update_call_meta(
                scenario_id,
                call_sid,
                ended_at=_utc_now() if event_type == "call.hangup" else None,
                extra={
                    "telephony_provider": "telnyx",
                    "telnyx_status": event_type,
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
async def api_run_scenario(scenario_id: int):
    from src.caller import make_call

    scenario = get_scenario(scenario_id)
    if scenario is None:
        return JSONResponse({"error": f"Unknown scenario {scenario_id}"}, status_code=404)
    try:
        call_sid = make_call(scenario_id)
        return JSONResponse({"call_sid": call_sid, "scenario_id": scenario_id})
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
