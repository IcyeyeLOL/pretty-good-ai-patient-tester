from __future__ import annotations

import asyncio
import json
import os
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

import click
from dotenv import load_dotenv

from scenarios.scenario_cards import get_all_scenarios, get_scenario

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

COMMON_CALL_ENV = [
    "DEEPGRAM_API_KEY",
    "ASSEMBLYAI_API_KEY",
    "CARTESIA_API_KEY",
    "BASE_URL",
    "TARGET_PHONE",
]
TWILIO_ENV = ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER"]
TELNYX_ENV = ["TELNYX_API_KEY", "TELNYX_CONNECTION_ID", "TELNYX_PHONE_NUMBER"]
TARGET_NUMBER = "+18054398008"
SUPPORTED_LLM_PROVIDERS = {"anthropic", "openai"}
SUPPORTED_TELEPHONY_PROVIDERS = {"twilio", "telnyx"}


def validate_env(require_call_keys: bool = True, use_test_target: bool = False) -> None:
    telephony_provider = os.environ.get("TELEPHONY_PROVIDER", "twilio").lower()
    required = ["BASE_URL", "TARGET_PHONE"]
    if require_call_keys:
        required = [*COMMON_CALL_ENV]
        if telephony_provider == "telnyx":
            required = [*required, *TELNYX_ENV]
        else:
            required = [*required, *TWILIO_ENV]
    if require_call_keys and use_test_target:
        required = [*required, "VERIFIED_TEST_PHONE"]
    missing = [name for name in required if not os.environ.get(name)]
    errors = []
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if require_call_keys:
        if provider not in SUPPORTED_LLM_PROVIDERS:
            errors.append("LLM_PROVIDER must be 'anthropic' or 'openai'.")
        elif provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            errors.append("Missing required environment variable: ANTHROPIC_API_KEY")
        elif provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            errors.append("Missing required environment variable: OPENAI_API_KEY")
        if telephony_provider not in SUPPORTED_TELEPHONY_PROVIDERS:
            errors.append("TELEPHONY_PROVIDER must be 'twilio' or 'telnyx'.")
    if missing:
        errors.append("Missing required environment variables: " + ", ".join(missing))
    base_url = os.environ.get("BASE_URL", "")
    if base_url.endswith("/"):
        errors.append("BASE_URL must not end with a slash.")
    if os.environ.get("TARGET_PHONE") != TARGET_NUMBER:
        errors.append(f"TARGET_PHONE must be exactly {TARGET_NUMBER}.")
    if use_test_target:
        from src.caller import _require_e164

        try:
            _require_e164("VERIFIED_TEST_PHONE", os.environ.get("VERIFIED_TEST_PHONE", ""))
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        for error in errors:
            click.echo(f"ERROR: {error}", err=True)
        action = "running calls" if require_call_keys else "starting the server"
        raise click.ClickException(f"Fix .env before {action}.")


def is_port_listening(host: str = "127.0.0.1", port: int = 8000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def start_background_server():
    if is_port_listening():
        click.echo("Using existing dev server on port 8000.")
        return None, None

    import uvicorn

    config = uvicorn.Config(
        "src.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
    server = uvicorn.Server(config)

    def serve():
        asyncio.run(server.serve())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    time.sleep(2)
    return server, thread


def warmup_local_server(timeout_seconds: float = 30.0) -> bool:
    """Preload the live-pipeline imports on the local dev server before calling.

    Moves cold-start import cost off the first call's pipeline-build path. Returns
    True if the server reported ready.
    """
    import httpx

    try:
        response = httpx.post("http://127.0.0.1:8000/api/warmup", timeout=timeout_seconds)
        ready = bool(response.json().get("ready"))
        click.echo(f"Warmup: {'ready' if ready else 'incomplete'}.")
        return ready
    except Exception as exc:
        click.echo(f"Warmup skipped: {exc}", err=True)
        return False


def wait_for_call_completion(call_sid: str, timeout_seconds: int = 360) -> str:
    from src.caller import get_call_status

    terminal_statuses = {"completed", "failed", "busy", "no-answer", "canceled"}
    deadline = time.time() + timeout_seconds
    last_status = "unknown"
    while time.time() < deadline:
        try:
            last_status = get_call_status(call_sid)
            click.echo(f"Call status: {last_status}")
            if last_status in terminal_statuses:
                return last_status
        except Exception as exc:
            click.echo(f"Unable to fetch call status yet: {exc}", err=True)
        time.sleep(5)
    return last_status


def wait_for_post_call_artifacts(scenario_id: int, call_sid: str, timeout_seconds: int = 480) -> Path:
    run_dir = Path("runs") / f"scenario_{scenario_id:02d}_{call_sid}"
    judge_path = run_dir / "judge_output.json"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if judge_path.exists():
            return run_dir
        time.sleep(5)
    return run_dir


SLOW_TURN_MS = 1800


def summarize_turn_latency(turns: list[dict]) -> dict:
    """Median/max perceived latency and slow-turn flags from latency_debug turns.

    `stt_to_first_audio_ms` is the perceived response latency (caller stops
    talking -> first bot audio). `stt_to_first_token_ms` isolates the LLM's
    time-to-first-token when both checkpoints are present.
    """
    perceived = [
        t["stt_to_first_audio_ms"] for t in turns if t.get("stt_to_first_audio_ms") is not None
    ]
    first_token = [
        t["first_llm_token"] - t["stt_committed"]
        for t in turns
        if t.get("first_llm_token") is not None and t.get("stt_committed") is not None
    ]
    summary: dict = {"turns_measured": len(perceived)}
    if perceived:
        summary["median_stt_to_first_audio_ms"] = int(statistics.median(perceived))
        summary["max_stt_to_first_audio_ms"] = max(perceived)
        summary["slow_turns"] = [
            {"index": i, "ms": v}
            for i, t in enumerate(turns)
            if (v := t.get("stt_to_first_audio_ms")) is not None and v > SLOW_TURN_MS
        ]
    if first_token:
        summary["median_stt_to_first_token_ms"] = int(statistics.median(first_token))
        summary["max_stt_to_first_token_ms"] = max(first_token)
    return summary


def _print_latency_summary(run_dir: Path) -> None:
    latency = _load_json(run_dir / "latency_debug.json")
    if not latency:
        return
    summary = summarize_turn_latency(latency.get("turns", []))
    if not summary.get("turns_measured"):
        return
    click.echo("\n=== Response latency ===")
    click.echo(
        f"stt_to_first_audio_ms: median={summary.get('median_stt_to_first_audio_ms')} "
        f"max={summary.get('max_stt_to_first_audio_ms')} (n={summary['turns_measured']})"
    )
    if "median_stt_to_first_token_ms" in summary:
        click.echo(
            f"stt_to_first_token_ms: median={summary['median_stt_to_first_token_ms']} "
            f"max={summary['max_stt_to_first_token_ms']}"
        )
    for slow in summary.get("slow_turns", []):
        click.echo(f"  SLOW turn #{slow['index']}: {slow['ms']}ms (> {SLOW_TURN_MS}ms)")


def _format_ms(ms: int | float | None) -> str:
    if ms is None:
        return "--:--"
    seconds = int(float(ms) / 1000)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _load_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        click.echo(f"Unable to read {path}: {exc}", err=True)
    return None


def _transcript_path(run_dir: Path) -> Path | None:
    for name in ("transcript_final.json", "transcript_live.json"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def print_run_artifacts(run_dir: Path) -> None:
    transcript_file = _transcript_path(run_dir)
    if not transcript_file:
        click.echo(f"\nNo transcript found yet in {run_dir}")
        return

    transcript = _load_json(transcript_file) or {}
    click.echo(f"\n=== Transcript: {transcript_file} ===")
    for turn in transcript.get("turns", []):
        speaker = turn.get("speaker", "unknown")
        timestamp = _format_ms(turn.get("start_ms"))
        text = " ".join(str(turn.get("text", "")).split())
        click.echo(f"[{timestamp}] {speaker}: {text}")

    timeline = _load_json(run_dir / "startup_timeline.json")
    if timeline and timeline.get("deltas"):
        click.echo("\n=== Startup timeline (ms) ===")
        for key, value in timeline["deltas"].items():
            click.echo(f"{key}: {value}")

    _print_latency_summary(run_dir)

    judge = _load_json(run_dir / "judge_output.json")
    if judge:
        click.echo("\n=== Judge ===")
        click.echo(f"Verdict: {judge.get('verdict')} ({judge.get('severity')})")
        if judge.get("summary"):
            click.echo(f"Summary: {judge['summary']}")


def run_one_scenario(scenario_id: int, use_test_target: bool = False) -> str | None:
    from src.caller import make_call

    scenario = get_scenario(scenario_id)
    if scenario is None:
        raise click.ClickException(f"Unknown scenario id: {scenario_id}")

    try:
        call_sid = make_call(scenario_id, use_test_target=use_test_target)
    except Exception as exc:
        message = str(exc)
        if "unverified" in message.lower() and "trial accounts" in message.lower():
            raise click.ClickException(
                "Twilio refused the call because this appears to be a trial account. "
                "Trial accounts can only call verified recipient numbers; upgrade Twilio "
                "or use a paid Twilio account before calling the assessment number."
            ) from exc
        raise click.ClickException(f"Failed to place call: {exc}") from exc

    click.echo("Call placed. Waiting for conversation to complete...")
    status = wait_for_call_completion(call_sid)
    if status != "completed":
        click.echo(f"Call ended with status '{status}'. Check provider logs before rerunning.")
        run_dir = wait_for_post_call_artifacts(scenario_id, call_sid, timeout_seconds=45)
        print_run_artifacts(run_dir)
        return call_sid

    click.echo("Call completed. Waiting for transcript and post-call analysis...")
    run_dir = wait_for_post_call_artifacts(scenario_id, call_sid)
    click.echo(f"Run complete. Check {run_dir}/ for artifacts.")
    print_run_artifacts(run_dir)
    return call_sid


def _ngrok_public_url() -> str | None:
    """Return the https public URL of a running ngrok tunnel to :8000, if any."""
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    tunnels = data.get("tunnels", [])
    for tunnel in tunnels:
        if tunnel.get("proto") == "https" and "8000" in tunnel.get("config", {}).get("addr", ""):
            return tunnel.get("public_url")
    for tunnel in tunnels:  # fall back to any https tunnel
        if tunnel.get("proto") == "https":
            return tunnel.get("public_url")
    return None


def _write_base_url(url: str) -> None:
    """Persist BASE_URL to the environment and .env so the app/webhooks use it."""
    os.environ["BASE_URL"] = url
    env_path = Path(".env")
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    out, found = [], False
    for line in lines:
        if line.startswith("BASE_URL="):
            out.append(f"BASE_URL={url}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"BASE_URL={url}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def bring_up_tunnel() -> str | None:
    """Ensure an ngrok tunnel to :8000 exists; write its URL to BASE_URL/.env.

    Reuses a running tunnel if one is up; otherwise starts `ngrok http 8000` in the
    background. Returns the public https URL or None if ngrok is unavailable.
    """
    import subprocess

    url = _ngrok_public_url()
    if url:
        _write_base_url(url)
        return url
    try:
        subprocess.Popen(
            ["ngrok", "http", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        click.echo("ngrok not found on PATH. Install ngrok, or set BASE_URL in .env manually.", err=True)
        return None
    for _ in range(30):
        time.sleep(0.5)
        url = _ngrok_public_url()
        if url:
            _write_base_url(url)
            return url
    return None


def load_judge_outputs() -> list[dict]:
    outputs = []
    for path in sorted(Path("runs").glob("scenario_*/judge_output.json")):
        try:
            outputs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            click.echo(f"Skipping unreadable {path}: {exc}", err=True)
    return outputs


@click.command()
@click.option("--scenario", "scenario_id", type=int, help="Run a single scenario by ID.")
@click.option("--all", "run_all", is_flag=True, help="Run all 10 scenarios sequentially.")
@click.option("--list", "list_scenarios", is_flag=True, help="Print all scenario names and IDs.")
@click.option("--report", "report", is_flag=True, help="Generate bug report from completed runs.")
@click.option("--server", "server_only", is_flag=True, help="Start the FastAPI server only.")
@click.option(
    "--up",
    "up",
    is_flag=True,
    help="One command: auto-start ngrok, set BASE_URL, and launch the server + UI.",
)
@click.option(
    "--test-target",
    is_flag=True,
    help="Call VERIFIED_TEST_PHONE instead of the assessment number. Use only for Twilio trial testing.",
)
def main(
    scenario_id: int | None,
    run_all: bool,
    list_scenarios: bool,
    report: bool,
    server_only: bool,
    up: bool,
    test_target: bool,
) -> None:
    selected = [scenario_id is not None, run_all, list_scenarios, report, server_only, up]
    if sum(bool(item) for item in selected) != 1:
        raise click.ClickException(
            "Choose exactly one option: --up, --scenario, --all, --list, --report, or --server."
        )
    if test_target and scenario_id is None:
        raise click.ClickException("--test-target can only be used with --scenario.")

    if list_scenarios:
        click.echo(f"{'ID':<4} {'Severity':<8} Name")
        click.echo("-" * 64)
        for scenario in get_all_scenarios():
            click.echo(f"{scenario.id:<4} {scenario.severity:<8} {scenario.name}")
        return

    if report:
        from src.judge_bot import generate_bug_report

        path = generate_bug_report(load_judge_outputs())
        click.echo(Path(path).read_text(encoding="utf-8"))
        return

    if up:
        # Single-command bring-up: tunnel + BASE_URL + server, ready for the UI.
        url = bring_up_tunnel()
        if url:
            click.echo(f"ngrok tunnel ready -> {url}")
        else:
            existing = os.environ.get("BASE_URL")
            if existing:
                click.echo(f"Using existing BASE_URL -> {existing}")
            else:
                raise click.ClickException(
                    "No ngrok tunnel and no BASE_URL set. Start ngrok or set BASE_URL in .env."
                )
        validate_env(require_call_keys=False)
        import uvicorn

        click.echo("Open the console at http://localhost:8000  (Ctrl+C to stop)")
        uvicorn.run("src.server:app", host="0.0.0.0", port=8000)
        return

    if server_only:
        validate_env(require_call_keys=False)
        import uvicorn

        # IMPORTANT: keep reload=False on Windows. With reload=True the uvicorn
        # worker comes up on the Proactor event loop (before server.py can set the
        # Selector policy), and Proactor fails socket accept() with WinError 10014,
        # which silently breaks every inbound WebSocket -> Telnyx media streaming
        # fails with 90046 "Failed to connect to destination". reload=False lets
        # uvicorn.run import the app first, so the Selector loop policy applies.
        # (Restart the server manually to pick up code changes.)
        uvicorn.run("src.server:app", host="0.0.0.0", port=8000)
        return

    validate_env(require_call_keys=True, use_test_target=test_target)

    server, thread = start_background_server()
    warmup_local_server()
    try:
        if scenario_id is not None:
            run_one_scenario(scenario_id, use_test_target=test_target)
        elif run_all:
            for scenario in get_all_scenarios():
                click.echo(f"\n=== Scenario {scenario.id}: {scenario.name} ===")
                run_one_scenario(scenario.id)
                if scenario.id != 10:
                    click.echo("Waiting 60 seconds before the next call...")
                    time.sleep(60)
            from src.judge_bot import generate_bug_report

            path = generate_bug_report(load_judge_outputs())
            click.echo(Path(path).read_text(encoding="utf-8"))
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        click.echo("Interrupted.", err=True)
        sys.exit(130)
