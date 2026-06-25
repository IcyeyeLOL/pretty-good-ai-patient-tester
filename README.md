# AI Patient Tester

Python voice-call tester for the Pretty Good AI engineering challenge. It places outbound calls to a healthcare scheduling agent, acts as a realistic patient across 10 scenarios, records/transcribes the call, and runs a post-call judge that flags bugs and weaknesses.

## What The System Does

1. Starts a scenario call through Telnyx or Twilio.
2. Runs a real-time patient voice pipeline:
   phone audio -> Deepgram STT -> Claude/OpenAI patient brain -> Cartesia TTS -> phone audio.
3. Saves the call artifacts under `runs/scenario_XX_<call_id>/`.
4. Runs the judge against the transcript and scenario expectations.
5. Serves a local UI for scenarios, transcripts, judge results, MP3 recordings, and debug artifacts.

## Tech Stack

| Part | Purpose |
| --- | --- |
| Python 3.12 | Main app, CLI, pipeline, judge, tests |
| FastAPI | Local backend, UI/API, Telnyx/Twilio webhooks, WebSocket media endpoint |
| Telnyx | Outbound calls and live bidirectional phone audio |
| ngrok | Public HTTPS/WSS tunnel to the local FastAPI server |
| Pipecat | Real-time voice pipeline framework |
| Deepgram | Speech-to-text for the office agent's audio |
| Anthropic Claude / OpenAI | Patient bot reasoning and judge LLM fallback |
| Cartesia | Patient text-to-speech |
| AssemblyAI | Optional final transcript path for provider recordings |
| pytest | Unit, offline judge, dry-run, and pipeline tests |

## Quickstart

### 1. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Configure environment

```powershell
Copy-Item .env.example .env
```

Fill in `.env`. Keep the final assessment number locked:

```text
TARGET_PHONE=+18054398008
```

For local testing, use your verified phone number:

```text
VERIFIED_TEST_PHONE=+1XXXXXXXXXX
```

Core required keys for the Telnyx path:

```text
TELEPHONY_PROVIDER=telnyx
TELNYX_API_KEY=...
TELNYX_CONNECTION_ID=...
TELNYX_PHONE_NUMBER=+1...

LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-4-6

DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
CARTESIA_MODEL=sonic-2
ASSEMBLYAI_API_KEY=...

OPENING_DELAY_SECONDS=0.5
BASE_URL=https://your-public-ngrok-url
```

Do not commit `.env`; it is ignored by git.

### 3. Start ngrok

In one terminal:

```powershell
ngrok http 8000
```

Copy the HTTPS URL into `.env` without a trailing slash:

```text
BASE_URL=https://abc123.ngrok-free.app
```

### 4. Run a test call

Use `--test-target` while iterating so the call goes to `VERIFIED_TEST_PHONE`:

```powershell
.\.venv\Scripts\python.exe run.py --scenario 1 --test-target
```

Run the assessment target only when ready:

```powershell
.\.venv\Scripts\python.exe run.py --scenario 1
```

## Common Commands

List scenarios:

```powershell
.\.venv\Scripts\python.exe run.py --list
```

Run a specific scenario:

```powershell
.\.venv\Scripts\python.exe run.py --scenario 4 --test-target
```

Run all scenarios:

```powershell
.\.venv\Scripts\python.exe run.py --all
```

Start only the FastAPI server:

```powershell
.\.venv\Scripts\python.exe run.py --server
```

Start or restart the background dev server:

```powershell
.\scripts\start-dev-server.ps1
.\scripts\start-dev-server.ps1 -Restart
```

Generate the compiled bug report:

```powershell
.\.venv\Scripts\python.exe run.py --report
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scenarios sim tests
```

## Scenarios

The project includes 10 scenario cards in `scenarios/scenario_cards.py`.

| ID | Scenario | What It Tests |
| --- | --- | --- |
| 1 | Basic Appointment Scheduling | Required intake fields and appointment confirmation |
| 2 | Weekend Hours Hallucination | Refusing weekend appointments and offering weekdays |
| 3 | Cancel and Reschedule | Cancellation state plus rescheduling pivot |
| 4 | Urgent Symptoms | Emergency escalation for chest symptoms |
| 5 | Third-Party PHI | Identity/relationship verification before disclosure |
| 6 | Medication Refill | Escalating concerning side effects |
| 7 | Insurance Uncertainty | Avoiding unverifiable coverage claims |
| 8 | Office Location | Avoiding invented location details |
| 9 | Multi-Intent Patient | Context tracking across several requests |
| 10 | Barge-In Handling | Interruption handling and avoiding repeated intake |

## Offline Validation Before Live Calls

Use offline checks before spending call budget.

Prompt fidelity:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_prompt_fidelity.py -q
```

Dry-run simulated conversations:

```powershell
.\.venv\Scripts\python.exe -m sim.dry_run --scenario 4 --agent both
.\.venv\Scripts\python.exe -m sim.dry_run --all --agent buggy
```

Judge-only tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_judge_bot.py tests\test_judge_rules.py tests\test_judge_offline.py -q
```

## Call Artifacts

Each completed call writes a run directory:

```text
runs/scenario_02_<call_id>/
  recording.mp3             # live mixed recording created by the app
  live_agent.wav            # raw inbound/callee audio
  live_patient.wav          # raw outbound/patient audio
  transcript_live.json      # transcript captured during the call
  transcript_final.json     # optional final transcript when provider recording exists
  judge_output.json         # PASS / WEAKNESS / BUG / ERROR
  call_meta.json            # provider status and call metadata
  startup_timeline.json     # call startup timing across provider + app
  latency_debug.json        # per-turn response latency
  telnyx_ws_events.jsonl    # Telnyx websocket media/control diagnostics
```

`run.py` prints the transcript, startup timing, response latency summary, and judge verdict when the call finishes.

## Judge Behavior

The judge reads:

- the scenario card,
- expected agent behavior,
- bug conditions,
- hidden trap/end condition,
- and the transcript.

It returns:

```json
{
  "verdict": "PASS | WEAKNESS | BUG | ERROR",
  "severity": "high | medium | low | null",
  "summary": "...",
  "evidence": "...",
  "bug_description": "...",
  "expected_behavior": "...",
  "call_reference": "..."
}
```

The judge is hybrid: deterministic rules catch scenario-critical failures, and the LLM judge supplies explanation and handles edge cases. Rule-triggered bug conditions override weaker LLM verdicts.

## Startup And Latency Diagnostics

`startup_timeline.json` answers: "Was the delay Telnyx/PSTN/ngrok, or our app?"

Important deltas:

| Field | Meaning |
| --- | --- |
| `initiated_to_answered_ms` | Ring/pickup time before the call is answered |
| `answered_to_streaming_started_ms` | Telnyx media-stream setup after pickup |
| `ws_start_to_first_outbound_media_ms` | App-side time from WebSocket start to first bot audio |
| `call_answered_to_first_bot_audio_ms` | Best caller-perceived startup metric |
| `pipeline_build_duration_ms` | Cold-start/pipeline build cost |
| `total_initiated_to_first_bot_audio_ms_incl_ring` | Total time including ring/answer |

`latency_debug.json` answers: "How long did each bot response take?"

Important fields:

| Field | Meaning |
| --- | --- |
| `stt_committed` | Final agent transcript was committed |
| `first_llm_token` | First LLM token arrived |
| `tts_started` | TTS began |
| `first_audio_out` | First patient audio frame left the app |
| `stt_to_first_audio_ms` | Perceived response latency |

## Telnyx Playback Diagnostics

If the bot appears in the transcript but the caller hears silence, inspect:

```text
runs/.../telnyx_ws_events.jsonl
```

Signals:

- `media_first` or `media_summary`: patient audio frames were serialized toward Telnyx.
- `has_stream_id: true`: outbound media included the stream id.
- inbound `error`: Telnyx rejected something.
- no outbound media: the issue is inside the app before the transport.

Known-good Telnyx call payload sends only the minimal streaming fields by default:

```text
stream_track=inbound_track
stream_bidirectional_mode=rtp
stream_bidirectional_codec=PCMU
```

The extra Telnyx knobs are opt-in because some combinations caused Telnyx `90046` connection failures. Only set these when explicitly A/B testing:

```text
TELNYX_STREAM_BIDIRECTIONAL_TARGET_LEGS=both|opposite
TELNYX_STREAM_BIDIRECTIONAL_SAMPLING_RATE=8000
TELNYX_STREAM_ESTABLISH_BEFORE_CALL_ORIGINATE=true|false
TELNYX_SEND_SILENCE_WHEN_IDLE=true|false
```

Audio probe mode bypasses the LLM and TTS and sends a test tone:

```powershell
$env:TELNYX_AUDIO_PROBE="1"
.\.venv\Scripts\python.exe run.py --scenario 1 --test-target
Remove-Item Env:\TELNYX_AUDIO_PROBE
```

If the caller hears the tone, Telnyx playback works and the issue is TTS/timing. If the caller hears silence, debug Telnyx/ngrok/websocket configuration.

## UI

Start the server and open the root page:

```powershell
.\.venv\Scripts\python.exe run.py --server
```

Then visit:

```text
http://127.0.0.1:8000
```

The UI exposes scenarios, run history, transcripts, judge results, MP3 playback/download, and debug artifacts.

## Final Submission Checklist

- `.env` is filled locally and never committed.
- `TARGET_PHONE` is exactly `+18054398008`.
- `VERIFIED_TEST_PHONE` is used for development calls.
- ngrok is running and `BASE_URL` matches the public HTTPS URL.
- Offline tests pass.
- At least one live call produces `recording.mp3`, `transcript_live.json`, and `judge_output.json`.
- The bug report is generated with `run.py --report`.

