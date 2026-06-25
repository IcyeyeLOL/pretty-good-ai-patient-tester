# AI Patient Tester

Python voice bot for the Pretty Good AI engineering challenge. It places outbound calls to the assessment number, acts as a realistic patient across 10 scenarios, records/transcribes each call, and uses a separate judge pass to identify bugs or quality issues.

## Setup

1. Clone the repo.
2. Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Create your environment file:

```bash
cp .env.example .env
```

4. Fill in `.env` with your Telnyx or Twilio, Anthropic, Deepgram, AssemblyAI, and Cartesia credentials. `TARGET_PHONE` must remain:

```bash
TARGET_PHONE=+18054398008
```

The default LLM provider is Anthropic:

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_claude_api_key
ANTHROPIC_MODEL=claude-sonnet-4-6
```

Cartesia TTS should use an active Sonic model:

```bash
CARTESIA_MODEL=sonic-2
OPENING_DELAY_SECONDS=0.5
DEEPGRAM_ENDPOINTING_MS=200
USER_AGGREGATION_TIMEOUT_SECONDS=0.45
AGENT_COMPLETE_DEBOUNCE_SECONDS=0.2
AGENT_FRAGMENT_DEBOUNCE_SECONDS=1.35
AGENT_FRAGMENT_MAX_HOLD_SECONDS=2.0
```

The default telephony provider is Telnyx:

```bash
TELEPHONY_PROVIDER=telnyx
TELNYX_API_KEY=your_telnyx_api_key
TELNYX_CONNECTION_ID=your_telnyx_voice_application_id
TELNYX_OUTBOUND_VOICE_PROFILE_ID=your_telnyx_outbound_voice_profile_id
TELNYX_PHONE_NUMBER=+15551234567
```

5. Start ngrok in one terminal:

```bash
ngrok http 8000
```

6. Copy the HTTPS ngrok URL into `.env`:

```bash
BASE_URL=https://abc123.ngrok-free.app
```

7. Run the first test call from another terminal:

```bash
.venv\Scripts\python.exe run.py --scenario 1
```

## Usage

List scenarios:

```bash
.venv\Scripts\python.exe run.py --list
```

Run one scenario:

```bash
.venv\Scripts\python.exe run.py --scenario 4
```

Run one scenario against a verified test phone number:

```bash
.venv\Scripts\python.exe run.py --scenario 1 --test-target
```

`--test-target` uses `VERIFIED_TEST_PHONE` and leaves `TARGET_PHONE` locked to the final assessment number.

Run all 10 scenarios:

```bash
.venv\Scripts\python.exe run.py --all
```

Start only the FastAPI server for manual testing:

```bash
.venv\Scripts\python.exe run.py --server
```

On Windows, the helper script starts the server in the background and writes logs to `logs/`:

```powershell
.\scripts\start-dev-server.ps1
.\scripts\start-dev-server.ps1 -Restart
```

Generate a bug report from completed judge outputs:

```bash
.venv\Scripts\python.exe run.py --report
```

## Validating the patient bot before spending call budget

Grading priority #1 is whether the patient bot holds a coherent conversation and
actively steers each scenario toward its test outcome. Two cheap layers verify
that *before* placing real calls:

1. **Static memo fidelity** (`tests/test_prompt_fidelity.py`): asserts every
   scenario prompt actually contains its hidden trap, key steering lines, and
   trap timing — so no memo detail is silently dropped from what the model sees.

2. **Offline conversation dry-run** (`sim/dry_run.py`): runs the real patient-bot
   LLM through a full text conversation against a simulated office agent, then
   scores **both sides** of every run:
   - **Patient** (the bot we submit): a regex in-character pre-filter plus an LLM
     patient-adherence judge that scores whether the caller stayed in character,
     sprang the scenario's hidden trap at the right moment, and steered the call.
     Verdict is `GOOD` / `WEAK` / `OOC`.
   - **Agent** (the system under test): the same post-call Judge used in
     production. The simulated agent runs `good` (should yield PASS) or `buggy`
     (commits the scenario's failure; should yield BUG), which also verifies the
     Judge actually catches the planted bug.

   Costs pennies, so iterate here before the ~$20 of real calls.

```bash
.venv\Scripts\python.exe -m sim.dry_run --scenario 4 --agent both
.venv\Scripts\python.exe -m sim.dry_run --all --agent buggy
```

Each dry-run writes a transcript and verdict under `runs/sim_*/`.

## Diagnosing Telnyx playback (caller hears silence)

The local `recording.mp3` / `live_patient.wav` only prove the app *generated*
patient audio — not that Telnyx played it into the call. These tools close that gap:

1. **Outbound/inbound websocket log** (`runs/.../telnyx_ws_events.jsonl`): the
   Telnyx serializer is wrapped to log every outbound media frame (count, byte
   length, codec, whether `stream_id` is present) and every inbound control event
   — including Telnyx `error` and `mark` frames, which the stock serializer
   silently drops. If you see `media_first`/`media_summary` lines, frames are
   leaving the app; if you see an `in`/`error` line, Telnyx rejected something.

2. **`stream_id` injection** (`TELNYX_INJECT_STREAM_ID=1`, default on): the stock
   serializer omits `stream_id` on outbound media; some Telnyx bidirectional
   setups need it to route playback. Set to `0` to A/B test the difference.

3. **Explicit bidirectional config**: `TELNYX_STREAM_BIDIRECTIONAL_TARGET_LEGS`
   (default `both`), `..._SAMPLING_RATE`, `..._MODE`, `..._CODEC`,
   `TELNYX_STREAM_ESTABLISH_BEFORE_CALL_ORIGINATE`, `TELNYX_SEND_SILENCE_WHEN_IDLE`.
   Previously these relied on Telnyx defaults. Test `both` vs `opposite`.

4. **Audio-path probe** (`TELNYX_AUDIO_PROBE=1`): on connect, send a known PCMU
   test tone straight to Telnyx, bypassing LLM/TTS. If the caller **hears the
   tone**, the playback path works and the bug is in TTS/timing. If the caller
   **hears silence**, the bug is the Telnyx payload/codec/target-leg/stream_id
   setup. This is the definitive bisection. Place one call with it enabled:

   ```bash
   TELNYX_AUDIO_PROBE=1 .venv\Scripts\python.exe run.py --scenario 1
   ```

## Locating the startup delay (is the ~10s before Telnyx media, or in our app?)

Every call writes `runs/.../startup_timeline.json` with wall-clock marks
collected across all components (CLI, Telnyx webhooks, websocket handler,
pipeline, serializer) on one comparable clock, plus a computed delta report:

```json
{
  "marks": { "call_initiated": ..., "call_answered": ..., "streaming_started": ...,
             "ws_accepted": ..., "ws_start_event": ..., "opener_scheduled": ...,
             "opener_queued": ..., "first_outbound_audio_before_transport": ...,
             "first_serialized_media": ... },
  "deltas": {
    "initiated_to_answered_ms": ...,            // Telnyx/PSTN ring+pickup
    "answered_to_streaming_started_ms": ...,    // Telnyx media-stream setup
    "streaming_started_to_ws_start_ms": ...,    // websocket establishment (incl. ngrok)
    "ws_start_to_first_outbound_media_ms": ..., // OUR app (opener + TTS + serialize)
    "call_answered_to_first_bot_audio_ms": ..., // CALLER-PERCEIVED (post-pickup)
    "call_answered_to_opener_scheduled_ms": ...,
    "opener_scheduled_to_first_serialized_media_ms": ...,
    "ws_start_to_pipeline_build_started_ms": ...,
    "pipeline_build_duration_ms": ...,         // cold-start import/build cost
    "total_initiated_to_first_bot_audio_ms_incl_ring": ... // includes ring/answer
  }
}
```

Read it top-down: if `initiated_to_answered_ms` and `..._to_streaming_started_ms`
dominate, the delay is Telnyx/PSTN/ngrok, not us. If
`ws_start_to_first_outbound_media_ms` dominates, the delay is app-side (opener
scheduling + first TTS). `call_answered_to_first_bot_audio_ms` is the honest
caller-perceived number (measured from pickup, not dial). `run.py` prints the
delta block plus a response-latency summary (median/max `stt_to_first_audio_ms`,
slow turns over 1800ms, and first-token latency) after each call.

The server preloads the live-pipeline imports on startup (and `run.py` hits
`/api/warmup` before the first call), so `pipeline_build_duration_ms` reflects
steady-state build cost, not one-time cold-start imports. **Start the server once
and do not restart between calls** so warmup stays effective.

### A/B tests for the transport settings

`stream_establish_before_call_originate=true` makes media safer but can slow call
connection. Toggle and compare `initiated_to_answered_ms` across two calls:

```bash
TELNYX_STREAM_ESTABLISH_BEFORE_CALL_ORIGINATE=true  .venv\Scripts\python.exe run.py --scenario 1
TELNYX_STREAM_ESTABLISH_BEFORE_CALL_ORIGINATE=false .venv\Scripts\python.exe run.py --scenario 1
```

Bidirectional playback routing (caller hears nothing): compare `both` vs `opposite`
while watching `telnyx_ws_events.jsonl` for outbound media + any `in`/`error`:

```bash
TELNYX_STREAM_BIDIRECTIONAL_TARGET_LEGS=both     .venv\Scripts\python.exe run.py --scenario 1
TELNYX_STREAM_BIDIRECTIONAL_TARGET_LEGS=opposite .venv\Scripts\python.exe run.py --scenario 1
```

Verified-explicit bidirectional defaults (see `.env.example`): `mode=rtp`,
`codec=PCMU`, `target_legs=both`, `sampling_rate=8000`, `send_silence_when_idle=true`,
and outbound media carries `stream_id` (`TELNYX_INJECT_STREAM_ID=1`).

`OPENING_DELAY_SECONDS` stays at `0.5`. Only raise it if the timeline shows the
opener firing *before* media playback is ready (i.e. `opener_queued` lands well
ahead of `streaming_started`/`ws_start_event`).

### Test-tone diagnostic

`TELNYX_AUDIO_PROBE=1` sends a tone instead of bot speech (bypasses LLM/TTS). Tone
heard quickly => Telnyx playback path is fine; silence => Telnyx/ngrok/websocket
config is the problem. See the section above.

## Outputs

Each completed call writes artifacts under:

```text
runs/scenario_04_CA1234567890abcdef/
  recording.mp3          # Twilio only; Telnyx evaluates the live transcript
  transcript_live.json
  transcript_final.json  # Twilio only; AssemblyAI speaker-diarized pass
  judge_output.json
  call_meta.json
  latency_debug.json     # per-turn timing checkpoints (STT -> LLM -> TTS -> audio)
```

With Telnyx there is no recording leg, so `recording.mp3` and `transcript_final.json`
are absent and the judge runs against `transcript_live.json`.

`latency_debug.json` has two parts. `events` records one-shot, call-level checkpoints
(`opener_scheduled_at`, `opener_queued_at`, `first_inbound_audio`,
`first_outbound_audio_before_transport`) — used to confirm the opener was actually
queued and whether we ever heard inbound audio. `turns` records, per patient turn,
milliseconds from call start to each checkpoint (agent speech started, STT committed,
LLM response start, first LLM token, TTS started, first outbound audio, patient turn
logged) plus a derived `stt_to_first_audio_ms` perceived-latency figure.

The opening is **callee-gated**: the bot waits to hear the callee's first speech and
responds deterministically (no LLM round-trip on "hello"). A blind opener fires only
as a silence fallback after `OPENING_DELAY_SECONDS`; if the callee then re-greets, the
bot re-greets back ("Hi, can you hear me? ...") instead of going silent. A bare
"Hello?" mid-call gets a universal recovery line, never a guess about the name.

The compiled report is written to:

```text
reports/bug_report.md
```

Only use one caller number for all calls and report it in E.164 format in the final submission.
