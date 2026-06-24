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
OPENING_DELAY_SECONDS=3
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

## Outputs

Each completed call writes artifacts under:

```text
runs/scenario_04_CA1234567890abcdef/
  recording.mp3
  transcript_live.json
  transcript_final.json
  judge_output.json
  call_meta.json
```

The compiled report is written to:

```text
reports/bug_report.md
```

Only use one caller number for all calls and report it in E.164 format in the final submission.
