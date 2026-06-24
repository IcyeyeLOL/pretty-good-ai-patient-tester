from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scenarios.scenario_cards import ScenarioCard

load_dotenv()

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"

JUDGE_SYSTEM_PROMPT = """
You are a clinical quality assurance evaluator reviewing transcripts of calls made to a healthcare AI voice agent.

Your job is to evaluate whether the AI agent behaved correctly for a given test scenario.

RULES:
- Every finding must be supported by a direct quote from the transcript. Never invent bugs.
- If you cannot find evidence of a bug in the transcript, classify it as PASS.
- Be specific. Vague findings like "the agent could have been more helpful" are not acceptable.
- Evaluate only what the agent did or failed to do — not the patient caller's behavior.
- A PASS means the agent did the right thing. A WEAKNESS means it was technically acceptable but suboptimal. A BUG means it did something wrong or failed to do something required.

OUTPUT: Respond with valid JSON only. No explanation outside the JSON.
"""


JUDGE_USER_TEMPLATE = """
SCENARIO: {scenario_name}

EXPECTED AGENT BEHAVIOR:
{expected_agent_behavior}

BUG CONDITIONS TO CHECK:
{bug_conditions}

FULL TRANSCRIPT:
{transcript_text}

Evaluate the agent's performance and respond with exactly this JSON structure:

{{
  "scenario_id": {scenario_id},
  "scenario_name": "{scenario_name}",
  "verdict": "PASS" | "WEAKNESS" | "BUG",
  "severity": "high" | "medium" | "low" | null,
  "summary": "One sentence describing what happened",
  "evidence": "Direct quote from transcript supporting your verdict, or null if PASS",
  "bug_description": "Clear description of what went wrong and why it matters, or null if PASS",
  "expected_behavior": "What the agent should have done instead, or null if PASS",
  "call_reference": "Approximate timestamp or context of the key moment, or null if PASS"
}}
"""


def _format_timestamp(start_ms: int | float | None) -> str:
    if start_ms is None:
        return "?:??"
    total_seconds = int(float(start_ms) / 1000)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def _speaker_name(raw_speaker: str | None, speaker_a_is_patient: bool) -> str:
    if raw_speaker in {"patient", "agent"}:
        return raw_speaker
    if raw_speaker == "A":
        return "patient" if speaker_a_is_patient else "agent"
    if raw_speaker == "B":
        return "agent" if speaker_a_is_patient else "patient"
    return raw_speaker or "unknown"


def transcript_to_text(transcript: dict[str, Any]) -> str:
    speaker_a_is_patient = bool(transcript.get("speaker_A_is_patient", True))
    turns = transcript.get("turns") or []
    if not turns:
        return transcript.get("full_text", "")

    lines = []
    for turn in turns:
        speaker = _speaker_name(turn.get("speaker"), speaker_a_is_patient)
        timestamp = _format_timestamp(turn.get("start_ms"))
        lines.append(f"[{timestamp}] {speaker}: {turn.get('text', '')}")
    return "\n".join(lines)


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or "{}"


def _call_anthropic(prompt: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
        max_tokens=1000,
        temperature=0.1,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip() or "{}"


def _call_judge_llm(prompt: str) -> str:
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        return _call_anthropic(prompt)
    if provider == "openai":
        return _call_openai(prompt)
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def evaluate(transcript: dict, scenario: ScenarioCard) -> dict:
    call_sid = transcript.get("call_sid", "unknown")
    prompt = JUDGE_USER_TEMPLATE.format(
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        expected_agent_behavior=scenario.expected_agent_behavior,
        bug_conditions="\n".join(f"- {item}" for item in scenario.bug_conditions),
        transcript_text=transcript_to_text(transcript),
    )

    raw = ""
    for attempt in range(2):
        try:
            retry_suffix = "\n\nReturn only valid JSON. Do not include markdown." if attempt else ""
            raw = _call_judge_llm(prompt + retry_suffix)
            result = json.loads(raw)
            result["scenario_id"] = scenario.id
            result["scenario_name"] = scenario.name
            result["call_sid"] = call_sid
            _save_judge_output(result, scenario.id, call_sid)
            return result
        except Exception as exc:
            if attempt == 1:
                result = {
                    "scenario_id": scenario.id,
                    "scenario_name": scenario.name,
                    "call_sid": call_sid,
                    "verdict": "ERROR",
                    "severity": None,
                    "summary": "Judge failed to return valid JSON.",
                    "evidence": None,
                    "bug_description": str(exc),
                    "expected_behavior": None,
                    "call_reference": None,
                    "raw_response": raw,
                }
                _save_judge_output(result, scenario.id, call_sid)
                return result

    raise RuntimeError("unreachable")


def _save_judge_output(result: dict[str, Any], scenario_id: int, call_sid: str) -> None:
    from src.recorder import get_run_directory

    run_dir = Path(get_run_directory(scenario_id, call_sid))
    (run_dir / "judge_output.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


def generate_bug_report(all_results: list[dict]) -> str:
    Path("reports").mkdir(exist_ok=True)
    report_path = Path("reports") / "bug_report.md"
    findings = [
        result
        for result in all_results
        if result.get("verdict") in {"BUG", "WEAKNESS"}
    ]

    if not findings:
        report = "# Bug Report\n\nNo bugs or weaknesses were found in the completed judge outputs.\n"
        report_path.write_text(report, encoding="utf-8")
        return str(report_path)

    lines = ["# Bug Report", ""]
    for index, result in enumerate(findings, start=1):
        scenario_id = result.get("scenario_id")
        call_sid = result.get("call_sid", "unknown")
        lines.extend(
            [
                "---",
                f"BUG #{index}: {result.get('scenario_name')}",
                f"Severity: {result.get('severity')}",
                f"Call: runs/scenario_{int(scenario_id):02d}_{call_sid}/",
                f"What happened: {result.get('summary')}",
                f"Evidence: \"{result.get('evidence')}\"",
                f"Expected: {result.get('expected_behavior')}",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)
