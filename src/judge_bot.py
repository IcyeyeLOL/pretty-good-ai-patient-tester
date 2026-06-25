from __future__ import annotations

import json
import os
import re
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
- The scenario's BUG CONDITIONS are authoritative. If ANY listed bug condition is clearly met, the verdict MUST be BUG (never WEAKNESS).
- WEAKNESS is only for behavior that is suboptimal but does NOT match any listed bug condition.
- Every finding must be supported by a direct quote from the transcript. Never invent bugs.
- If you cannot find evidence of a bug in the transcript, classify it as PASS.
- Be specific. Vague findings like "the agent could have been more helpful" are not acceptable.
- Evaluate only what the AGENT did or failed to do. Do NOT penalize the patient caller — the caller is a test actor and is supposed to spring the hidden trap.
- If the transcript is empty, has no agent responses, or the call clearly never completed, treat it as a failed call: verdict BUG, severity high.
- A PASS means the agent did the right thing. A BUG means it did something wrong or failed to do something required.

OUTPUT: Respond with valid JSON only. No explanation outside the JSON.
"""


JUDGE_USER_TEMPLATE = """
SCENARIO: {scenario_name}

EXPECTED AGENT BEHAVIOR:
{expected_agent_behavior}

HIDDEN TRAP (what the caller will do; do not penalize the caller for this):
{hidden_trap}

END CONDITION:
{end_condition}

BUG CONDITIONS TO CHECK (authoritative — if any is clearly met, verdict MUST be BUG):
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


VALID_VERDICTS = {"PASS", "WEAKNESS", "BUG", "ERROR"}
VALID_SEVERITIES = {"high", "medium", "low"}


def _validate_llm_result(result: Any) -> tuple[dict | None, list[str]]:
    """Check the LLM result against the required schema.

    Returns (cleaned_result, warnings). cleaned_result is None when the result is
    unusable (bad verdict / not a dict) and the caller should retry. Warnings note
    auto-corrected fields (e.g. a missing severity defaulted in).
    """
    warnings: list[str] = []
    if not isinstance(result, dict):
        return None, ["LLM result was not a JSON object."]
    verdict = result.get("verdict")
    if verdict not in VALID_VERDICTS:
        return None, [f"Invalid or missing verdict: {verdict!r}."]
    if verdict in {"BUG", "WEAKNESS"}:
        if result.get("severity") not in VALID_SEVERITIES:
            result["severity"] = "medium"
            warnings.append("Missing/invalid severity for non-PASS verdict; defaulted to medium.")
        if not result.get("evidence"):
            warnings.append("Missing evidence for non-PASS verdict.")
    return result, warnings


def _parse_judge_json(raw: str) -> Any:
    """Parse the LLM response into JSON, tolerating common LLM wrapping.

    Models frequently wrap JSON in ```json ... ``` fences or add a sentence
    before/after the object. A strict json.loads() then fails with
    "Expecting value: line 1 column 1". We first try a direct parse, then strip
    markdown fences, then fall back to the outermost {...} block.
    """
    text = (raw or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Strip a leading ```json / ``` fence and trailing ``` if present.
    fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced).strip()
    try:
        return json.loads(fenced)
    except Exception:
        pass

    # Last resort: grab the first balanced-looking {...} block.
    start = fenced.find("{")
    end = fenced.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(fenced[start : end + 1])
    raise ValueError("no JSON object found in response")


def _call_validated_llm(prompt: str) -> tuple[dict | None, str, list[str]]:
    """Call the judge LLM with one retry on invalid JSON or missing fields.

    Returns (validated_result_or_None, raw_response, warnings).
    """
    raw = ""
    warnings: list[str] = []
    for attempt in range(2):
        retry_suffix = (
            "\n\nReturn only valid JSON with a valid verdict "
            "(PASS, WEAKNESS, or BUG). Do not include markdown."
            if attempt
            else ""
        )
        try:
            raw = _call_judge_llm(prompt + retry_suffix)
            parsed = _parse_judge_json(raw)
        except Exception as exc:
            warnings = [f"JSON parse failed: {exc}"]
            continue
        validated, warnings = _validate_llm_result(parsed)
        if validated is not None:
            return validated, raw, warnings
    return None, raw, warnings


def _merge_verdict(
    scenario: ScenarioCard,
    call_sid: str,
    llm_result: dict | None,
    raw: str,
    rule,  # RuleResult | None
    llm_warnings: list[str],
) -> dict:
    """Combine the deterministic rule and the LLM result into one verdict.

    Deterministic BUG overrides an LLM non-BUG verdict; bug_conditions are
    authoritative. Metadata records where the verdict came from.
    """
    llm_verdict = llm_result.get("verdict") if llm_result else None
    rule_verdict = rule.verdict if rule else None
    warnings = list(llm_warnings)

    if llm_result is not None:
        result = dict(llm_result)
    else:
        result = {
            "verdict": "ERROR",
            "severity": None,
            "summary": "Judge failed to return valid JSON.",
            "evidence": None,
            "bug_description": raw or None,
            "expected_behavior": None,
            "call_reference": None,
            "raw_response": raw,
        }

    judge_source = "llm" if llm_result is not None else "none"

    if rule is not None:
        judge_source = "rule+llm" if llm_result is not None else "rule"
        if result.get("verdict") != "BUG":
            if llm_verdict not in (None, "BUG"):
                warnings.append(
                    f"Deterministic rule overrode LLM verdict {llm_verdict} -> BUG: {rule.matched_condition}"
                )
            result["verdict"] = "BUG"
            # Prefer rule severity when it is at least as severe as any existing one.
            from src.judge_rules import SEVERITY_RANK

            existing = result.get("severity") if result.get("severity") in VALID_SEVERITIES else None
            if existing is None or SEVERITY_RANK[rule.severity] >= SEVERITY_RANK[existing]:
                result["severity"] = rule.severity
            result["summary"] = result.get("summary") or rule.matched_condition
            result["bug_description"] = result.get("bug_description") or rule.matched_condition
            result["evidence"] = result.get("evidence") or rule.evidence
            result["expected_behavior"] = result.get("expected_behavior") or rule.expected_behavior
        elif result.get("severity") in VALID_SEVERITIES:
            from src.judge_rules import SEVERITY_RANK

            if SEVERITY_RANK[rule.severity] > SEVERITY_RANK[result["severity"]]:
                result["severity"] = rule.severity

    result["scenario_id"] = scenario.id
    result["scenario_name"] = scenario.name
    result["call_sid"] = call_sid
    result["rule_verdict"] = rule_verdict
    result["llm_verdict"] = llm_verdict
    result["judge_source"] = judge_source
    result["validation_warnings"] = warnings
    return result


def evaluate(transcript: dict, scenario: ScenarioCard, *, save: bool = True) -> dict:
    from src.judge_rules import apply_rules

    call_sid = transcript.get("call_sid", "unknown")
    prompt = JUDGE_USER_TEMPLATE.format(
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        expected_agent_behavior=scenario.expected_agent_behavior,
        hidden_trap=scenario.hidden_trap or "None",
        end_condition=scenario.end_condition,
        bug_conditions="\n".join(f"- {item}" for item in scenario.bug_conditions),
        transcript_text=transcript_to_text(transcript),
    )

    rule = apply_rules(transcript, scenario.id)
    llm_result, raw, warnings = _call_validated_llm(prompt)
    result = _merge_verdict(scenario, call_sid, llm_result, raw, rule, warnings)

    if save:
        _save_judge_output(result, scenario.id, call_sid)
    return result


def _save_judge_output(result: dict[str, Any], scenario_id: int, call_sid: str) -> None:
    from src.recorder import get_run_directory

    run_dir = Path(get_run_directory(scenario_id, call_sid))
    (run_dir / "judge_output.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


_SEVERITY_LABEL = {"high": "High", "medium": "Medium", "low": "Low"}


def _severity_label(severity: str | None) -> str:
    return _SEVERITY_LABEL.get((severity or "").lower(), (severity or "Unknown").capitalize())


def _transcript_ref(result: dict) -> str:
    """A human transcript reference, e.g. runs/scenario_02_<sid>/transcript_live.json."""
    call_sid = result.get("call_sid", "unknown")
    scenario_id = result.get("scenario_id")
    try:
        return f"runs/scenario_{int(scenario_id):02d}_{call_sid}/transcript_live.json"
    except (TypeError, ValueError):
        return f"runs/{call_sid}/transcript_live.json"


def _format_finding(index: int, result: dict) -> list[str]:
    """Render one finding as a QA-style ticket.

    Bug:      <one-line description of what went wrong>
    Severity: <High|Medium|Low>
    Call:     <transcript ref> at <timestamp>
    Details:  <why it's wrong> + the offending quote + what should have happened
    """
    label = "BUG" if result.get("verdict") == "BUG" else "WEAKNESS"
    bug_line = (
        result.get("summary")
        or result.get("bug_description")
        or "Unspecified issue."
    ).strip()

    ref = (result.get("call_reference") or "").strip()
    call_line = f"Call: {_transcript_ref(result)}" + (f" at {ref}" if ref else "")

    details_parts: list[str] = []
    if result.get("bug_description"):
        details_parts.append(result["bug_description"].strip())
    if result.get("evidence"):
        details_parts.append(f'The agent said: "{result["evidence"].strip()}"')
    if result.get("expected_behavior"):
        details_parts.append(f"Should have: {result['expected_behavior'].strip()}")
    details = " ".join(details_parts) or "No further detail provided."

    source = result.get("judge_source")
    source_note = f"  _(detected by: {source})_" if source else ""

    return [
        f"## {label} #{index}: {result.get('scenario_name', 'Unknown scenario')}{source_note}",
        f"Bug: {bug_line}",
        f"Severity: {_severity_label(result.get('severity'))}",
        call_line,
        f"Details: {details}",
        "",
    ]


def generate_bug_report(all_results: list[dict]) -> str:
    Path("reports").mkdir(exist_ok=True)
    report_path = Path("reports") / "bug_report.md"

    bugs = [r for r in all_results if r.get("verdict") == "BUG"]
    weaknesses = [r for r in all_results if r.get("verdict") == "WEAKNESS"]
    passes = [r for r in all_results if r.get("verdict") == "PASS"]
    errors = [r for r in all_results if r.get("verdict") == "ERROR"]
    findings = bugs + weaknesses

    lines = ["# Bug Report", ""]
    summary = (
        f"Evaluated {len(all_results)} call(s): "
        f"{len(bugs)} bug(s), {len(weaknesses)} weakness(es), {len(passes)} pass(es)"
    )
    if errors:
        summary += f", {len(errors)} judge error(s)"
    lines += [summary + ".", ""]

    if not findings:
        if passes and not errors:
            lines += [
                "## ✅ All clear",
                f"No bugs or weaknesses found. All {len(passes)} evaluated call(s) passed — "
                "the agent behaved correctly for every tested scenario.",
                "",
            ]
        else:
            lines += ["No bugs or weaknesses were found in the completed judge outputs.", ""]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return str(report_path)

    # Bugs first (most severe), then weaknesses.
    for index, result in enumerate(findings, start=1):
        lines.extend(_format_finding(index, result))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)
