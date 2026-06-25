"""Offline judge validation — no live calls, no real LLM/Telnyx.

Two layers:
1. Synthetic fixtures (tests/fixtures/judge/*.json) exercise the deterministic
   rules + hybrid merge with the LLM stubbed to a deliberately-wrong verdict, so
   we prove the rules carry the verdict on their own.
2. Replay tests load real captured transcripts (runs/.../transcript_live.json)
   and assert the hardened judge reaches the right verdict deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scenarios.scenario_cards import get_scenario
from src import judge_bot
from src.judge_rules import apply_rules

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "judge"
RUNS_DIR = Path(__file__).parent.parent / "runs"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture(name: str) -> dict:
    return _load(FIXTURE_DIR / name)


# ── deterministic rules on synthetic fixtures ─────────────────────────────────

def test_fixture_s2_pass_closed_weekends_no_rule():
    assert apply_rules(_fixture("scenario_02_pass_closed_weekends.json"), 2) is None


def test_fixture_s2_books_saturday_is_bug_high():
    rule = apply_rules(_fixture("scenario_02_bug_books_saturday.json"), 2)
    assert rule is not None and rule.verdict == "BUG" and rule.severity == "high"


def test_fixture_s2_no_weekend_explanation_is_bug_medium():
    rule = apply_rules(_fixture("scenario_02_bug_no_closed_weekend_explanation.json"), 2)
    assert rule is not None and rule.verdict == "BUG" and rule.severity == "medium"


def test_fixture_s3_refuses_cancel_is_bug_high():
    rule = apply_rules(_fixture("scenario_03_bug_refuses_cancel.json"), 3)
    assert rule is not None and rule.verdict == "BUG" and rule.severity == "high"


def test_fixture_s4_no_escalation_is_bug_high():
    rule = apply_rules(_fixture("scenario_04_bug_no_emergency_escalation.json"), 4)
    assert rule is not None and rule.verdict == "BUG" and rule.severity == "high"


# ── scenario 1 rule (collect details before confirming) ───────────────────────

def test_scenario_1_books_without_identity_is_bug(monkeypatch, isolated_cwd):
    scenario = get_scenario(1)
    monkeypatch.setattr(judge_bot, "_call_judge_llm", lambda _p: json.dumps({"verdict": "PASS"}))
    transcript = {
        "scenario_id": 1,
        "call_sid": "SIMs1",
        "speaker_A_is_patient": True,
        "turns": [
            {"speaker": "patient", "text": "I'd like to schedule a checkup.", "start_ms": 0},
            {"speaker": "agent", "text": "Sure! How about next Tuesday at 9am?", "start_ms": 1},
            {"speaker": "patient", "text": "That works.", "start_ms": 2},
            {"speaker": "agent", "text": "Great, you're all set for Tuesday at 9am.", "start_ms": 3},
        ],
    }
    result = judge_bot.evaluate(transcript, scenario, save=False)
    assert result["verdict"] == "BUG"
    assert result["rule_verdict"] == "BUG"
    assert result["judge_source"] == "rule+llm"


def test_scenario_1_collects_identity_no_rule():
    transcript = {
        "scenario_id": 1,
        "call_sid": "SIMs1ok",
        "speaker_A_is_patient": True,
        "turns": [
            {"speaker": "patient", "text": "I'd like to schedule a checkup.", "start_ms": 0},
            {"speaker": "agent", "text": "Sure, may I have your name and date of birth?", "start_ms": 1},
            {"speaker": "patient", "text": "Maria Johnson, March 7 1991.", "start_ms": 2},
            {"speaker": "agent", "text": "Thanks, you're all set for Tuesday at 9am.", "start_ms": 3},
        ],
    }
    assert apply_rules(transcript, 1) is None


# ── hybrid evaluate: rule overrides a wrong LLM on a fixture ───────────────────

def test_hybrid_s2_no_explanation_overrides_llm_weakness(monkeypatch, isolated_cwd):
    scenario = get_scenario(2)
    monkeypatch.setattr(
        judge_bot,
        "_call_judge_llm",
        lambda _p: json.dumps({"verdict": "WEAKNESS", "severity": "low",
                               "summary": "Could have explained more.", "evidence": "No."}),
    )
    result = judge_bot.evaluate(
        _fixture("scenario_02_bug_no_closed_weekend_explanation.json"), scenario, save=False
    )
    assert result["verdict"] == "BUG"
    assert result["severity"] == "medium"
    assert result["llm_verdict"] == "WEAKNESS"
    assert result["judge_source"] == "rule+llm"


# ── markdown-fenced LLM JSON is parsed (regression for "Expecting value") ─────

def test_parse_judge_json_strips_markdown_fence():
    raw = '```json\n{"verdict": "BUG", "severity": "high"}\n```'
    parsed = judge_bot._parse_judge_json(raw)
    assert parsed["verdict"] == "BUG"


def test_parse_judge_json_handles_prose_around_object():
    raw = 'Here is my evaluation:\n{"verdict": "PASS"}\nLet me know if you need more.'
    assert judge_bot._parse_judge_json(raw)["verdict"] == "PASS"


def test_evaluate_accepts_fenced_llm_output(monkeypatch, isolated_cwd):
    scenario = get_scenario(1)
    fenced = '```json\n' + json.dumps({
        "verdict": "BUG", "severity": "high",
        "summary": "Provided a slot without verifying availability.",
        "evidence": "I have you down for Tuesday at 9AM.",
    }) + '\n```'
    monkeypatch.setattr(judge_bot, "_call_judge_llm", lambda _p: fenced)
    transcript = {
        "scenario_id": 1, "call_sid": "SIMfence", "speaker_A_is_patient": True,
        "turns": [
            {"speaker": "patient", "text": "I'd like a checkup.", "start_ms": 0},
            {"speaker": "agent", "text": "Name and date of birth?", "start_ms": 1},
            {"speaker": "patient", "text": "Maria Johnson, March 7 1991.", "start_ms": 2},
            {"speaker": "agent", "text": "I have you down for Tuesday at 9AM.", "start_ms": 3},
        ],
    }
    result = judge_bot.evaluate(transcript, scenario, save=False)
    assert result["verdict"] == "BUG"
    assert result["judge_source"] == "llm"
    assert result["validation_warnings"] == []


# ── replay tests from real captured transcripts ───────────────────────────────

def _find_run_transcript(prefix: str) -> Path | None:
    matches = sorted(RUNS_DIR.glob(f"{prefix}*/transcript_live.json"))
    return matches[0] if matches else None


def test_replay_scenario_02_becomes_bug_medium():
    path = _find_run_transcript("scenario_02_e6aef33a")
    if path is None:
        pytest.skip("captured scenario 2 transcript not present")
    rule = apply_rules(_load(path), 2)
    assert rule is not None and rule.verdict == "BUG" and rule.severity == "medium"


def test_replay_scenario_03_stays_bug_high():
    path = _find_run_transcript("scenario_03_6ef81a94")
    if path is None:
        pytest.skip("captured scenario 3 transcript not present")
    rule = apply_rules(_load(path), 3)
    assert rule is not None and rule.verdict == "BUG" and rule.severity == "high"
