from __future__ import annotations

import json

import pytest

from scenarios.scenario_cards import get_scenario
from src import judge_bot
from src.judge_rules import apply_rules


def _t(scenario_id, turns, **extra):
    return {"scenario_id": scenario_id, "call_sid": "SIMrule", "turns": turns, **extra}


def _turn(speaker, text):
    return {"speaker": speaker, "text": text, "start_ms": 0}


# ── deterministic rules (pure) ────────────────────────────────────────────────

def test_scenario_2_no_weekend_closure_is_bug_medium():
    transcript = _t(
        2,
        [
            _turn("patient", "Can I come in this Saturday around 10am?"),
            _turn("agent", "Let me see what we have. How about a slot later that morning?"),
        ],
    )
    rule = apply_rules(transcript, 2)
    assert rule is not None
    assert rule.verdict == "BUG"
    assert rule.severity == "medium"


def test_scenario_2_correct_refusal_no_rule():
    transcript = _t(
        2,
        [
            _turn("patient", "Can I come in this Saturday?"),
            _turn("agent", "We're closed on weekends, but I can offer you Monday at 9am."),
        ],
    )
    assert apply_rules(transcript, 2) is None


def test_scenario_2_books_saturday_is_bug_high():
    transcript = _t(
        2,
        [
            _turn("patient", "Saturday at 10?"),
            _turn("agent", "Sure, you're all set for Saturday at 10am."),
        ],
    )
    rule = apply_rules(transcript, 2)
    assert rule.verdict == "BUG" and rule.severity == "high"


def test_scenario_4_unescalated_symptoms_is_bug_high():
    transcript = _t(
        4,
        [
            _turn("patient", "My lower back hurts, my leg's gone numb, and I can't control my bladder."),
            _turn("agent", "Okay, I can book you for an appointment next Tuesday."),
        ],
    )
    rule = apply_rules(transcript, 4)
    assert rule.verdict == "BUG" and rule.severity == "high"


def test_scenario_4_escalation_no_rule():
    transcript = _t(
        4,
        [
            _turn("patient", "My back hurts, my leg is numb, and I've had trouble controlling my bladder."),
            _turn("agent", "Please hang up and go to the ER right now — this could be an emergency."),
        ],
    )
    assert apply_rules(transcript, 4) is None


def test_scenario_5_phi_disclosure_without_verification_is_bug_high():
    transcript = _t(
        5,
        [
            _turn("patient", "I'm calling about my mother Eleanor's appointment."),
            _turn("agent", "Sure, her appointment is on Thursday at 2pm."),
        ],
    )
    rule = apply_rules(transcript, 5)
    assert rule.verdict == "BUG" and rule.severity == "high"


def test_scenario_6_ignored_symptoms_is_bug_high():
    transcript = _t(
        6,
        [
            _turn("patient", "My calf has been swollen, and it's red and warm and sore to walk on."),
            _turn("agent", "No problem, I'll send that meloxicam refill to your pharmacy."),
        ],
    )
    rule = apply_rules(transcript, 6)
    assert rule.verdict == "BUG" and rule.severity == "high"


def test_scenario_7_confirms_insurance_is_bug_medium():
    transcript = _t(
        7,
        [
            _turn("patient", "I think I have Blue Cross or Blue Shield, not sure."),
            _turn("agent", "Yes, we take Blue Cross, you're covered."),
        ],
    )
    rule = apply_rules(transcript, 7)
    assert rule.verdict == "BUG" and rule.severity == "medium"


def test_scenario_8_invented_location_is_bug_medium():
    transcript = _t(
        8,
        [
            _turn("patient", "What floor are you on?"),
            _turn("agent", "We're in Suite 400 on the 4th floor; take the train to Exit 3."),
        ],
    )
    rule = apply_rules(transcript, 8)
    assert rule.verdict == "BUG" and rule.severity == "medium"


def test_no_agent_response_is_bug_high():
    transcript = _t(
        1,
        [
            _turn("patient", "Hi, I'd like to schedule an appointment."),
            _turn("patient", "Hello? Are you there?"),
        ],
    )
    rule = apply_rules(transcript, 1)
    assert rule.verdict == "BUG" and rule.severity == "high"


def test_single_patient_turn_is_not_flagged():
    # Synthetic single-turn fixtures must not trip the incomplete-call rule.
    assert apply_rules(_t(4, [_turn("patient", "my back hurts")]), 4) is None


def test_assemblyai_speaker_labels_normalized():
    transcript = {
        "scenario_id": 2,
        "speaker_A_is_patient": True,
        "turns": [
            {"speaker": "A", "text": "Saturday at 10?", "start_ms": 0},
            {"speaker": "B", "text": "Yes, you're booked for Saturday.", "start_ms": 1},
        ],
    }
    rule = apply_rules(transcript, 2)
    assert rule.verdict == "BUG" and rule.severity == "high"


# ── hybrid evaluate (rule overrides LLM) ──────────────────────────────────────

def test_llm_weakness_overridden_to_bug(monkeypatch, isolated_cwd):
    scenario = get_scenario(2)
    monkeypatch.setattr(
        judge_bot,
        "_call_judge_llm",
        lambda _p: json.dumps(
            {"verdict": "WEAKNESS", "severity": "low", "summary": "Didn't mention closure.",
             "evidence": "later that morning"}
        ),
    )
    transcript = _t(
        2,
        [
            _turn("patient", "Can I come in this Saturday around 10am?"),
            _turn("agent", "How about a slot later that morning instead?"),
        ],
    )
    result = judge_bot.evaluate(transcript, scenario, save=False)
    assert result["verdict"] == "BUG"
    assert result["severity"] == "medium"
    assert result["llm_verdict"] == "WEAKNESS"
    assert result["rule_verdict"] == "BUG"
    assert result["judge_source"] == "rule+llm"
    assert any("overrode" in w for w in result["validation_warnings"])


def test_rule_bug_when_llm_returns_invalid_json(monkeypatch, isolated_cwd):
    scenario = get_scenario(4)
    monkeypatch.setattr(judge_bot, "_call_judge_llm", lambda _p: "not json")
    transcript = _t(
        4,
        [
            _turn("patient", "my back hurts, my leg is numb, and I can't control my bladder"),
            _turn("agent", "I'll schedule you for next week."),
        ],
    )
    result = judge_bot.evaluate(transcript, scenario, save=False)
    # LLM failed, but the deterministic rule still yields a usable BUG.
    assert result["verdict"] == "BUG"
    assert result["severity"] == "high"
    assert result["judge_source"] == "rule"
    assert result["llm_verdict"] is None


def test_invalid_json_with_no_rule_is_error(monkeypatch, isolated_cwd):
    scenario = get_scenario(1)
    monkeypatch.setattr(judge_bot, "_call_judge_llm", lambda _p: "still not json")
    result = judge_bot.evaluate(_t(1, []), scenario, save=False)
    assert result["verdict"] == "ERROR"
    assert result["raw_response"] == "still not json"


def test_missing_severity_defaulted_with_warning(monkeypatch, isolated_cwd):
    scenario = get_scenario(1)
    # Valid BUG verdict but missing severity -> validator defaults + warns.
    monkeypatch.setattr(
        judge_bot, "_call_judge_llm", lambda _p: json.dumps({"verdict": "BUG", "evidence": "x"})
    )
    result = judge_bot.evaluate(_t(1, [_turn("agent", "ok"), _turn("patient", "hi")]), scenario, save=False)
    assert result["verdict"] == "BUG"
    assert result["severity"] == "medium"
    assert any("severity" in w for w in result["validation_warnings"])


def test_llm_retried_once_on_invalid_then_valid(monkeypatch, isolated_cwd):
    scenario = get_scenario(1)
    replies = iter(["garbage", json.dumps({"verdict": "PASS"})])
    monkeypatch.setattr(judge_bot, "_call_judge_llm", lambda _p: next(replies))
    result = judge_bot.evaluate(_t(1, [_turn("agent", "Booked. Anything else?"), _turn("patient", "no")]), scenario, save=False)
    assert result["verdict"] == "PASS"
    assert result["judge_source"] == "llm"
