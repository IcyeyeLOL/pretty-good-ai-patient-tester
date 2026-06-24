from __future__ import annotations

import json
from pathlib import Path

from scenarios.scenario_cards import get_scenario
from src import judge_bot


def test_format_timestamp():
    assert judge_bot._format_timestamp(90000) == "1:30"
    assert judge_bot._format_timestamp(None) == "?:??"


def test_speaker_name_maps_assemblyai_labels():
    assert judge_bot._speaker_name("A", speaker_a_is_patient=True) == "patient"
    assert judge_bot._speaker_name("B", speaker_a_is_patient=True) == "agent"
    assert judge_bot._speaker_name("A", speaker_a_is_patient=False) == "agent"
    assert judge_bot._speaker_name("patient", speaker_a_is_patient=False) == "patient"


def test_transcript_to_text_formats_turns():
    transcript = {
        "speaker_A_is_patient": True,
        "turns": [
            {"speaker": "A", "start_ms": 90000, "text": "Hi there"},
            {"speaker": "B", "start_ms": 95000, "text": "How can I help?"},
        ],
    }

    assert judge_bot.transcript_to_text(transcript) == (
        "[1:30] patient: Hi there\n[1:35] agent: How can I help?"
    )


def test_transcript_to_text_falls_back_to_full_text():
    assert judge_bot.transcript_to_text({"full_text": "plain transcript"}) == "plain transcript"


def test_evaluate_returns_and_saves_mocked_llm_result(monkeypatch, isolated_cwd):
    scenario = get_scenario(1)
    assert scenario is not None
    llm_result = {
        "scenario_id": 1,
        "scenario_name": scenario.name,
        "verdict": "PASS",
        "severity": None,
        "summary": "Agent completed scheduling safely.",
        "evidence": None,
        "bug_description": None,
        "expected_behavior": None,
        "call_reference": None,
    }
    monkeypatch.setattr(judge_bot, "_call_judge_llm", lambda _prompt: json.dumps(llm_result))

    result = judge_bot.evaluate(
        {"call_sid": "CA123", "turns": [{"speaker": "patient", "start_ms": 0, "text": "Hi"}]},
        scenario,
    )

    assert result["verdict"] == "PASS"
    assert result["scenario_id"] == 1
    assert result["call_sid"] == "CA123"
    assert (Path("runs") / "scenario_01_CA123" / "judge_output.json").exists()


def test_evaluate_returns_error_on_invalid_json(monkeypatch, isolated_cwd):
    scenario = get_scenario(1)
    assert scenario is not None
    monkeypatch.setattr(judge_bot, "_call_judge_llm", lambda _prompt: "not json")

    result = judge_bot.evaluate({"call_sid": "CA123", "turns": []}, scenario)

    assert result["verdict"] == "ERROR"
    assert result["call_sid"] == "CA123"
    assert result["raw_response"] == "not json"


def test_generate_bug_report_no_findings(isolated_cwd):
    report_path = Path(judge_bot.generate_bug_report([]))

    assert report_path.exists()
    assert "No bugs or weaknesses" in report_path.read_text(encoding="utf-8")


def test_generate_bug_report_includes_bug_details(isolated_cwd):
    report_path = Path(
        judge_bot.generate_bug_report(
            [
                {
                    "scenario_id": 4,
                    "scenario_name": "Urgent Symptoms Disguised as Scheduling",
                    "call_sid": "CA123",
                    "verdict": "BUG",
                    "severity": "high",
                    "summary": "Agent scheduled despite urgent symptoms.",
                    "evidence": "I can book you next week.",
                    "expected_behavior": "Recommend emergency care.",
                }
            ]
        )
    )
    report = report_path.read_text(encoding="utf-8")

    assert "Urgent Symptoms Disguised as Scheduling" in report
    assert "Severity: high" in report
    assert "I can book you next week." in report
