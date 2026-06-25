from __future__ import annotations

import pytest

from sim import dry_run


def test_buggy_directive_exists_for_every_scenario():
    from scenarios.scenario_cards import get_all_scenarios

    assert set(dry_run.BUGGY_DIRECTIVE) == {s.id for s in get_all_scenarios()}


def test_agent_system_injects_buggy_override():
    good = dry_run._agent_system(2, "good")
    buggy = dry_run._agent_system(2, "buggy")

    assert "CLOSED on weekends" in good
    assert "OVERRIDE" not in good
    assert dry_run.BUGGY_DIRECTIVE[2] in buggy


def test_patient_messages_start_with_user_and_alternate():
    turns = [
        {"speaker": "patient", "text": "Hi, scheduling please."},
        {"speaker": "agent", "text": "Sure, your name?"},
        {"speaker": "patient", "text": "Maria Johnson."},
    ]
    msgs = dry_run._patient_messages(turns)

    assert msgs[0]["role"] == "user"  # Anthropic requires a leading user turn
    roles = [m["role"] for m in msgs]
    # No two consecutive roles are identical (valid alternation).
    assert all(a != b for a, b in zip(roles, roles[1:]))
    # Patient lines are 'assistant' from the patient's POV.
    assert msgs[2]["content"] == "Sure, your name?" and msgs[2]["role"] == "user"


def test_agent_messages_map_patient_to_user():
    turns = [
        {"speaker": "patient", "text": "Hello"},
        {"speaker": "agent", "text": "Hi there"},
    ]
    msgs = dry_run._agent_messages(turns)
    assert msgs[0] == {"role": "user", "content": "Hello"}
    assert msgs[1] == {"role": "assistant", "content": "Hi there"}


def test_run_conversation_assembles_transcript_and_ends_on_goodbye():
    """Drive the loop with a scripted fake chat — no API calls."""
    replies = iter(
        [
            "Sure, what is your name?",       # agent turn 1
            "Maria Johnson.",                  # patient turn 1
            "You're booked for Tuesday.",      # agent turn 2
            "Okay, great. Thank you. Bye.",    # patient turn 2 -> triggers end
        ]
    )

    def fake_chat(system, messages, *, temperature, max_tokens):
        return next(replies)

    transcript = dry_run.run_conversation(1, "good", max_turns=14, chat=fake_chat)

    speakers = [t["speaker"] for t in transcript["turns"]]
    # opening(patient) + [agent, patient] + [agent, patient]
    assert speakers == ["patient", "agent", "patient", "agent", "patient"]
    assert transcript["turns"][0]["text"]  # opening line populated
    assert transcript["agent_mode"] == "good"
    assert transcript["in_character_violations"] == []


def test_run_conversation_flags_in_character_violation():
    replies = iter(
        [
            "How can I help you today?",                 # agent
            "As an AI assistant, I cannot do that.",     # patient breaks role
            "Anything else?",                            # agent
            "Okay, great. Thank you. Bye.",              # patient ends
        ]
    )

    def fake_chat(system, messages, *, temperature, max_tokens):
        return next(replies)

    transcript = dry_run.run_conversation(1, "good", max_turns=14, chat=fake_chat)

    assert transcript["in_character_violations"], "role break should be recorded"
    # The sanitizer should have replaced the broken line with a safe fallback.
    broken_turn = transcript["turns"][2]["text"].lower()
    assert "as an ai" not in broken_turn


def _fake_patient_chat(payload):
    """Return a fixed patient-judge JSON string regardless of prompt."""
    import json

    return lambda system, messages, *, temperature, max_tokens: json.dumps(payload)


# ── Patient-side scorer (the bot we are submitting) ──

def test_score_patient_adherence_parses_verdict():
    from scenarios.scenario_cards import get_scenario

    transcript = {
        "scenario_id": 2,
        "turns": [{"speaker": "patient", "text": "Saturday at 10?", "start_ms": 0}],
        "speaker_A_is_patient": True,
    }
    chat = _fake_patient_chat(
        {"in_character": True, "sprang_trap": True, "steered": True,
         "verdict": "GOOD", "evidence": "Saturday at 10?", "notes": "Pushed the Saturday trap."}
    )

    score = dry_run.score_patient_adherence(transcript, get_scenario(2), chat=chat)

    assert score["verdict"] == "GOOD"
    assert score["sprang_trap"] is True


def test_score_patient_adherence_handles_non_json():
    from scenarios.scenario_cards import get_scenario

    transcript = {"scenario_id": 1, "turns": [], "speaker_A_is_patient": True}
    chat = lambda system, messages, *, temperature, max_tokens: "sorry, not json"

    score = dry_run.score_patient_adherence(transcript, get_scenario(1), chat=chat)

    assert score["verdict"] == "ERROR"
    assert score["raw"] == "sorry, not json"


def test_evaluate_run_scores_both_patient_and_agent(monkeypatch, isolated_cwd, fake_env):
    from src import judge_bot

    # Agent judge says BUG (correct for buggy agent)...
    monkeypatch.setattr(
        judge_bot, "_call_judge_llm", lambda _p: '{"verdict": "BUG", "summary": "Booked Saturday."}'
    )
    # ...and the patient judge (injected) says the caller did its job.
    patient_chat = _fake_patient_chat(
        {"in_character": True, "sprang_trap": True, "steered": True,
         "verdict": "GOOD", "evidence": "even one Saturday a month?", "notes": "Sprang the trap."}
    )

    transcript = {
        "scenario_id": 2,
        "call_sid": "SIMbuggy1",
        "agent_mode": "buggy",
        "turns": [{"speaker": "patient", "text": "Saturday at 10?", "start_ms": 0}],
        "speaker_A_is_patient": True,
        "in_character_violations": [],
    }

    result = dry_run.evaluate_run(transcript, patient_chat=patient_chat)

    # patient side
    assert result["patient_verdict"] == "GOOD"
    assert result["patient_sprang_trap"] is True
    assert result["patient_ok"] is True
    # agent side
    assert result["verdict"] == "BUG"
    assert result["judge_ok"] is True
    assert not list((isolated_cwd / "runs").glob("scenario_*"))


def test_evaluate_run_fails_patient_when_out_of_character(monkeypatch, isolated_cwd, fake_env):
    from src import judge_bot

    monkeypatch.setattr(judge_bot, "_call_judge_llm", lambda _p: '{"verdict": "PASS"}')
    # Patient judge flags an out-of-character break.
    patient_chat = _fake_patient_chat(
        {"in_character": False, "sprang_trap": False, "steered": False,
         "verdict": "OOC", "evidence": "How can I help you today?", "notes": "Acted like staff."}
    )

    transcript = {
        "scenario_id": 1,
        "call_sid": "SIMgood1",
        "agent_mode": "good",
        # regex already caught a violation too
        "turns": [{"speaker": "patient", "text": "How can I help you today?", "start_ms": 0}],
        "speaker_A_is_patient": True,
        "in_character_violations": ["How can I help you today?"],
    }

    result = dry_run.evaluate_run(transcript, patient_chat=patient_chat)

    assert result["patient_verdict"] == "OOC"
    assert result["regex_in_character"] is False
    assert result["patient_ok"] is False


def test_evaluate_run_fails_patient_when_judge_marks_not_in_character(
    monkeypatch, isolated_cwd, fake_env
):
    from src import judge_bot

    monkeypatch.setattr(judge_bot, "_call_judge_llm", lambda _p: '{"verdict": "PASS"}')
    patient_chat = _fake_patient_chat(
        {
            "in_character": False,
            "sprang_trap": True,
            "steered": True,
            "verdict": "WEAK",
            "evidence": "How can I help you today?",
            "notes": "Contradictory judge payload should still fail.",
        }
    )

    transcript = {
        "scenario_id": 1,
        "call_sid": "SIMgood2",
        "agent_mode": "good",
        "turns": [{"speaker": "patient", "text": "How can I help you today?", "start_ms": 0}],
        "speaker_A_is_patient": True,
        "in_character_violations": [],
    }

    result = dry_run.evaluate_run(transcript, patient_chat=patient_chat)

    assert result["patient_verdict"] == "WEAK"
    assert result["patient_ok"] is False


def test_evaluate_run_marks_agent_judge_mismatch(monkeypatch, isolated_cwd, fake_env):
    from src import judge_bot

    # Buggy agent, but the judge wrongly says PASS -> agent-side mismatch must surface.
    monkeypatch.setattr(judge_bot, "_call_judge_llm", lambda _p: '{"verdict": "PASS"}')
    patient_chat = _fake_patient_chat({"verdict": "GOOD", "in_character": True, "sprang_trap": True})

    transcript = {
        "scenario_id": 4,
        "call_sid": "SIMbuggy2",
        "agent_mode": "buggy",
        "turns": [{"speaker": "patient", "text": "my back hurts and my leg is numb", "start_ms": 0}],
        "speaker_A_is_patient": True,
        "in_character_violations": [],
    }

    result = dry_run.evaluate_run(transcript, patient_chat=patient_chat)

    assert result["verdict"] == "PASS"
    assert result["expected"] == "BUG"
    assert result["judge_ok"] is False
    # patient still scored independently
    assert result["patient_ok"] is True
