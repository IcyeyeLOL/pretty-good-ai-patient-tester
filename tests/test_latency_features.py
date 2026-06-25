import json

import pytest

from run import summarize_turn_latency
from scenarios.scenario_cards import get_scenario
from src.pipeline import name_fastpath_response
from src.startup_timeline import StartupTimeline, compute_startup_deltas
from src.warmup import warm_runtime


# ── new startup deltas ────────────────────────────────────────────────────────

def test_call_answered_to_first_bot_audio_uses_serialized_media():
    marks = {
        "call_initiated": 1000,
        "call_answered": 3000,
        "opener_scheduled": 3200,
        "first_serialized_media": 3600,
        "first_outbound_audio_before_transport": 3550,
    }
    deltas = compute_startup_deltas(marks)
    assert deltas["call_answered_to_first_bot_audio_ms"] == 600
    assert deltas["call_answered_to_opener_scheduled_ms"] == 200
    assert deltas["opener_scheduled_to_first_serialized_media_ms"] == 400
    # initiated total is now explicitly labeled as including ring time.
    assert deltas["total_initiated_to_first_bot_audio_ms_incl_ring"] == 2600


def test_pipeline_build_duration_delta():
    marks = {
        "ws_start_event": 100,
        "pipeline_build_started": 150,
        "pipeline_built": 950,
    }
    deltas = compute_startup_deltas(marks)
    assert deltas["ws_start_to_pipeline_build_started_ms"] == 50
    assert deltas["pipeline_build_duration_ms"] == 800


def test_pipeline_build_marks_merge_into_timeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Simulate the server marking build start and the pipeline marking build done.
    StartupTimeline(2, "sid").mark("ws_start_event", 100)
    StartupTimeline(2, "sid").mark("pipeline_build_started", 150)
    StartupTimeline(2, "sid").mark("pipeline_built", 950)

    data = json.loads((tmp_path / "runs" / "scenario_02_sid" / "startup_timeline.json").read_text())
    assert data["marks"]["pipeline_build_started"] == 150
    assert data["marks"]["pipeline_built"] == 950
    assert data["deltas"]["pipeline_build_duration_ms"] == 800


# ── warmup ────────────────────────────────────────────────────────────────────

def test_warm_runtime_ready_and_idempotent():
    first = warm_runtime()
    assert first["ready"] is True
    second = warm_runtime()
    assert second["already_warm"] is True
    assert second["ready"] is True


# ── response-latency summary ──────────────────────────────────────────────────

def test_summarize_turn_latency_median_max_and_slow_flag():
    turns = [
        {"stt_committed": 0, "first_llm_token": 300, "stt_to_first_audio_ms": 900},
        {"stt_committed": 0, "first_llm_token": 500, "stt_to_first_audio_ms": 2000},
        {"stt_committed": 0, "first_llm_token": 400, "stt_to_first_audio_ms": 1100},
    ]
    s = summarize_turn_latency(turns)
    assert s["turns_measured"] == 3
    assert s["median_stt_to_first_audio_ms"] == 1100
    assert s["max_stt_to_first_audio_ms"] == 2000
    assert s["slow_turns"] == [{"index": 1, "ms": 2000}]
    assert s["median_stt_to_first_token_ms"] == 400
    assert s["max_stt_to_first_token_ms"] == 500


def test_summarize_turn_latency_handles_missing_fields():
    s = summarize_turn_latency([{"stt_committed": 0}, {}])
    assert s["turns_measured"] == 0
    assert "median_stt_to_first_audio_ms" not in s


# ── name fast-path ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "question",
    [
        "Can I get your name?",
        "What's your name?",
        "Who am I speaking with?",
        "May I ask who's calling?",
        "Can I have your full name please?",
    ],
)
def test_name_fastpath_triggers_for_pure_name_request(question):
    line = name_fastpath_response(question, "Maria Johnson")
    assert line is not None
    assert "Maria Johnson" in line


@pytest.mark.parametrize(
    "question",
    [
        "Can I get your name and date of birth?",
        "What's your name and the reason for your visit?",
        "What is your insurance provider?",
        "What day works for your appointment?",
        "Are you having any chest pain?",
        "Can I get your phone number?",
        "Can you come in this Saturday?",
    ],
)
def test_name_fastpath_does_not_trigger_for_substantive_questions(question):
    assert name_fastpath_response(question, "David Park") is None


def test_name_fastpath_uses_scenario_card_name():
    scenario = get_scenario(2)  # David Park
    line = name_fastpath_response("What's your name?", scenario.caller_name)
    assert scenario.caller_name in line


def test_name_fastpath_empty_inputs():
    assert name_fastpath_response("", "Maria") is None
    assert name_fastpath_response("What's your name?", "") is None
