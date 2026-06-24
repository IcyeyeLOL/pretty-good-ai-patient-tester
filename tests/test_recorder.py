from __future__ import annotations

import json
from pathlib import Path

from src.recorder import get_run_directory, update_call_meta


def test_get_run_directory_creates_expected_path(isolated_cwd):
    run_dir = Path(get_run_directory(1, "CA123"))

    assert run_dir == Path("runs") / "scenario_01_CA123"
    assert run_dir.exists()


def test_update_call_meta_creates_metadata_file(isolated_cwd):
    meta_path = Path(
        update_call_meta(
            1,
            "CA123",
            started_at="2026-06-23T00:00:00+00:00",
            recording_sid="RE123",
            extra={"provider": "test"},
        )
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["scenario_id"] == 1
    assert meta["scenario_name"] == "Basic Appointment Scheduling"
    assert meta["call_sid"] == "CA123"
    assert meta["started_at"] == "2026-06-23T00:00:00+00:00"
    assert meta["recording_sid"] == "RE123"
    assert meta["provider"] == "test"


def test_update_call_meta_merges_existing_file(isolated_cwd):
    update_call_meta(1, "CA123", recording_sid="RE123", extra={"first": True})
    meta_path = Path(
        update_call_meta(
            1,
            "CA123",
            ended_at="2026-06-23T00:01:00+00:00",
            duration_seconds=60,
            extra={"second": True},
        )
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["recording_sid"] == "RE123"
    assert meta["ended_at"] == "2026-06-23T00:01:00+00:00"
    assert meta["duration_seconds"] == 60
    assert meta["first"] is True
    assert meta["second"] is True
