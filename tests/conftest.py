from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


TEST_ENV = {
    "BASE_URL": "https://test.ngrok-free.app",
    "TARGET_PHONE": "+18054398008",
    "DEEPGRAM_API_KEY": "fake-deepgram-key",
    "ASSEMBLYAI_API_KEY": "fake-assemblyai-key",
    "CARTESIA_API_KEY": "fake-cartesia-key",
    "ANTHROPIC_API_KEY": "fake-anthropic-key",
    "LLM_PROVIDER": "anthropic",
    "TELEPHONY_PROVIDER": "twilio",
    "TWILIO_ACCOUNT_SID": "ACfake",
    "TWILIO_AUTH_TOKEN": "fake-twilio-token",
    "TWILIO_PHONE_NUMBER": "+15550000001",
    "TELNYX_API_KEY": "fake-telnyx-key",
    "TELNYX_CONNECTION_ID": "fake-connection-id",
    "TELNYX_PHONE_NUMBER": "+15550000002",
    "VERIFIED_TEST_PHONE": "+15550000003",
}


@pytest.fixture
def fake_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key, value in TEST_ENV.items():
        monkeypatch.setenv(key, value)
    yield


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in TEST_ENV:
        monkeypatch.delenv(key, raising=False)
    for key in list(os.environ):
        if key.startswith(("TWILIO_", "TELNYX_")):
            monkeypatch.delenv(key, raising=False)
    yield
