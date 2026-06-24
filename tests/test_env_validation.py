from __future__ import annotations

import pytest
from click import ClickException

from run import validate_env


def test_correct_full_twilio_config_passes(fake_env):
    validate_env(require_call_keys=True)


def test_server_mode_requires_only_safe_basics(fake_env, monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)

    validate_env(require_call_keys=False)


def test_target_phone_lock_is_enforced(fake_env, monkeypatch):
    monkeypatch.setenv("TARGET_PHONE", "+15550000000")

    with pytest.raises(ClickException):
        validate_env(require_call_keys=True)


def test_base_url_must_not_end_with_slash(fake_env, monkeypatch):
    monkeypatch.setenv("BASE_URL", "https://test.ngrok-free.app/")

    with pytest.raises(ClickException):
        validate_env(require_call_keys=True)


@pytest.mark.parametrize(
    "missing_key",
    ["DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY", "CARTESIA_API_KEY"],
)
def test_common_call_keys_are_required(fake_env, monkeypatch, missing_key: str):
    monkeypatch.delenv(missing_key, raising=False)

    with pytest.raises(ClickException):
        validate_env(require_call_keys=True)


def test_anthropic_key_required_for_anthropic_provider(fake_env, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ClickException):
        validate_env(require_call_keys=True)


def test_openai_key_required_for_openai_provider(fake_env, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ClickException):
        validate_env(require_call_keys=True)


def test_unsupported_llm_provider_fails(fake_env, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    with pytest.raises(ClickException):
        validate_env(require_call_keys=True)


def test_twilio_keys_required_for_twilio_provider(fake_env, monkeypatch):
    monkeypatch.setenv("TELEPHONY_PROVIDER", "twilio")
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)

    with pytest.raises(ClickException):
        validate_env(require_call_keys=True)


def test_telnyx_keys_required_for_telnyx_provider(fake_env, monkeypatch):
    monkeypatch.setenv("TELEPHONY_PROVIDER", "telnyx")
    monkeypatch.delenv("TELNYX_API_KEY", raising=False)

    with pytest.raises(ClickException):
        validate_env(require_call_keys=True)


def test_use_test_target_requires_verified_test_phone(fake_env, monkeypatch):
    monkeypatch.delenv("VERIFIED_TEST_PHONE", raising=False)

    with pytest.raises(ClickException):
        validate_env(require_call_keys=True, use_test_target=True)
