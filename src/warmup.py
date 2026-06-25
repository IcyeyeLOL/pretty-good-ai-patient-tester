"""Cold-start warmup.

`build_pipeline` imports Pipecat, the LLM/STT/TTS service classes, and the
Telnyx serializer lazily — on the first call those imports happen *inside* the
websocket handler, adding seconds to `pipeline_build_duration_ms` for the first
call after the server starts. Preloading them at server startup moves that cost
off the call path so the first real call is as fast as subsequent ones.

`warm_runtime()` is idempotent and best-effort: a missing optional dependency is
recorded, not raised, so warmup never blocks the server from starting.
"""

from __future__ import annotations

_WARMED = False


def warm_runtime() -> dict:
    """Preload heavy imports used by the live pipeline. Returns a load report."""
    global _WARMED
    report: dict = {"already_warm": _WARMED, "loaded": [], "failed": {}}
    if _WARMED:
        report["ready"] = True
        return report

    def _try(name: str, fn) -> None:
        try:
            fn()
            report["loaded"].append(name)
        except Exception as exc:  # optional/uninstalled deps must not break startup
            report["failed"][name] = str(exc)

    def _pipecat_core():
        import pipecat.pipeline.pipeline  # noqa: F401
        import pipecat.pipeline.runner  # noqa: F401
        import pipecat.pipeline.task  # noqa: F401
        import pipecat.processors.aggregators.openai_llm_context  # noqa: F401
        import pipecat.transports.network.fastapi_websocket  # noqa: F401

    def _serializers():
        import pipecat.serializers.telnyx  # noqa: F401
        import pipecat.serializers.twilio  # noqa: F401

    _try("pipecat_core", _pipecat_core)
    _try("serializers", _serializers)
    _try("deepgram", lambda: __import__("pipecat.services.deepgram", fromlist=["x"]))
    _try("cartesia", lambda: __import__("pipecat.services.cartesia", fromlist=["x"]))
    _try("anthropic", lambda: __import__("pipecat.services.anthropic", fromlist=["x"]))
    _try("openai", lambda: __import__("pipecat.services.openai", fromlist=["x"]))
    _try("deepgram_sdk", lambda: __import__("deepgram", fromlist=["LiveOptions"]))

    def _silero_vad():
        # Construct once so the onnx model is downloaded/loaded off the call path.
        from pipecat.audio.vad.silero import SileroVADAnalyzer

        SileroVADAnalyzer(sample_rate=8000)

    _try("silero_vad", _silero_vad)
    _try("telnyx_instrumentation", lambda: __import__("src.telnyx_instrumentation", fromlist=["x"]))
    _try("pipeline", lambda: __import__("src.pipeline", fromlist=["build_pipeline"]))

    def _scenarios():
        from scenarios.prompts import get_prompt  # noqa: F401
        from scenarios.scenario_cards import get_all_scenarios

        # Touch the data so any lazy construction happens now.
        get_all_scenarios()

    _try("scenarios", _scenarios)

    # Consider the runtime warm if the core path loaded; optional providers (e.g.
    # openai when only anthropic is installed) are allowed to be absent.
    required = {"pipecat_core", "serializers", "deepgram", "cartesia", "pipeline", "scenarios"}
    _WARMED = required.issubset(set(report["loaded"]))
    report["ready"] = _WARMED
    return report
