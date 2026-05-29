import pytest

from llm_orchestrator.providers.base import LLMProvider, LLMResult, LLMTransientError
from llm_orchestrator.llm_config import LLMSettings
from llm_orchestrator.llm_client import LLMClient, _trace_llm_inputs, _trace_llm_outputs


class FakeProvider(LLMProvider):
    def __init__(self, name: str, fail_times: int = 0):
        self.provider_name = name
        self._fail_times = fail_times
        self.calls = 0

    # PRODUCTION FIX: Added json_mode parameter to match the updated Phase 8 interface
    async def invoke(self, prompt: str, json_mode: bool = False) -> LLMResult:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise LLMTransientError("transient")
        return LLMResult(content='{"ok": true}', model="fake", provider=self.provider_name)


@pytest.mark.asyncio
async def test_fallback_triggers(monkeypatch):
    # 1. Inject dummy API key so OpenAIProvider initialization doesn't crash
    monkeypatch.setenv("LLM_OPENAI_API_KEY", "dummy-test-key")

    s = LLMSettings(primary_provider="openai", fallback_provider="ollama", max_retries=3)

    c = LLMClient(s)
    # 2. override providers with FakeProvider
    c._providers["openai"] = FakeProvider("openai", fail_times=3)
    c._providers["ollama"] = FakeProvider("ollama", fail_times=0)

    out = await c.invoke("hi")
    assert out.provider == "ollama"


@pytest.mark.asyncio
async def test_retry_then_success(monkeypatch):
    # 1. Inject dummy API key so OpenAIProvider initialization doesn't crash
    monkeypatch.setenv("LLM_OPENAI_API_KEY", "dummy-test-key")

    s = LLMSettings(primary_provider="openai", fallback_provider="ollama", max_retries=3)

    c = LLMClient(s)
    # 2. override providers with FakeProvider
    c._providers["openai"] = FakeProvider("openai", fail_times=2)
    c._providers["ollama"] = FakeProvider("ollama", fail_times=0)

    out = await c.invoke("hi")
    assert out.provider == "openai"


def test_langsmith_llm_trace_redacts_client_and_provider_objects():
    traced = _trace_llm_inputs(
        {
            "self": object(),
            "prompt": "Use DOC_1 to explain pump P-23 bearing wear.",
            "force_provider": "openai",
            "json_mode": True,
            "is_judge": False,
        }
    )

    assert traced == {
        "prompt": "Use DOC_1 to explain pump P-23 bearing wear.",
        "prompt_chars": 44,
        "force_provider": "openai",
        "json_mode": True,
        "is_judge": False,
    }
    assert "self" not in traced
    assert "api_key" not in str(traced).lower()


def test_langsmith_llm_trace_outputs_include_model_without_full_secret_surface():
    traced = _trace_llm_outputs(
        LLMResult(
            content="supported answer " * 200,
            model="gpt-4.1-mini",
            provider="openai",
        )
    )

    assert traced["provider"] == "openai"
    assert traced["model"] == "gpt-4.1-mini"
    assert traced["content_chars"] > len(traced["content_preview"])
    assert "api_key" not in str(traced).lower()
