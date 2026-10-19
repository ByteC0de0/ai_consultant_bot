import pytest

from bot.llm import OpenAICompatibleLLM, is_local
from tests.conftest import make_config


@pytest.mark.parametrize(
    "url, local",
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:11434/v1", True),
        ("http://[::1]:11434/v1", True),
        ("https://api.openai.com/v1", False),
        ("https://openrouter.ai/api/v1", False),
    ],
)
def test_is_local(url, local):
    assert is_local(url) is local


def test_local_model_bypasses_system_proxy(monkeypatch):
    # При включённом VPN системный прокси перехватывал запросы к Ollama и отвечал 503
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:10809")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    llm = OpenAICompatibleLLM(make_config(llm_base_url="http://localhost:11434/v1"))
    assert llm.client._client.trust_env is False


def test_cloud_provider_keeps_system_proxy():
    llm = OpenAICompatibleLLM(make_config(llm_base_url="https://api.openai.com/v1"))
    assert llm.client._client.trust_env is True
