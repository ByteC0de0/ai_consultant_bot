import pytest
from pydantic import ValidationError

from bot.config import BASE_DIR, Config
from tests.conftest import make_config


def test_defaults():
    config = Config(_env_file=None, bot_token="1:abc", llm_api_key="key")
    assert config.llm_base_url == "https://api.openai.com/v1"
    assert config.knowledge_dir == BASE_DIR / "knowledge"
    assert config.knowledge_always == ("contacts.md",)


def test_provider_switch_only_through_settings():
    config = make_config(
        llm_base_url="http://localhost:11434/v1/", llm_model="qwen2.5:14b", llm_api_key="ollama"
    )
    assert config.llm_base_url == "http://localhost:11434/v1"
    assert config.llm_model == "qwen2.5:14b"


def test_lists_parsed_from_env(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "1:abc")
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("KNOWLEDGE_ALWAYS", "contacts.md, rules.txt")
    monkeypatch.setenv("EXAMPLE_QUESTIONS", "Цены? | Адрес?")
    config = Config(_env_file=None)
    assert config.knowledge_always == ("contacts.md", "rules.txt")
    assert config.example_questions == ("Цены?", "Адрес?")


@pytest.mark.parametrize(
    "overrides",
    [
        {"bot_token": "не токен"},
        {"llm_api_key": ""},
        {"llm_base_url": "localhost:11434"},
        {"llm_temperature": 3},
        {"knowledge_top_k": -1},
        {"example_questions": "x" * 65},
    ],
)
def test_invalid_values_rejected(overrides):
    with pytest.raises(ValidationError):
        make_config(**overrides)
