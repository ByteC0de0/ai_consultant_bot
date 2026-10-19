from pathlib import Path

import pytest

from bot.config import Config
from bot.consultant import Consultant
from bot.knowledge import KnowledgeBase
from tests.helpers import FakeLLM, Harness

# Настоящая база знаний демо-кофейни: тесты проверяют поиск на живых данных
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


def make_config(**overrides) -> Config:
    values = dict(
        bot_token="123456:TEST",
        llm_api_key="test",
        knowledge_dir=KNOWLEDGE_DIR,
    )
    values.update(overrides)
    # _env_file=None: тесты не читают настоящий .env
    return Config(_env_file=None, **values)


@pytest.fixture
def config() -> Config:
    return make_config()


@pytest.fixture
def knowledge(config) -> KnowledgeBase:
    kb = KnowledgeBase(
        config.knowledge_dir,
        chunk_size=config.chunk_size,
        always_include=config.knowledge_always,
    )
    kb.refresh()
    return kb


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def consultant(config, knowledge, llm) -> Consultant:
    return Consultant(
        knowledge,
        llm,
        company=config.company_name,
        history_size=config.history_size,
        top_k=config.knowledge_top_k,
    )


@pytest.fixture
def harness(config, consultant) -> Harness:
    return Harness(config, consultant)
