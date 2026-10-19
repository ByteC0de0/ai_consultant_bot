from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import GetMe
from aiogram.types import User

import main
from main import build_consultant, wait_for_telegram
from tests.conftest import make_config
from tests.helpers import async_test


@async_test
async def test_wait_for_telegram_retries_until_online(monkeypatch):
    monkeypatch.setattr(main.asyncio, "sleep", AsyncMock())
    bot = AsyncMock()
    me = User(id=1, is_bot=True, first_name="Бот", username="demo_bot")
    bot.me.side_effect = [
        TelegramNetworkError(GetMe(), "нет сети"),
        TelegramNetworkError(GetMe(), "нет сети"),
        me,
    ]
    await wait_for_telegram(bot, first_delay=1, max_delay=60)
    assert bot.me.await_count == 3
    assert [c.args[0] for c in main.asyncio.sleep.await_args_list] == [1, 2]


def test_build_consultant_from_settings():
    consultant, _ = build_consultant(make_config(knowledge_top_k=3, history_size=4))
    assert consultant.top_k == 3
    assert consultant.history.size == 4
    assert consultant.knowledge.chunks


def test_empty_knowledge_base_stops_start(tmp_path):
    with pytest.raises(RuntimeError, match="База знаний пуста"):
        build_consultant(make_config(knowledge_dir=tmp_path))
