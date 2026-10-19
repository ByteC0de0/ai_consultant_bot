"""Точка входа: читаем настройки и базу знаний, подключаем ИИ и запускаем бота."""

import asyncio
import contextlib
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError, TelegramNetworkError
from aiogram.types import BotCommand

from bot.config import Config, load_config
from bot.consultant import Consultant
from bot.handlers import router
from bot.knowledge import KnowledgeBase
from bot.llm import OpenAICompatibleLLM

logger = logging.getLogger("bot")

COMMANDS = [
    BotCommand(command="start", description="Что умеет бот"),
    BotCommand(command="reset", description="Начать разговор заново"),
]


async def set_commands(bot: Bot) -> None:
    """Меню команд (кнопка «/» в Telegram)."""
    try:
        await bot.set_my_commands(COMMANDS)
    except TelegramAPIError as e:
        # Без меню команд бот всё равно работает — не падаем из-за сбоя сети при старте
        logger.warning("Не удалось задать меню команд: %s", e)


async def wait_for_telegram(bot: Bot, first_delay: float = 5, max_delay: float = 60) -> None:
    """Ждёт, пока Telegram станет доступен. Без этого aiogram падает на старте,
    если в момент запуска пропала сеть. Ответ кэшируется в bot.me()."""
    delay = first_delay
    while True:
        try:
            me = await bot.me()
            logger.info("Связь с Telegram есть: @%s", me.username)
            return
        except TelegramNetworkError as e:
            logger.warning("Нет связи с Telegram (%s). Повтор через %s с", e, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)


def build_consultant(config: Config) -> tuple[Consultant, OpenAICompatibleLLM]:
    knowledge = KnowledgeBase(
        config.knowledge_dir,
        chunk_size=config.chunk_size,
        always_include=config.knowledge_always,
    )
    knowledge.refresh()
    if not knowledge.chunks:
        raise RuntimeError(f"База знаний пуста: положите .md или .txt в {config.knowledge_dir}")
    llm = OpenAICompatibleLLM(config)
    consultant = Consultant(
        knowledge,
        llm,
        company=config.company_name,
        history_size=config.history_size,
        top_k=config.knowledge_top_k,
    )
    return consultant, llm


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = load_config()
    consultant, llm = build_consultant(config)
    logger.info("ИИ: %s через %s", config.llm_model, config.llm_base_url)

    bot = Bot(
        token=config.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    # Всё, что передано в Dispatcher(...), aiogram подставит в обработчики по имени
    # аргумента: def handler(message, consultant: Consultant, config: Config)
    dp = Dispatcher(config=config, consultant=consultant)
    dp.include_router(router)
    dp.startup.register(set_commands)

    try:
        await wait_for_telegram(bot)
        await dp.start_polling(bot)
    finally:
        await llm.close()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
