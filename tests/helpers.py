"""Фейковый Telegram и фейковый ИИ для тестов: апдейты подаются в диспетчер напрямую,
запросы бота к API записываются в список, а «модель» возвращает заготовленный ответ."""

import asyncio
import functools
import itertools
from datetime import datetime

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.base import BaseSession
from aiogram.enums import ParseMode
from aiogram.methods import SendMessage
from aiogram.types import (
    CallbackQuery,
    Chat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Sticker,
    Update,
    User,
)

from bot.config import Config
from bot.consultant import Consultant
from bot.handlers import router
from bot.llm import LLMError

CLIENT = User(id=1, is_bot=False, first_name="Иван")
OTHER = User(id=2, is_bot=False, first_name="Пётр")

# Роутер можно подключить только к одному диспетчеру, поэтому он общий на все тесты,
# а config и consultant передаются в каждый апдейт отдельно.
DP = Dispatcher()
DP.include_router(router)


def async_test(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


class FakeLLM:
    """Запоминает, что ему отправили, и отвечает заготовкой (или падает)."""

    def __init__(self, reply: str = "Ответ консультанта") -> None:
        self.reply = reply
        self.fail = False
        self.requests: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        if self.fail:
            raise LLMError("нет связи")
        return self.reply

    @property
    def last_system(self) -> str:
        return self.requests[-1][0]["content"]


class FakeSession(BaseSession):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list = []
        self._message_ids = itertools.count(100)

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        if isinstance(method, SendMessage):
            return Message(
                message_id=next(self._message_ids),
                date=datetime.now(),
                chat=Chat(id=method.chat_id or 0, type="private"),
                text=method.text,
            )
        return True

    async def close(self) -> None:
        pass

    async def stream_content(self, *args, **kwargs):
        yield b""


class Harness:
    def __init__(self, config: Config, consultant: Consultant) -> None:
        self.config = config
        self.consultant = consultant
        self.session = FakeSession()
        self.bot = Bot(
            "123456:TEST",
            session=self.session,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        self._ids = itertools.count(1)

    async def _feed(self, **kwargs) -> None:
        update = Update(update_id=next(self._ids), **kwargs)
        await DP.feed_update(
            self.bot, update, config=self.config, consultant=self.consultant
        )

    def _message(self, user: User, **kwargs) -> Message:
        return Message(
            message_id=next(self._ids),
            date=datetime.now(),
            chat=Chat(id=user.id, type="private"),
            from_user=user,
            **kwargs,
        )

    async def send(self, user: User, text: str) -> None:
        await self._feed(message=self._message(user, text=text))

    async def send_sticker(self, user: User) -> None:
        sticker = Sticker(
            file_id="s", file_unique_id="s", type="regular",
            width=512, height=512, is_animated=False, is_video=False,
        )
        await self._feed(message=self._message(user, sticker=sticker))

    async def click(self, user: User, data: str) -> None:
        callback = CallbackQuery(
            id=str(next(self._ids)),
            from_user=user,
            chat_instance="test",
            data=data,
            message=self._message(user, text="..."),
        )
        await self._feed(callback_query=callback)

    def messages(self, chat_id: int) -> list[SendMessage]:
        return [
            call
            for call in self.session.calls
            if isinstance(call, SendMessage) and call.chat_id == chat_id
        ]

    def texts(self, chat_id: int) -> list[str]:
        return [m.text for m in self.messages(chat_id)]

    def last(self, chat_id: int) -> SendMessage:
        return self.messages(chat_id)[-1]

    def buttons(self, chat_id: int) -> list[InlineKeyboardButton]:
        for call in reversed(self.messages(chat_id)):
            if isinstance(call.reply_markup, InlineKeyboardMarkup):
                return [b for row in call.reply_markup.inline_keyboard for b in row]
        return []
