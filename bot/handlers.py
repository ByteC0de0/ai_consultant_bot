"""Обработчики Telegram: /start, /reset, вопросы текстом и кнопки-примеры."""

import logging
from html import escape

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from aiogram.utils.chat_action import ChatActionSender

from bot.config import Config
from bot.consultant import Consultant
from bot.llm import LLMError

logger = logging.getLogger(__name__)

router = Router()

TELEGRAM_LIMIT = 4096

EXAMPLE_PREFIX = "example:"

ERROR_TEXT = (
    "Извините, не получилось подготовить ответ — сервис временно недоступен. "
    "Попробуйте ещё раз через минуту."
)


def greeting(config: Config) -> str:
    text = (
        f"Здравствуйте! Я онлайн-консультант <b>{escape(config.company_name)}</b>.\n\n"
        "Отвечу на вопросы об услугах и ценах, адресах и графике работы, доставке "
        "и всём остальном. Просто напишите вопрос своими словами."
    )
    if config.example_questions:
        examples = "\n".join(f"• {escape(q)}" for q in config.example_questions)
        text += f"\n\nНапример:\n{examples}\n\nНажмите на пример ниже, чтобы попробовать."
    return text + "\n\n/reset — начать разговор заново"


def examples_keyboard(config: Config) -> InlineKeyboardMarkup | None:
    if not config.example_questions:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=q, callback_data=f"{EXAMPLE_PREFIX}{i}")]
            for i, q in enumerate(config.example_questions)
        ]
    )


def split_text(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Длинный ответ — несколькими сообщениями, по возможности по границе абзаца."""
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


async def reply_to_question(
    bot: Bot, chat_id: int, question: str, consultant: Consultant, config: Config
) -> None:
    question = question.strip()
    if len(question) > config.max_question_length:
        await bot.send_message(
            chat_id,
            f"Вопрос слишком длинный. Сформулируйте его короче — до "
            f"{config.max_question_length} символов.",
        )
        return
    try:
        async with ChatActionSender.typing(bot=bot, chat_id=chat_id):
            answer = await consultant.answer(chat_id, question)
    except LLMError:
        await bot.send_message(chat_id, ERROR_TEXT)
        return
    # parse_mode=None: ответ ИИ отправляем как обычный текст, чтобы случайные
    # «<» или «&» в нём не сломали HTML-разметку Telegram
    for part in split_text(answer):
        await bot.send_message(chat_id, part, parse_mode=None)


@router.message(CommandStart())
@router.message(Command("help"))
async def cmd_start(message: Message, config: Config) -> None:
    await message.answer(greeting(config), reply_markup=examples_keyboard(config))


@router.message(Command("reset"))
async def cmd_reset(message: Message, consultant: Consultant) -> None:
    consultant.reset(message.chat.id)
    await message.answer("История диалога очищена. Задайте новый вопрос.")


@router.message(F.text.startswith("/"))
async def unknown_command(message: Message) -> None:
    await message.answer("Такой команды нет. /start — что я умею, /reset — начать заново.")


@router.message(F.text)
async def question(message: Message, bot: Bot, consultant: Consultant, config: Config) -> None:
    await reply_to_question(bot, message.chat.id, message.text, consultant, config)


@router.message()
async def not_text(message: Message) -> None:
    await message.answer("Я понимаю только текст. Напишите вопрос сообщением.")


@router.callback_query(F.data.startswith(EXAMPLE_PREFIX))
async def example_clicked(
    callback: CallbackQuery, bot: Bot, consultant: Consultant, config: Config
) -> None:
    await callback.answer()
    index = callback.data.removeprefix(EXAMPLE_PREFIX)
    # Кнопка могла остаться от старого приветствия, а примеры в .env с тех пор поменяли
    if (
        callback.message is None
        or not index.isdigit()
        or int(index) >= len(config.example_questions)
    ):
        return
    text = config.example_questions[int(index)]
    chat_id = callback.message.chat.id
    await bot.send_message(chat_id, f"❓ {escape(text)}")
    await reply_to_question(bot, chat_id, text, consultant, config)
