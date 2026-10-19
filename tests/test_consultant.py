import pytest

from bot.consultant import EMPTY_CONTEXT, Consultant, build_messages, clean_reply
from bot.history import ChatHistory
from bot.knowledge import Chunk
from bot.llm import LLMError
from tests.helpers import FakeLLM, async_test


def test_build_messages_order():
    chunk = Chunk(source="menu.md", title="Меню / Кофе", text="Эспрессо — 150 ₽")
    history = [
        {"role": "user", "content": "Привет"},
        {"role": "assistant", "content": "Здравствуйте!"},
    ]
    messages = build_messages("Кофейня", [chunk], history, "Сколько стоит эспрессо?")
    assert messages[0]["role"] == "system"
    assert "[menu.md — Меню / Кофе]\nЭспрессо — 150 ₽" in messages[0]["content"]
    assert "Кофейня" in messages[0]["content"]
    assert messages[1:3] == history
    assert messages[-1] == {"role": "user", "content": "Сколько стоит эспрессо?"}


def test_build_messages_without_context():
    messages = build_messages("Кофейня", [], [], "Вопрос")
    assert EMPTY_CONTEXT in messages[0]["content"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("**Капучино** — 220 ₽", "Капучино — 220 ₽"),
        ("<think>считаю цены</think>\nЛатте — 230 ₽", "Латте — 230 ₽"),
        ("### Доставка\n* 150 ₽\n- бесплатно от 1500 ₽", "Доставка\n• 150 ₽\n• бесплатно от 1500 ₽"),
        ("Используйте `код`", "Используйте код"),
    ],
)
def test_clean_reply(raw, expected):
    assert clean_reply(raw) == expected


def test_history_keeps_last_messages_only():
    history = ChatHistory(size=4)
    for i in range(3):
        history.add(1, f"вопрос {i}", f"ответ {i}")
    assert [m["content"] for m in history.get(1)] == ["вопрос 1", "ответ 1", "вопрос 2", "ответ 2"]
    assert history.last_question(1) == "вопрос 2"
    assert history.get(2) == []


def test_history_disabled():
    history = ChatHistory(size=0)
    history.add(1, "вопрос", "ответ")
    assert history.get(1) == []


@async_test
async def test_answer_uses_relevant_knowledge(consultant: Consultant, llm: FakeLLM):
    llm.reply = "**Доставка** стоит 150 ₽"
    answer = await consultant.answer(1, "Сколько стоит доставка?")
    assert answer == "Доставка стоит 150 ₽"
    system = llm.last_system
    assert "Стоимость доставки — 150 ₽" in system
    assert "@zerna_manager" in system  # контакт менеджера есть всегда
    assert "Чизкейк" not in system  # меню десертов к доставке не относится


@async_test
async def test_follow_up_question_searches_with_previous(consultant: Consultant, llm: FakeLLM):
    await consultant.answer(1, "Где находится кофейня на Чистопольской?")
    await consultant.answer(1, "А парковка там есть?")
    # Во втором вопросе нет слова «Чистопольская», но нужный раздел всё равно найден
    assert "бесплатная парковка у входа" in llm.last_system
    assert llm.requests[-1][1]["content"] == "Где находится кофейня на Чистопольской?"


@async_test
async def test_topic_change_not_spoiled_by_previous_question(consultant: Consultant, llm: FakeLLM):
    await consultant.answer(1, "Сколько стоит капучино на овсяном молоке?")
    await consultant.answer(1, "До скольки вы работаете в субботу?")
    # Обе кофейни должны попасть в контекст, а не только меню из прошлого вопроса
    assert "ул. Баумана" in llm.last_system
    assert "ул. Чистопольская" in llm.last_system


@async_test
async def test_failed_answer_not_saved_to_history(consultant: Consultant, llm: FakeLLM):
    llm.fail = True
    with pytest.raises(LLMError):
        await consultant.answer(1, "Вопрос")
    assert consultant.history.get(1) == []


@async_test
async def test_only_thinking_in_reply_is_error(consultant: Consultant, llm: FakeLLM):
    llm.reply = "<think>хм</think>"
    with pytest.raises(LLMError):
        await consultant.answer(1, "Вопрос")
