"""Сценарии из Telegram целиком: апдейт → обработчик → консультант → ответ."""

from bot.handlers import ERROR_TEXT, split_text
from tests.conftest import make_config
from tests.helpers import CLIENT, OTHER, Harness, async_test


@async_test
async def test_start_shows_greeting_and_examples(harness: Harness):
    await harness.send(CLIENT, "/start")
    text = harness.last(CLIENT.id).text
    assert "Кофейня «Зёрна»" in text
    assert "Сколько стоит доставка?" in text
    assert "/reset" in text
    buttons = [b.text for b in harness.buttons(CLIENT.id)]
    assert buttons == list(harness.config.example_questions)


@async_test
async def test_start_without_examples(consultant):
    harness = Harness(make_config(example_questions=""), consultant)
    await harness.send(CLIENT, "/start")
    assert "Например" not in harness.last(CLIENT.id).text
    assert harness.last(CLIENT.id).reply_markup is None


@async_test
async def test_question_gets_answer_as_plain_text(harness: Harness, llm):
    llm.reply = "Доставка стоит 150 ₽ <бесплатно от 1500 ₽>"
    await harness.send(CLIENT, "Сколько стоит доставка?")
    last = harness.last(CLIENT.id)
    assert last.text == "Доставка стоит 150 ₽ <бесплатно от 1500 ₽>"
    assert last.parse_mode is None  # «<» в ответе не ломает HTML-разметку
    assert "Стоимость доставки — 150 ₽" in llm.last_system


@async_test
async def test_dialog_is_remembered_and_reset(harness: Harness, llm):
    await harness.send(CLIENT, "Сколько стоит латте?")
    await harness.send(CLIENT, "А на овсяном?")
    sent = llm.requests[-1]
    assert [m["content"] for m in sent[1:]] == [
        "Сколько стоит латте?",
        "Ответ консультанта",
        "А на овсяном?",
    ]

    await harness.send(CLIENT, "/reset")
    assert "очищена" in harness.last(CLIENT.id).text
    await harness.send(CLIENT, "Привет")
    assert len(llm.requests[-1]) == 2  # только system и новый вопрос


@async_test
async def test_history_is_separate_per_chat(harness: Harness, llm):
    await harness.send(CLIENT, "Вопрос Ивана")
    await harness.send(OTHER, "Вопрос Петра")
    assert [m["content"] for m in llm.requests[-1][1:]] == ["Вопрос Петра"]


@async_test
async def test_llm_failure_gives_friendly_error(harness: Harness, llm):
    llm.fail = True
    await harness.send(CLIENT, "Сколько стоит доставка?")
    assert harness.last(CLIENT.id).text == ERROR_TEXT


@async_test
async def test_too_long_question_not_sent_to_llm(harness: Harness, llm):
    await harness.send(CLIENT, "а" * (harness.config.max_question_length + 1))
    assert "слишком длинный" in harness.last(CLIENT.id).text
    assert llm.requests == []


@async_test
async def test_long_answer_split_into_messages(harness: Harness, llm):
    llm.reply = ("Строка ответа\n" * 400).strip()  # ~5600 символов
    await harness.send(CLIENT, "Расскажите всё")
    parts = harness.texts(CLIENT.id)
    assert len(parts) == 2
    assert all(len(p) <= 4096 for p in parts)
    assert "\n".join(parts) == llm.reply


def test_split_text_without_line_breaks():
    parts = split_text("x" * 5000, limit=4096)
    assert [len(p) for p in parts] == [4096, 904]


@async_test
async def test_example_button_asks_question(harness: Harness, llm):
    await harness.send(CLIENT, "/start")
    button = harness.buttons(CLIENT.id)[2]
    await harness.click(CLIENT, button.callback_data)
    assert harness.texts(CLIENT.id)[-2] == f"❓ {button.text}"
    assert llm.requests[-1][-1]["content"] == button.text
    assert harness.last(CLIENT.id).text == "Ответ консультанта"


@async_test
async def test_stale_example_button_ignored(harness: Harness, llm):
    await harness.click(CLIENT, "example:99")
    await harness.click(CLIENT, "example:abc")
    assert llm.requests == []
    assert harness.messages(CLIENT.id) == []


@async_test
async def test_unknown_command_and_sticker(harness: Harness, llm):
    await harness.send(CLIENT, "/menu")
    assert "Такой команды нет" in harness.last(CLIENT.id).text
    await harness.send_sticker(CLIENT)
    assert "только текст" in harness.last(CLIENT.id).text
    assert llm.requests == []
