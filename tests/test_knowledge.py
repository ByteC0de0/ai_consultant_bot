import os

import pytest

from bot.knowledge import KnowledgeBase, split_document, stem, tokenize


def titles(chunks) -> list[str]:
    return [c.title.split(" / ")[-1] for c in chunks]


def test_stem_word_forms_match():
    assert stem("доставка") == stem("доставки") == stem("доставку")
    assert stem("работаете") == stem("работы")
    assert stem("находитесь") == stem("находимся")


def test_tokenize_drops_stop_words_and_normalizes():
    assert tokenize("Где вы? Ёлка!") == [stem("елка")]


def test_split_document_by_headings_keeps_path():
    text = "# Меню\nВступление\n## Кофе\nЭспрессо — 150 ₽\n### Сиропы\nВаниль\n## Чай\nЧёрный"
    chunks = split_document("menu.md", text, chunk_size=1000)
    assert [c.title for c in chunks] == [
        "Меню",
        "Меню / Кофе",
        "Меню / Кофе / Сиропы",
        "Меню / Чай",
    ]
    assert chunks[1].text == "Эспрессо — 150 ₽"
    assert all(c.source == "menu.md" for c in chunks)


def test_long_section_is_split_without_losing_text():
    paragraphs = [f"Абзац {i}: " + "слово " * 30 for i in range(10)]
    text = "# Раздел\n" + "\n\n".join(paragraphs)
    chunks = split_document("big.txt", text, chunk_size=400)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    for i in range(10):
        assert f"Абзац {i}:" in joined


def test_plain_text_without_headings():
    chunks = split_document("notes.txt", "Просто текст\nбез заголовков", chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0].title == ""
    assert chunks[0].render().startswith("[notes.txt]")


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Сколько стоит доставка?", "Доставка"),
        ("До скольки вы работаете в субботу?", "Кофейня на Баумана"),
        ("Сколько стоит капучино на овсяном молоке?", "Добавки к напиткам"),
        ("Можно прийти с собакой?", "Можно ли прийти с собакой?"),
        ("Где вы находитесь?", "Кофейня на Баумана"),
        ("Вы открыты 1 января?", "Праздничные дни"),
    ],
)
def test_search_finds_relevant_section(knowledge, question, expected):
    assert expected in titles(knowledge.search(question, top_k=5))


def test_search_returns_nothing_for_unrelated_question(knowledge):
    assert knowledge.search("Какая погода в Москве?", top_k=5) == []


def test_context_always_contains_manager_contacts(knowledge):
    # Даже если поиск ничего не нашёл, модель должна знать, куда отправить клиента
    chunks = knowledge.context_for("Какая погода в Москве?", top_k=5)
    assert [c.source for c in chunks] == ["contacts.md"]
    assert "@zerna_manager" in chunks[0].text


def test_context_limited_to_top_k(knowledge):
    chunks = knowledge.context_for("Сколько стоит кофе, доставка, чай и завтрак?", top_k=2)
    assert len(chunks) == 3  # 2 найденных + контакты
    assert chunks[-1].source == "contacts.md"
    assert len(set(chunks)) == len(chunks)


def test_top_k_zero_gives_whole_base(knowledge):
    assert knowledge.context_for("что угодно", top_k=0) == knowledge.chunks


def test_reloads_when_files_replaced(tmp_path):
    (tmp_path / "info.md").write_text("# График\nРаботаем с 9 до 18", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"not text")  # не .md/.txt — пропускаем
    kb = KnowledgeBase(tmp_path)
    assert kb.refresh() is True
    assert kb.refresh() is False  # ничего не менялось — не перечитываем
    assert "с 9 до 18" in kb.context_for("график", top_k=3)[0].text

    path = tmp_path / "info.md"
    path.write_text("# График\nТеперь работаем круглосуточно", encoding="utf-8")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    (tmp_path / "prices.txt").write_text("Стрижка — 1000 рублей", encoding="utf-8")

    assert "круглосуточно" in kb.context_for("график", top_k=3)[0].text
    assert "1000" in kb.context_for("стрижка", top_k=3)[0].text

    path.unlink()
    assert kb.context_for("график", top_k=3) == []


def test_missing_directory_gives_empty_base(tmp_path):
    kb = KnowledgeBase(tmp_path / "nope")
    kb.refresh()
    assert kb.chunks == []
    assert kb.search("что угодно", top_k=3) == []


def test_outline_lists_every_section_once(knowledge):
    outline = knowledge.outline()
    assert "Меню и цены / Завтраки (каждый день до 12:00)" in outline
    assert "Контакты" in outline
    assert len(outline) == len(set(outline))


@pytest.mark.parametrize("question", ["Какой ассортимент в кофейне?", "Что у вас есть из еды?"])
def test_broad_question_finds_menu_overview(knowledge, question):
    assert "Меню и цены" in [c.title for c in knowledge.search(question, top_k=5)]
