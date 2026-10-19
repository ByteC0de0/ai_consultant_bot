"""Задать вопрос консультанту из консоли, без Telegram. Удобно, чтобы проверить
базу знаний и модель: настройки ИИ берутся из того же .env, токен бота не нужен.

    python ask.py "Сколько стоит доставка?"          # один или несколько вопросов
    python ask.py --context "Где вы находитесь?"     # плюс какие фрагменты нашёл поиск
    python ask.py                                    # диалог: вопросы по одному, пустая строка — выход
"""

import argparse
import asyncio
import sys
import time

from bot.config import Config
from main import build_consultant

CHAT_ID = 0


async def run(questions: list[str], show_context: bool) -> None:
    # Токен Telegram здесь не нужен, подставляем заглушку
    config = Config(bot_token="0:console")
    consultant, llm = build_consultant(config)
    print(f"Модель: {config.llm_model} ({config.llm_base_url})\n")
    interactive = not questions
    try:
        while True:
            if interactive:
                question = input("Вы: ").strip()
                if not question:
                    break
            elif questions:
                question = questions.pop(0)
                print(f"Вопрос: {question}")
            else:
                break
            if show_context:
                found = consultant.find_context(CHAT_ID, question)
                titles = "; ".join(f"{c.source}: {c.title}" for c in found)
                print(f"[Фрагменты: {titles}]")
            started = time.perf_counter()
            answer = await consultant.answer(CHAT_ID, question)
            print(f"Бот ({time.perf_counter() - started:.1f} с): {answer}\n")
    finally:
        await llm.close()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", nargs="*", help="вопросы; без них — интерактивный режим")
    parser.add_argument("--context", action="store_true", help="показать найденные фрагменты")
    args = parser.parse_args()
    asyncio.run(run(list(args.questions), args.context))


if __name__ == "__main__":
    main()
