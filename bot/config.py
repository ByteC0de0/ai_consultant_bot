"""Настройки бота из .env. Все проверки значений — здесь, в одной Pydantic-модели.

Провайдер ИИ (OpenAI, OpenRouter, ProxyAPI, DeepSeek, Ollama…) переключается только
через LLM_BASE_URL / LLM_API_KEY / LLM_MODEL — код при этом не меняется."""

from pathlib import Path
from typing import Annotated

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# NoDecode: строку "a.md,b.md" из .env разбираем сами, а не как JSON
StrList = Annotated[tuple[str, ...], NoDecode]


def _split(value: object, separator: str) -> object:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(separator) if part.strip())
    return value


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    bot_token: str = Field(pattern=r"^\d+:[\w-]+$")
    company_name: str = Field("Кофейня «Зёрна»", min_length=1, max_length=64)

    # --- ИИ: любой OpenAI-совместимый API ---
    llm_api_key: str = Field(min_length=1)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = Field("gpt-4o-mini", min_length=1)
    llm_temperature: float = Field(0.2, ge=0, le=2)
    llm_timeout: float = Field(60, gt=0, le=600)  # секунд на один ответ
    llm_max_tokens: int = Field(700, ge=50, le=8000)

    # --- База знаний ---
    knowledge_dir: Path = Path("knowledge")
    # Сколько самых подходящих фрагментов подставлять в запрос. 0 — всю базу целиком
    knowledge_top_k: int = Field(5, ge=0, le=50)
    # Максимальный размер фрагмента в символах
    chunk_size: int = Field(1200, ge=200, le=10000)
    # Файлы, которые подставляются в каждый запрос целиком (например, контакты менеджера)
    knowledge_always: StrList = ("contacts.md",)

    # --- Диалог ---
    # Сколько последних сообщений (вопросов и ответов вместе) бот помнит
    history_size: int = Field(6, ge=0, le=50)
    max_question_length: int = Field(1000, ge=50, le=4000)

    @field_validator("llm_base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("адрес должен начинаться с http:// или https://")
        return value.rstrip("/")

    # Примеры вопросов в приветствии (кнопки). В .env — через «|»
    example_questions: StrList = (
        "Сколько стоит капучино на овсяном молоке?",
        "До скольки вы работаете в субботу?",
        "Сколько стоит доставка?",
        "Можно прийти с собакой?",
    )

    @field_validator("knowledge_always", mode="before")
    @classmethod
    def _parse_names(cls, value: object) -> object:
        return _split(value, ",")

    @field_validator("example_questions", mode="before")
    @classmethod
    def _parse_questions(cls, value: object) -> object:
        return _split(value, "|")

    @field_validator("example_questions")
    @classmethod
    def _check_questions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        # Текст кнопки в Telegram — не длиннее 64 символов, кнопок разумно не больше 8
        if len(value) > 8 or any(len(q) > 64 for q in value):
            raise ValueError("до 8 примеров, каждый не длиннее 64 символов")
        return value

    @field_validator("knowledge_dir")
    @classmethod
    def _absolute_dir(cls, value: Path) -> Path:
        return value if value.is_absolute() else BASE_DIR / value


def load_config() -> Config:
    try:
        return Config()
    except ValidationError as e:
        problems = "\n".join(
            f"  {'.'.join(map(str, err['loc'])).upper() or 'настройки'}: {err['msg']}"
            for err in e.errors()
        )
        raise RuntimeError(f"Ошибка в настройках .env:\n{problems}") from None
