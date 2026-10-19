"""Клиент ИИ через OpenAI-совместимый API.

Один и тот же код работает с OpenAI, OpenRouter, ProxyAPI, DeepSeek, локальным Ollama
и любым другим сервисом с таким же API — меняются только LLM_BASE_URL, LLM_API_KEY
и LLM_MODEL в .env."""

import logging
from typing import Protocol

import openai
from openai import AsyncOpenAI

from bot.config import Config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """ИИ не ответил: нет связи, неверный ключ, кончились лимиты и т. п."""


class LLM(Protocol):
    async def complete(self, messages: list[dict[str, str]]) -> str: ...


class OpenAICompatibleLLM:
    def __init__(self, config: Config) -> None:
        self.model = config.llm_model
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens
        self.client = AsyncOpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            timeout=config.llm_timeout,
            max_retries=1,
        )

    async def complete(self, messages: list[dict[str, str]]) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except openai.OpenAIError as e:
            logger.error("Ошибка ИИ (%s): %s", type(e).__name__, e)
            raise LLMError(str(e)) from e
        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise LLMError("пустой ответ модели")
        return content

    async def close(self) -> None:
        await self.client.close()
