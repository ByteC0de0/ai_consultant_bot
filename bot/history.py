"""Память диалога: последние N сообщений каждого чата (в оперативной памяти)."""

from collections import deque


class ChatHistory:
    def __init__(self, size: int) -> None:
        self.size = size
        self._chats: dict[int, deque[dict[str, str]]] = {}

    def get(self, chat_id: int) -> list[dict[str, str]]:
        return list(self._chats.get(chat_id, ()))

    def add(self, chat_id: int, question: str, answer: str) -> None:
        if self.size == 0:
            return
        messages = self._chats.setdefault(chat_id, deque(maxlen=self.size))
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

    def last_question(self, chat_id: int) -> str:
        for message in reversed(self._chats.get(chat_id, ())):
            if message["role"] == "user":
                return message["content"]
        return ""

    def clear(self, chat_id: int) -> None:
        self._chats.pop(chat_id, None)
