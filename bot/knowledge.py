"""База знаний: читает .md/.txt из папки, режет на фрагменты и ищет подходящие.

Поиск — BM25 по словам с упрощённым русским стеммингом, без векторной БД и внешних
сервисов. Для базы в десятки–сотни страниц этого хватает; при росте базы класс можно
заменить на векторный поиск, не трогая остальной код (нужен только метод search).

Файлы перечитываются автоматически, если их заменили, добавили или удалили, —
перезапускать бота после правки базы не нужно."""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

EXTENSIONS = (".md", ".txt")

WORD_RE = re.compile(r"[a-zа-я0-9]+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

# Окончания русских слов: «доставка», «доставки», «доставку» → «доставк».
# Грубо, но для поиска по небольшой базе достаточно. Длинные окончания — первыми.
ENDINGS = sorted(
    """
    иями ями ами ией ого его ому ему ыми ими ешь ете ишь ите ают яют ует юет ить ать ять еть уть
    ться тся ая яя ое ее ые ие ий ый ой ей ую юю ом ем ах ях ов ев ам ям ия ье ья ью ию ым им ых
    их ут ют ит ет ат ят ть ся а я о е ы и у ю ь й
    """.split(),
    key=len,
    reverse=True,
)
MIN_STEM = 3
# После отрезания окончания оставляем не больше 5 букв: «работаете» и «работы» → «работ»,
# «находитесь» и «находимся» → «наход». Совпадений-ошибок от этого заметно меньше, чем пользы
MAX_STEM = 5

STOP_WORDS = frozenset(
    """
    а без бы в во вам вас вы где да для до если есть еще же за и из или им их к как ко когда
    ли мне мы на над не нет но о об от по под при про с со так там то у уже чем что чтобы это
    я ваш ваша ваше ваши мой моя можно какой какая какие который сколько ли бы
    the a an and or of to in on for is are what how
    """.split()
)


def stem(word: str) -> str:
    for ending in ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= MIN_STEM:
            return word[: -len(ending)][:MAX_STEM]
    return word[:MAX_STEM]


def tokenize(text: str) -> list[str]:
    words = WORD_RE.findall(text.lower().replace("ё", "е"))
    return [stem(w) for w in words if w not in STOP_WORDS]


@dataclass(frozen=True)
class Chunk:
    source: str  # имя файла
    title: str  # заголовки раздела: «Меню и цены / Завтраки»
    text: str

    def render(self) -> str:
        header = f"[{self.source} — {self.title}]" if self.title else f"[{self.source}]"
        return f"{header}\n{self.text}"


def _split_long(text: str, size: int) -> list[str]:
    """Режет текст по абзацам, затем по строкам, чтобы куски были не длиннее size."""
    if len(text) <= size:
        return [text]
    parts: list[str] = []
    current = ""
    for block in re.split(r"\n\s*\n", text):
        pieces = [block] if len(block) <= size else block.splitlines()
        for piece in pieces:
            # Совсем длинная строка без переносов — режем как есть
            while len(piece) > size:
                if current:
                    parts.append(current)
                    current = ""
                parts.append(piece[:size])
                piece = piece[size:]
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) > size:
                parts.append(current)
                current = piece
            else:
                current = candidate
    if current:
        parts.append(current)
    return [p.strip() for p in parts if p.strip()]


def split_document(source: str, text: str, chunk_size: int) -> list[Chunk]:
    """Делит документ по заголовкам Markdown. У каждого фрагмента — путь заголовков,
    чтобы модель понимала контекст («Кофейня на Баумана» → график относится к ней)."""
    chunks: list[Chunk] = []
    headings: dict[int, str] = {}  # уровень заголовка → текст
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        title = " / ".join(headings[level] for level in sorted(headings))
        for part in _split_long(body, chunk_size) if body else []:
            chunks.append(Chunk(source=source, title=title, text=part))
        lines.clear()

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if not match:
            lines.append(line)
            continue
        flush()
        level = len(match.group(1))
        for deeper in [lvl for lvl in headings if lvl >= level]:
            del headings[deeper]
        headings[level] = match.group(2)
    flush()
    return chunks


@dataclass
class KnowledgeBase:
    directory: Path
    chunk_size: int = 1200
    always_include: tuple[str, ...] = ()

    chunks: list[Chunk] = field(default_factory=list, init=False)
    _signature: tuple = field(default=(), init=False)
    _tokens: list[Counter] = field(default_factory=list, init=False)
    _doc_freq: Counter = field(default_factory=Counter, init=False)
    _avg_len: float = field(default=0.0, init=False)

    def _files(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(
            p for p in self.directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

    def refresh(self) -> bool:
        """Перечитывает базу, если файлы изменились. Возвращает True, если перечитал."""
        files = self._files()
        signature = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        if signature == self._signature:
            return False
        chunks: list[Chunk] = []
        for path in files:
            text = path.read_text(encoding="utf-8-sig")
            chunks.extend(split_document(path.name, text, self.chunk_size))
        self._index(chunks)
        self._signature = signature
        logger.info("База знаний: %d файлов, %d фрагментов", len(files), len(chunks))
        return True

    def _index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        # В индекс попадает и заголовок: вопрос «где вы находитесь» найдёт раздел «Адреса»
        self._tokens = [Counter(tokenize(f"{c.title}\n{c.text}")) for c in chunks]
        self._doc_freq = Counter()
        for tokens in self._tokens:
            self._doc_freq.update(tokens.keys())
        lengths = [sum(t.values()) for t in self._tokens]
        self._avg_len = sum(lengths) / len(lengths) if lengths else 0.0

    def search(self, query: str, top_k: int) -> list[Chunk]:
        """Самые подходящие фрагменты (BM25). Фрагменты без единого совпадения не берём."""
        terms = set(tokenize(query))
        if not terms or not self.chunks:
            return []
        k1, b = 1.5, 0.75
        n = len(self.chunks)
        scored = []
        for index, tokens in enumerate(self._tokens):
            length = sum(tokens.values())
            score = 0.0
            for term in terms:
                tf = tokens.get(term, 0)
                if not tf:
                    continue
                df = self._doc_freq[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / self._avg_len))
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self.chunks[index] for _, index in scored[:top_k]]

    def outline(self) -> list[str]:
        """Оглавление базы: все разделы без повторов. Подставляется в каждый запрос,
        чтобы на общий вопрос («какой у вас ассортимент?») модель видела, что вообще
        есть в базе, и не додумывала."""
        self.refresh()
        return list(dict.fromkeys(c.title or c.source for c in self.chunks))

    def always(self) -> list[Chunk]:
        names = {name.lower() for name in self.always_include}
        return [c for c in self.chunks if c.source.lower() in names]

    def context_for(self, query: str, top_k: int, previous: str = "") -> list[Chunk]:
        """Фрагменты для запроса к ИИ: обязательные файлы + найденные поиском.
        previous — прошлый вопрос клиента: уточнение «а в выходные?» ищется вместе с ним,
        но сначала идут фрагменты по самому вопросу, чтобы смена темы не сбивала поиск.
        top_k=0 — вся база целиком (удобно, пока база маленькая)."""
        self.refresh()
        if top_k == 0:
            return list(self.chunks)
        always = self.always()
        limit = top_k + len(always)
        candidates = self.search(query, limit)
        if previous:
            candidates += self.search(f"{previous}\n{query}", limit)
        found: list[Chunk] = []
        for chunk in candidates:
            if len(found) == top_k:
                break
            if chunk not in found and chunk not in always:
                found.append(chunk)
        # Найденное — первым: модель внимательнее к началу контекста, и служебные
        # файлы вроде контактов не перебивают ответ по существу
        return found + always
