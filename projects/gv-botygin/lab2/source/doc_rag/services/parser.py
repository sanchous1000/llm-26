"""PDF -> Markdown по страницам.
"""
import io
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

MAX_HEADING_LEVEL = 3
MAX_HEADING_LEN = 200
# стиль (размер, жирность) считается заголовочным, только если им набрана малая доля текста
MAX_HEADING_STYLE_SHARE = 0.15
# колонтитул: нормализованный текст повторяется на >= этой доли страниц (и не менее чем на 3)
HEADER_FOOTER_MIN_SHARE = 0.3
HEADER_FOOTER_ZONE = 0.10  # доля высоты страницы сверху/снизу

BULLET_RE = re.compile(r"^([•·●▪■◦\-–—*])\s+")
NUMBERED_RE = re.compile(r"^(\d{1,3}[.)]|[а-яa-z][.)])\s+", re.IGNORECASE)
HYPHEN_END_RE = re.compile(r"\w-$")
TOC_LINE_RE = re.compile(r"\.{5,}\s*\d+\s*$")  # строка оглавления: «Раздел.......12»
# лигатуры TeX-шрифтов, которые pdfplumber отдаёт как (cid:N)
ROMAN_CAPS_RE = re.compile(r"[IVX]{1,5}\.\s+[A-ZА-Я][A-ZА-Я0-9\s\-–,&:]{2,80}")
LETTER_SECTION_RE = re.compile(r"[A-Z]\.\s+\S")
NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,2})\.?\s+\S")
CID_RE = re.compile(r"\(cid:(\d+)\)")
CID_LIGATURES = {11: "ff", 12: "fi", 13: "fl", 14: "ffi", 15: "ffl"}


@dataclass
class PageMarkdown:
    page: int  # нумерация с 1
    markdown: str


@dataclass
class _Line:
    text: str
    top: float
    bottom: float
    x0: float
    size: float
    bold: bool
    chars: int
    col: int = 0  # 0/1 - левая/правая колонка, 2 - строка на всю ширину страницы
    hint: int = 0  # типографский заголовок без выделения кеглем: 1 - «III. CAPS», 2 - «A. Italic»


@dataclass
class _Table:
    top: float
    markdown: str
    col: int = 0


class ParserService:
    def parse_pdf(self, source: bytes | str | Path) -> list[PageMarkdown]:
        """Возвращает постраничный Markdown. Пустые страницы пропускаются."""
        stream = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source

        with pdfplumber.open(stream) as pdf:
            raw_pages = []  # [(page_number, height, blocks)]
            for page in pdf.pages:
                raw_pages.append((page.page_number, float(page.height), self._extract_blocks(page)))

        pages_lines = [[b for b in blocks if isinstance(b, _Line)] for _, _, blocks in raw_pages]
        body_size = self._body_size(pages_lines)
        repeated = self._repeated_margin_texts(raw_pages)
        # типографские заголовки получают «виртуальный» кегль чуть выше текста: римские выше буквенных
        for lines in pages_lines:
            for ln in lines:
                if ln.hint:
                    ln.size, ln.bold = body_size + 0.03 - 0.01 * ln.hint, True
        levels = self._heading_levels(pages_lines, body_size)

        result = []
        for number, height, blocks in raw_pages:
            blocks = [b for b in blocks if not self._is_noise(b, height, repeated)]
            markdown = self._render(blocks, body_size, levels)
            if markdown.strip():
                result.append(PageMarkdown(page=number, markdown=markdown))
        return result

    # ------------------------------------------------------------------ извлечение

    def _extract_blocks(self, page) -> list[_Line | _Table]:
        tables = [t for t in page.find_tables() if self._is_real_table(t)]
        table_blocks = []
        for t in tables:
            md = self._table_to_md(t.extract())
            if md:
                table_blocks.append((t, _Table(top=t.bbox[1], markdown=md)))

        # x_tolerance_ratio: порог разрыва слов зависит от кегля - иначе в выровненном по ширине
        # тексте с узкими пробелами слова склеиваются
        words = page.extract_words(
            extra_attrs=["size", "fontname", "upright"], keep_blank_chars=False, x_tolerance_ratio=0.1
        )
        # повёрнутый текст (штампы на полях, вертикальные подписи) - не часть основного потока
        words = [w for w in words if w["upright"] and not self._in_tables(w, tables)]

        gutter = self._find_gutter(words, float(page.width))
        if gutter is None:
            lines = self._group_lines(words, col=0)
            for _, tb in table_blocks:
                tb.col = 0
        else:
            def col_of(x0: float, x1: float) -> int:
                if x1 <= gutter + 1:
                    return 0
                if x0 >= gutter - 1:
                    return 1
                return 2  # пересекает желоб - заголовок/рисунок на всю ширину

            word_col = {id(w): col_of(w["x0"], w["x1"]) for w in words}
            # строка, в которой слова идут через желоб с обычным межсловным пробелом (например,
            # крупный заголовок статьи), - строка на всю ширину, а не две колонки
            for row in self._cluster_rows(words):
                row.sort(key=lambda w: w["x0"])
                if any(word_col[id(w)] == 2 for w in row):
                    word_col.update({id(w): 2 for w in row})
                    continue
                for a, b in zip(row, row[1:]):
                    if a["x1"] <= gutter < b["x0"] and b["x0"] - a["x1"] < 0.6 * max(a["size"], b["size"]):
                        word_col.update({id(w): 2 for w in row})
                        break
            lines = []
            for col in (0, 1, 2):
                part = [w for w in words if word_col[id(w)] == col]
                lines += self._group_lines(part, col=col)
            for t, tb in table_blocks:
                tb.col = col_of(t.bbox[0], t.bbox[2])

        blocks = lines + [tb for _, tb in table_blocks]
        self._normalize_indent(blocks)
        return self._reading_order(blocks)

    @staticmethod
    def _find_gutter(words: list[dict], width: float) -> float | None:
        """x-координата желоба между двумя колонками текста или None для одной колонки."""
        n = len(words)
        if n < 40:
            return None
        lo, hi = int(width * 0.3), int(width * 0.7)
        cover = [0] * (hi - lo + 1)
        for w in words:
            for x in range(max(int(w["x0"]), lo), min(int(w["x1"]), hi) + 1):
                cover[x - lo] += 1

        limit = 0.04 * n
        best, run_start = (0, 0), None
        for i, c in enumerate(cover + [limit + 1]):
            if c <= limit and run_start is None:
                run_start = i
            elif c > limit and run_start is not None:
                if i - run_start > best[1] - best[0]:
                    best = (run_start, i)
                run_start = None
        start, end = lo + best[0], lo + best[1]
        if end - start < 6:
            return None

        left = sum(1 for w in words if w["x1"] <= end)
        right = sum(1 for w in words if w["x0"] >= start)
        if left < 0.2 * n or right < 0.2 * n:
            return None
        return (start + end) / 2

    @staticmethod
    def _normalize_indent(blocks: list) -> None:
        """x0 строк - относительно левого края своей колонки (для определения абзацного отступа)."""
        lines = [b for b in blocks if isinstance(b, _Line)]
        for col in {ln.col for ln in lines}:
            group = [ln for ln in lines if ln.col == col]
            counts = Counter()
            for ln in group:
                counts[round(ln.x0 / 2)] += ln.chars
            base = counts.most_common(1)[0][0] * 2  # абзацный отступ считаем от основного левого края
            for ln in group:
                ln.x0 -= base

    @staticmethod
    def _reading_order(blocks: list) -> list:
        """Одна колонка - сверху вниз. Две - полосы между строками на всю ширину: левая колонка,
        правая колонка, затем сама строка на всю ширину."""
        full = sorted((b for b in blocks if b.col == 2), key=lambda b: b.top)

        def key(b):
            if b.col == 2:
                return (full.index(b), 2, b.top)
            band = sum(1 for f in full if f.top < b.top)
            return (band, b.col, b.top)

        return sorted(blocks, key=key)

    @staticmethod
    def _is_real_table(table) -> bool:
        rows = table.extract()
        return len(rows) >= 2 and max(len(r) for r in rows) >= 2

    @staticmethod
    def _in_tables(word, tables) -> bool:
        cx = (word["x0"] + word["x1"]) / 2
        cy = (word["top"] + word["bottom"]) / 2
        return any(t.bbox[0] <= cx <= t.bbox[2] and t.bbox[1] <= cy <= t.bbox[3] for t in tables)

    @staticmethod
    def _is_bold(fontname: str) -> bool:
        name = fontname.lower()
        return any(k in name for k in ("bold", "black", "heavy"))

    @staticmethod
    def _cluster_rows(words: list[dict]) -> list[list[dict]]:
        """Слова с близкой вертикальной позицией - в одну строку."""
        groups: list[list[dict]] = []
        for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
            mid = (w["top"] + w["bottom"]) / 2
            if groups:
                ref = groups[-1]
                ref_mid = sum((x["top"] + x["bottom"]) / 2 for x in ref) / len(ref)
                if abs(mid - ref_mid) < 0.5 * max(w["size"], ref[0]["size"]):
                    ref.append(w)
                    continue
            groups.append([w])
        return groups

    def _group_lines(self, words: list[dict], col: int = 0) -> list[_Line]:
        lines = []
        for g in self._cluster_rows(words):
            g.sort(key=lambda w: w["x0"])
            text = g[0]["text"]
            for prev, cur in zip(g, g[1:]):
                gap = cur["x0"] - prev["x1"]
                text += ("" if gap < 0.1 * cur["size"] else " ") + cur["text"]
            text = CID_RE.sub(lambda m: CID_LIGATURES.get(int(m.group(1)), ""), text)
            italic_chars = sum(len(w["text"]) for w in g if "italic" in w["fontname"].lower() or "oblique" in w["fontname"].lower())
            chars = sum(len(w["text"]) for w in g)
            size_counts = Counter()
            bold_chars = 0
            for w in g:
                size_counts[round(w["size"], 1)] += len(w["text"])
                if self._is_bold(w["fontname"]):
                    bold_chars += len(w["text"])
            text = text.strip()
            if not text:
                continue
            size = size_counts.most_common(1)[0][0]
            bold = bold_chars >= 0.8 * chars
            # научные статьи: разделы набраны капсом «III. METHODOLOGY» (капитель мельче текста) или
            # курсивом «A. Subsection» - кеглем их не отличить, определяем по виду строки
            hint = 0
            if ROMAN_CAPS_RE.fullmatch(text):
                hint = 1
            elif italic_chars >= 0.8 * chars and LETTER_SECTION_RE.match(text) and len(text) <= 100:
                hint = 2
            lines.append(
                _Line(
                    text=text,
                    top=min(w["top"] for w in g),
                    bottom=max(w["bottom"] for w in g),
                    x0=g[0]["x0"],
                    size=size,
                    bold=bold,
                    chars=chars,
                    col=col,
                    hint=hint,
                )
            )
        return lines

    @staticmethod
    def _table_to_md(rows: list[list]) -> str:
        def clean(cell) -> str:
            return re.sub(r"\s+", " ", cell or "").strip().replace("|", "\\|")

        rows = [[clean(c) for c in r] for r in rows]
        rows = [r for r in rows if any(r)]
        if len(rows) < 2:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        # компактный GitHub-Markdown без выравнивания пробелами (они впустую тратят токены чанка)
        out = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
        out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n".join(out)

    # ------------------------------------------------------------------ статистика

    @staticmethod
    def _body_size(pages_lines: list[list[_Line]]) -> float:
        counts = Counter()
        for lines in pages_lines:
            for ln in lines:
                counts[ln.size] += ln.chars
        return counts.most_common(1)[0][0] if counts else 0.0

    @staticmethod
    def _norm(text: str) -> str:
        """Ключ колонтитула: цифры заменены на #, порядок слов не важен (строки из двух колонок
        могут склеиваться по-разному на разных страницах)."""
        return " ".join(sorted(re.sub(r"\d+", "#", text.lower()).split()))

    def _in_margin(self, block, height: float) -> bool:
        return block.bottom < height * HEADER_FOOTER_ZONE or block.top > height * (1 - HEADER_FOOTER_ZONE)

    def _repeated_margin_texts(self, raw_pages) -> set[str]:
        """Тексты в верхнем/нижнем поле, повторяющиеся на многих страницах (колонтитулы)."""
        counts = Counter()
        for _, height, blocks in raw_pages:
            seen = {
                self._norm(b.text)
                for b in blocks
                if isinstance(b, _Line) and self._in_margin(b, height)
            }
            counts.update(seen)
        threshold = max(3, HEADER_FOOTER_MIN_SHARE * len(raw_pages))
        return {t for t, c in counts.items() if c >= threshold}

    def _is_noise(self, block, height: float, repeated: set[str]) -> bool:
        if not isinstance(block, _Line):
            return False
        if TOC_LINE_RE.search(block.text):  # оглавление дублирует заголовки и засоряет поиск
            return True
        if not self._in_margin(block, height):
            return False
        # номер страницы («12», «Page 5») или повторяющийся колонтитул
        return (
            bool(re.fullmatch(r"[\W\d]*\d+[\W\d]*", block.text))
            or bool(re.search(r"page\s+\d+", block.text, re.IGNORECASE))
            or self._norm(block.text) in repeated
        )

    def _is_heading_candidate(self, ln: _Line, body_size: float) -> bool:
        if len(ln.text) > MAX_HEADING_LEN or ln.text.endswith((",", ";")):
            return False
        if re.fullmatch(r"[\W\d]+", ln.text):  # одни цифры/знаки
            return False
        if ln.size > body_size + 0.5:
            return True
        return ln.bold and ln.size >= body_size - 0.5

    def _heading_levels(self, pages_lines: list[list[_Line]], body_size: float) -> dict[tuple, int]:
        """Стиль (размер, жирность) -> уровень заголовка 1..3."""
        total = sum(ln.chars for lines in pages_lines for ln in lines) or 1
        style_chars: Counter = Counter()
        style_lines: Counter = Counter()
        for lines in pages_lines:
            for ln in lines:
                if self._is_heading_candidate(ln, body_size):
                    style_chars[(ln.size, ln.bold)] += ln.chars
                    style_lines[(ln.size, ln.bold)] += 1

        styles = [s for s, c in style_chars.items() if c / total <= MAX_HEADING_STYLE_SHARE]
        # стили, встречающиеся только на первой странице (название, авторы), - не разделы:
        # иначе они становились бы h1 и попадали в путь заголовков каждого чанка
        later = {(ln.size, ln.bold) for lines in pages_lines[1:] for ln in lines}
        titles = {s for s in styles if s not in later and style_lines[s] <= 6}
        if len(styles) > len(titles):
            styles = [s for s in styles if s not in titles]
        styles.sort(key=lambda s: (-s[0], not s[1]))
        return {s: min(i + 1, MAX_HEADING_LEVEL) for i, s in enumerate(styles)}

    # ------------------------------------------------------------------ сборка Markdown

    def _render(self, blocks: list[_Line | _Table], body_size: float, levels: dict[tuple, int]) -> str:
        lines_only = [b for b in blocks if isinstance(b, _Line)]
        leading = self._leading(lines_only)
        left = 0.0  # x0 уже отсчитан от основного левого края колонки

        out: list[str] = []
        para: list[str] = []  # строки текущего абзаца
        heading: list[_Line] = []

        def flush_para():
            if para:
                out.append(self._join_para(para))
                para.clear()

        def flush_heading():
            if heading:
                text = " ".join(h.text for h in heading)
                # вложенность «1.», «1.1», «4.4.2» надёжнее, чем кегль: все пункты часто набраны одним стилем
                m = NUMBERED_HEADING_RE.match(text)
                level = min(m.group(1).count(".") + 1, MAX_HEADING_LEVEL) if m else levels[(heading[0].size, heading[0].bold)]
                out.append("#" * level + " " + text)
                heading.clear()

        prev: _Line | None = None
        for b in blocks:
            if isinstance(b, _Table):
                flush_heading()
                flush_para()
                out.append(b.markdown)
                prev = None
                continue

            level = levels.get((b.size, b.bold)) if self._is_heading_candidate(b, body_size) else None
            # переход в другую колонку/полосу: вертикальный зазор не определён
            col_break = prev is not None and (b.col != prev.col or b.top < prev.top)
            gap = 0.0 if prev is None or col_break else b.top - prev.bottom

            if level:
                flush_para()
                same = heading and (heading[0].size, heading[0].bold) == (b.size, b.bold) and gap < b.size
                if heading and not same:
                    flush_heading()
                heading.append(b)
            else:
                flush_heading()
                # абзац может продолжаться в соседней колонке, если предыдущая строка не закончила предложение
                continues = col_break and bool(para) and not re.search(r"[.!?:]$", para[-1])
                new_par = (
                    (prev is None or col_break) and not continues
                    or gap > max(1.5 * leading, 0.3 * b.size)
                    or b.x0 > left + 0.8 * b.size
                    or BULLET_RE.match(b.text)
                    or NUMBERED_RE.match(b.text)
                )
                if new_par:
                    flush_para()
                para.append(b.text)
            prev = b

        flush_heading()
        flush_para()
        return "\n\n".join(out)

    @staticmethod
    def _leading(lines: list[_Line]) -> float:
        """Типичный межстрочный зазор (bottom -> top следующей строки) внутри абзаца."""
        gaps = [b.top - a.bottom for a, b in zip(lines, lines[1:]) if 0 <= b.top - a.bottom < 2 * a.size]
        return statistics.median(gaps) if gaps else 0.0

    @staticmethod
    def _join_para(lines: list[str]) -> str:
        text = lines[0]
        for nxt in lines[1:]:
            if HYPHEN_END_RE.search(text) and nxt[:1].islower():
                text = text[:-1] + nxt  # перенос слова
            else:
                text += " " + nxt
        text = re.sub(r"[ \t]+", " ", text).strip()
        m = BULLET_RE.match(text)
        if m:
            text = "- " + text[m.end():]
        return text


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    for p in ParserService().parse_pdf(Path(sys.argv[1])):
        print(f"<!-- page {p.page} -->\n{p.markdown}\n")
