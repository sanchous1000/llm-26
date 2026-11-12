"""Шаг 2. Конвертация исходных документов в нормализованный Markdown с метаданными.

Поддерживаемые форматы: DOCX (материалы НИР), XLSX (план НИР), PDF (статьи-источники),
MD / YAML / PY / SH (репозиторий с результатами).

Каждый документ сохраняется в data/markdown/<id>.md:
    ---
    doc_id, title, source_type, format, path, date, page_mode, ...
    ---
    # Заголовки / списки / таблицы ...
    <!-- page: N -->      ← метки страниц (PDF — точные, DOCX — по разрывам страниц Word)

Запуск из каталога lab2:  python source/parse_docs.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import yaml

import settings
from settings import nfc

PAGE_MARK = "<!-- page: {} -->"


# ============================================================ общие утилиты
def clean_text(text: str) -> str:
    text = nfc(text or "")
    text = text.replace("­", "").replace(" ", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def md_cell(text: str) -> str:
    return clean_text(str(text)).replace("\n", " ").replace("|", "\\|")


def md_table(rows: list[list[str]]) -> str:
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def file_date(path: Path) -> str:
    return dt.date.fromtimestamp(path.stat().st_mtime).isoformat()


def doc_id_for(rel_path: str) -> str:
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", rel_path)
    return re.sub(r"[\\/]+", "__", re.sub(r"[^\w\-./ ()]+", "", stem)).replace(" ", "_")


def semester_of(rel_path: str) -> str | None:
    m = re.search(r"Семестр\s+(I{1,3}|IV)", rel_path)
    return m.group(1) if m else None


# ============================================================ DOCX
def _docx_iter_blocks(document):
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def xml_text(element) -> str:
    """Текст из всех w:t/w:tab/w:br элемента, включая content controls (w:sdt),
    которые python-docx Paragraph.text пропускает."""
    from docx.oxml.ns import qn

    t_tag, tab_tag, br_tag = qn("w:t"), qn("w:tab"), qn("w:br")
    out = []
    for el in element.iter(t_tag, tab_tag, br_tag):
        if el.tag == t_tag:
            out.append(el.text or "")
        elif el.tag == tab_tag:
            out.append(" ")
        elif el.get(qn("w:type")) != "page":
            out.append("\n")
    return "".join(out)


def _heading_level_from_style(par) -> int | None:
    name = (par.style.name if par.style is not None else "").lower()
    m = re.match(r"(heading|заголовок)\s*(\d)", name)
    if m:
        return int(m.group(2))
    if name in ("title", "название"):
        return 1
    return None


def _is_toc(par) -> bool:
    name = (par.style.name if par.style is not None else "").lower()
    return name.startswith("toc") or "оглавлен" in name or name.startswith("table of contents")


def _run_bold(run, par) -> bool:
    if run.bold is not None:
        return bool(run.bold)
    st = run.style or par.style
    return bool(st is not None and st.font is not None and st.font.bold)


def _bold_prefix(par) -> tuple[str, str]:
    """Разделить абзац на жирный префикс и остальной текст."""
    bold, rest, in_prefix = [], [], True
    for run in par.runs:
        if not run.text:
            continue
        if in_prefix and _run_bold(run, par):
            bold.append(run.text)
        else:
            in_prefix = False
            rest.append(run.text)
    return clean_text("".join(bold)), clean_text("".join(rest))


_NUM_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.{1,120})$")


def _numbered_heading(text: str) -> tuple[int, str] | None:
    """«1.1. Предмет работы.» → (3, «1.1. Предмет работы»)."""
    text = text.rstrip(" .:")
    m = _NUM_HEADING.match(text)
    if not m or len(text) > 130:
        return None
    depth = m.group(1).count(".") + 1
    return min(1 + depth, 6), text


def _is_list(par) -> tuple[bool, int]:
    from docx.oxml.ns import qn

    ppr = par._p.pPr
    num = ppr.find(qn("w:numPr")) if ppr is not None else None
    if num is not None:
        ilvl = num.find(qn("w:ilvl"))
        return True, int(ilvl.get(qn("w:val"))) if ilvl is not None else 0
    name = (par.style.name if par.style is not None else "").lower()
    return ("list" in name or "списо" in name), 0


def _count_breaks(element, use_rendered: bool) -> int:
    from docx.oxml.ns import qn

    if use_rendered:
        return len(element.findall(".//" + qn("w:lastRenderedPageBreak")))
    explicit = sum(1 for br in element.iter(qn("w:br")) if br.get(qn("w:type")) == "page")
    before = sum(1 for pb in element.iter(qn("w:pageBreakBefore"))
                 if pb.get(qn("w:val"), "true") not in ("false", "0"))
    return explicit + before


def _docx_table(table) -> list[list[str]]:
    rows = []
    for row in table.rows:
        cells, prev = [], None
        for cell in row.cells:
            # python-docx повторяет объединённые ячейки — убираем подряд идущие дубли
            if prev is not None and cell._tc is prev:
                continue
            prev = cell._tc
            cells.append(md_cell(xml_text(cell._tc)))
        rows.append(cells)
    return rows


def parse_docx(path: Path) -> tuple[str, dict]:
    import docx

    document = docx.Document(str(path))
    body = document.element.body
    from docx.oxml.ns import qn

    use_rendered = body.find(".//" + qn("w:lastRenderedPageBreak")) is not None
    has_breaks = use_rendered or _count_breaks(body, False) > 0
    blocks = list(_docx_iter_blocks(document))
    has_styled_headings = any(
        getattr(b, "style", None) is not None and hasattr(b, "runs") and _heading_level_from_style(b)
        and clean_text(xml_text(b._p)) for b in blocks)

    out: list[str] = []
    page = 1
    if has_breaks:
        out.append(PAGE_MARK.format(page))
    n_headings = 0

    def emit_page(new_page: int):
        nonlocal page
        if has_breaks and new_page != page:
            page = new_page
            out.append(PAGE_MARK.format(page))

    for block in blocks:
        if hasattr(block, "rows"):  # таблица
            rows = _docx_table(block)
            width = max((len(r) for r in rows), default=0)
            if width <= 1:  # «рамка» из одной ячейки — это просто текст
                out += [r[0] for r in rows if r and r[0]]
            else:
                out.append(md_table(rows))
            emit_page(page + _count_breaks(block._tbl, use_rendered))
            continue

        par = block
        breaks = _count_breaks(par._p, use_rendered)
        if breaks:
            emit_page(page + breaks)
        if _is_toc(par):
            continue
        text = clean_text(xml_text(par._p))
        if not text:
            continue

        level = _heading_level_from_style(par)
        if level:
            out.append("#" * level + " " + text)
            n_headings += 1
            continue

        if not has_styled_headings:
            # ТЗ и задания оформлены без стилей: заголовки — жирные строки с номером
            bold, rest = _bold_prefix(par)
            nh = _numbered_heading(bold) if bold else None
            if nh and (not rest or bold.endswith(".")):
                out.append("#" * nh[0] + " " + nh[1])
                n_headings += 1
                if rest:
                    out.append(rest)
                continue
            if bold and not rest and not out_has_heading(out) and len(bold) < 100:
                out.append("# " + bold)  # первая жирная строка — название документа
                n_headings += 1
                continue

        is_list, ilvl = _is_list(par)
        out.append(("  " * ilvl + "- " + text) if is_list else text)

    # Разрывы бывают только между крупными частями (документ собран не в Word, нет
    # lastRenderedPageBreak). Если на «странице» больше ~5 реальных страниц текста —
    # номера страниц вводят в заблуждение: убираем их, цитируем по разделам.
    if has_breaks:
        seg_words, words_now = [], 0
        for item in out:
            if item.startswith("<!-- page"):
                seg_words.append(words_now)
                words_now = 0
            else:
                words_now += len(re.findall(r"\w+", item))
        seg_words.append(words_now)
        if max(seg_words) > 1500:
            out = [x for x in out if not x.startswith("<!-- page")]
            has_breaks = False

    props = document.core_properties
    meta = {
        "format": "docx",
        "page_mode": "approx" if has_breaks else "none",
        "pages": page if has_breaks else None,
        "n_headings": n_headings,
        "doc_date": (props.modified or props.created).date().isoformat()
        if (props.modified or props.created) else None,
    }
    return join_blocks(out), meta


def out_has_heading(out: list[str]) -> bool:
    return any(line.startswith("#") for line in out)


def join_blocks(blocks: list[str]) -> str:
    """Склеить блоки: соседние пункты списка — через перевод строки, остальное — через пустую."""
    text, prev_list = [], False
    for b in blocks:
        is_list = b.lstrip().startswith("- ")
        if text:
            text.append("\n" if (is_list and prev_list) else "\n\n")
        text.append(b)
        prev_list = is_list
    return "".join(text) + "\n"


# ============================================================ XLSX
def _xlsx_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, dt.datetime):
        return v.date().isoformat()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return md_cell(str(v).replace("\n", " "))


def parse_xlsx(path: Path, sheets_cfg: list[dict] | None) -> tuple[str, dict]:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    names = [nfc(n) for n in wb.sheetnames]
    wanted = sheets_cfg or [{"sheet": n} for n in names]
    out = [f"# {nfc(path.stem)}"]
    used = []
    for item in wanted:
        sheet = nfc(item["sheet"])
        if sheet not in names:
            print(f"  [warn] {path.name}: нет листа «{sheet}»")
            continue
        ws = wb[wb.sheetnames[names.index(sheet)]]
        rows = [[_xlsx_value(c) for c in r] for r in ws.iter_rows(values_only=True)]
        rows = [r for r in rows if any(r)]
        if not rows:
            continue
        width = max(max((i + 1 for i, c in enumerate(r) if c), default=0) for r in rows)
        rows = [r[:width] for r in rows]
        header_idx = next((i for i, r in enumerate(rows) if sum(bool(c) for c in r) >= 3), None)
        out.append(f"## {item.get('title', sheet)}")
        used.append(sheet)
        preamble = rows[:header_idx] if header_idx is not None else rows
        if not item.get("skip_preamble"):
            out += [" ".join(c for c in r if c) for r in preamble]
        if header_idx is not None:
            table = rows[header_idx:]
            ncols = max(sum(1 for c in r if c) for r in table)
            table = [[c for c in r][:width] for r in table]
            # убираем полностью пустые столбцы
            keep = [j for j in range(width) if any(r[j] for r in table)]
            out.append(md_table([[r[j] for j in keep] for r in table]) if ncols > 1 else "")
    return join_blocks([b for b in out if b]), {
        "format": "xlsx", "page_mode": "none", "pages": None, "sheets": used}


# ============================================================ PDF
_BOLD_FONT = re.compile(r"bold|black|heavy|semibold|demi|cmbx|medi", re.I)
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S|^[IVX]+\.\s+\S|^[A-H](?:\.\d+)*\s+[A-Z]")


def _pdf_lines(page) -> list[dict]:
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            # вертикальный водяной знак arXiv и прочий повёрнутый текст пропускаем
            if abs(line.get("dir", (1, 0))[1]) > 0.5:
                continue
            spans = line["spans"]
            text = "".join(s["text"] for s in spans)  # пробелы бывают отдельными span'ами
            ink = [s for s in spans if s["text"].strip()]
            if not ink:
                continue
            lines.append({
                "text": text,
                "size": max(s["size"] for s in ink),
                "bold": all((s["flags"] & 16) or _BOLD_FONT.search(s["font"]) for s in ink),
            })
    return lines


def parse_pdf(path: Path, drop_references: bool = True) -> tuple[str, dict]:
    try:
        import pymupdf
    except ImportError:  # старые версии PyMuPDF
        import fitz as pymupdf

    doc = pymupdf.open(str(path))
    pages = [_pdf_lines(p) for p in doc]
    sizes = Counter()
    for lines in pages:
        for ln in lines:
            if len(ln["text"]) >= 40:  # длинные строки — основной текст, а не подписи/таблицы
                sizes[round(ln["size"])] += len(ln["text"])
    body = sizes.most_common(1)[0][0] if sizes else 10.0

    # колонтитулы: строки, повторяющиеся на большинстве страниц
    norm = lambda s: re.sub(r"\d+", "#", s.strip().lower())
    freq = Counter(k for lines in pages for k in {norm(ln["text"]) for ln in lines})
    repeated = {k for k, v in freq.items() if len(pages) > 3 and v > 0.5 * len(pages)}

    def heading_level(t: str, ln: dict, pno: int) -> int | None:
        words = t.split()
        if not words or len(words) > 14 or t.endswith((".", ",", ";", "-", ":")) or ln["size"] < body - 0.3:
            return None
        if not (t[0].isupper() or t[0].isdigit()):
            return None
        m = re.match(r"^(\d+(?:\.\d+)*)\.?\s", t)
        bigger = ln["size"] >= body + 1.0
        if m and (ln["bold"] or bigger):
            return min(1 + m.group(1).count(".") + 1, 6)
        if bigger:
            return 1 if (pno == 1 and ln["size"] >= body + 3) else 2
        if ln["bold"] and (len(words) <= 8 or _NUMBERED.match(t)):
            return 2
        return None

    out, n_headings, title = [], 0, None
    stop = False
    for pno, lines in enumerate(pages, 1):
        if stop:
            break
        out.append(PAGE_MARK.format(pno))
        para: list[str] = []
        prev_heading = None  # (уровень, кегль) предыдущей строки-заголовка

        def flush():
            if para:
                text = re.sub(r"(\w)- (\w)", r"\1\2", " ".join(para))  # переносы по слогам
                out.append(clean_text(text))
                para.clear()

        for ln in lines:
            t = clean_text(ln["text"])
            if (not t or ln["size"] < 0.8 * body or norm(t) in repeated
                    or re.fullmatch(r"\d{1,3}", t) or t.startswith("arXiv:")):
                continue
            level = heading_level(t, ln, pno)
            if (not level and prev_heading and abs(ln["size"] - prev_heading[1]) < 0.2
                    and len(t.split()) <= 14 and out and out[-1].startswith("#")):
                level = prev_heading[0]  # продолжение многострочного заголовка со строчной буквы
            if level:
                flush()
                if drop_references and re.fullmatch(r"(\d+\.?\s*)?(references|bibliography)", t.lower()):
                    stop = True
                    break
                if prev_heading == (level, ln["size"]) and out and out[-1].startswith("#"):
                    out[-1] += " " + t  # многострочный заголовок
                else:
                    out.append("#" * level + " " + t)
                    n_headings += 1
                if level == 1 and title is None:
                    title = out[-1].lstrip("# ")
                prev_heading = (level, ln["size"])
                continue
            prev_heading = None
            para.append(t)
            if t.endswith((".", "!", "?", ":")) and len(t) < 55:
                flush()  # короткая строка с точкой — вероятно, конец абзаца
        flush()
    title = next((x[2:].strip() for x in out if x.startswith("# ")), title)  # после склейки строк
    meta_title = clean_text(doc.metadata.get("title") or "")
    return join_blocks(out), {
        "format": "pdf", "page_mode": "exact", "pages": len(pages), "n_headings": n_headings,
        "pdf_title": meta_title if len(meta_title) > 15 and " " in meta_title else title}


# ============================================================ репозиторий
def _git_date(repo: Path, rel: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%cs", "--", rel],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def _split_code(text: str, lang: str) -> list[tuple[str, str]]:
    """Разбить код на сегменты по верхнеуровневым def/class (py) или ключам (yaml)."""
    if lang == "python":
        pat = re.compile(r"^(?:@\w|def |async def |class )")
        name_re = re.compile(r"^(?:async def|def|class)\s+(\w+)")
    elif lang == "yaml":
        pat = re.compile(r"^[A-Za-z_][\w-]*:")
        name_re = re.compile(r"^([A-Za-z_][\w-]*):")
    else:
        return [("", text)]
    segments, cur, cur_name = [], [], "module"
    for line in text.splitlines():
        if pat.match(line) and cur and not (cur and cur[-1].startswith("@")):
            segments.append((cur_name, "\n".join(cur)))
            cur = []
        if pat.match(line):
            m = name_re.match(line)
            if m:
                cur_name = m.group(1)
        cur.append(line)
    if cur:
        segments.append((cur_name, "\n".join(cur)))
    return [(n, s) for n, s in segments if s.strip()]


def parse_repo_file(path: Path, rel: str) -> tuple[str, dict]:
    raw = nfc(path.read_text(encoding="utf-8", errors="replace"))
    ext = path.suffix.lower()
    if ext == ".md":
        return raw.strip() + "\n", {"format": "md", "page_mode": "none"}
    lang = {".py": "python", ".yaml": "yaml", ".yml": "yaml", ".sh": "bash"}.get(ext, "")
    out = [f"# {rel}"]
    for name, seg in _split_code(raw, lang):
        if name:
            out.append(f"## {name}")
        out.append(f"```{lang}\n{seg.rstrip()}\n```")
    return "\n\n".join(out) + "\n", {"format": ext.lstrip("."), "page_mode": "none"}


def iter_repo_files(cfg: dict):
    repo = settings.LAB_DIR / cfg["corpus"]["repo_path"]
    if not repo.exists():
        print(f"  [warn] репозиторий не найден: {repo} — пропускаю")
        return
    inc, exc = cfg["corpus"]["repo_include"], cfg["corpus"].get("repo_exclude", [])
    for p in sorted(repo.rglob("*")):
        if not p.is_file() or ".git" in p.parts:
            continue
        rel = p.relative_to(repo).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in inc) and not any(fnmatch.fnmatch(rel, pat) for pat in exc):
            yield repo, p, rel


# ============================================================ оркестрация
def write_doc(md_dir: Path, meta: dict, body: str) -> dict:
    front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    (md_dir / f"{meta['doc_id']}.md").write_text(f"---\n{front}\n---\n\n{body}", encoding="utf-8")
    words = len(re.findall(r"\w+", body))
    return {**meta, "words": words}


def parse_all(cfg: dict) -> list[dict]:
    raw_dir = settings.lab_path(cfg, "raw_dir")
    md_dir = settings.lab_path(cfg, "markdown_dir")
    md_dir.mkdir(parents=True, exist_ok=True)
    for old in md_dir.glob("*.md"):  # пересборка с нуля — идемпотентно
        old.unlink()
    skip = cfg["corpus"].get("skip_names", [])
    repo_dir = (settings.LAB_DIR / cfg["corpus"]["repo_path"]).resolve()
    xlsx_cfg = {nfc(k): v for k, v in cfg["corpus"].get("xlsx_sheets", {}).items()}
    manifest = []

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or any(s in path.name for s in skip):
            continue
        if repo_dir in path.resolve().parents:
            continue  # репозиторий обрабатывается отдельно
        rel = nfc(path.relative_to(raw_dir).as_posix())
        ext = path.suffix.lower()
        try:
            if ext == ".docx":
                body, meta = parse_docx(path)
                stype = "nir_document"
            elif ext == ".xlsx":
                body, meta = parse_xlsx(path, xlsx_cfg.get(nfc(path.name)))
                stype = "nir_plan"
            elif ext == ".pdf":
                body, meta = parse_pdf(path, cfg["corpus"].get("pdf_drop_references", True))
                stype = "article"
            elif ext in (".md", ".txt"):
                body, meta = path.read_text(encoding="utf-8"), {"format": ext[1:], "page_mode": "none"}
                stype = "note"
            else:
                print(f"  [skip] {rel}: формат не поддерживается")
                continue
        except Exception as e:
            print(f"  [error] {rel}: {e!r}")
            continue
        title = meta.pop("pdf_title", None) or nfc(path.stem)
        semester = semester_of(rel)
        full_meta = {
            "doc_id": doc_id_for(rel),
            "title": (f"Семестр {semester} / {title}" if semester and stype != "article" else title),
            "source_type": stype,
            "path": f"data/raw/{rel}",
            "date": meta.pop("doc_date", None) or file_date(path),
            "semester": semester,
            **meta,
        }
        manifest.append(write_doc(md_dir, full_meta, body))
        print(f"  [ok] {rel}: {manifest[-1]['words']} слов, page_mode={full_meta['page_mode']}")

    for repo, path, rel in iter_repo_files(cfg) or []:
        body, meta = parse_repo_file(path, rel)
        full_meta = {
            "doc_id": "repo__" + doc_id_for(rel),
            "title": f"itmo-research/{rel}",
            "source_type": "repo_doc" if meta["format"] == "md" else "repo_code",
            "path": f"{cfg['corpus']['repo_path']}/{rel}",
            "url": f"https://github.com/alinocco/itmo-research/blob/main/{rel}",
            "date": _git_date(repo, rel) or file_date(path),
            "semester": None,
            **meta,
        }
        manifest.append(write_doc(md_dir, full_meta, body))
    n_repo = sum(1 for m in manifest if m["source_type"].startswith("repo"))
    print(f"  [ok] репозиторий: {n_repo} файлов")

    total = sum(m["words"] for m in manifest)
    summary = {
        "documents": len(manifest),
        "total_words": total,
        "approx_pages": round(total / 300),  # ~300 слов на страницу 14 pt, 1.5 интервала
        "by_type": dict(Counter(m["source_type"] for m in manifest)),
        "docs": manifest,
    }
    (md_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Итого: {len(manifest)} документов, {total} слов (≈{summary['approx_pages']} стр.) → {md_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    settings.add_common_args(parser)
    args = parser.parse_args()
    parse_all(settings.load_config(args.config, args.set))


if __name__ == "__main__":
    main()
