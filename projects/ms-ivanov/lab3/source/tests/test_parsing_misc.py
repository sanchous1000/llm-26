import shutil
from datetime import UTC, datetime
from pathlib import Path

import pymupdf
import pytest
import yaml

from source.bibliography.models import SourceKind
from source.conversion_pipeline import DOCUMENT_FILENAME, run_conversion
from source.parsing.frontmatter import build_front_matter
from source.parsing.models import markdown_stats
from source.parsing.pdf import PdfConverter
from source.parsing.placeholders import PlaceholderStore
from source.parsing.postprocess import tidy
from source.parsing.readme import ReadmeConverter
from source.parsing.registry import build_converter
from source.parsing.webpage import WebPageConverter
from source.storage import EntryRecord, EntryStore, FileRecord
from source.tests.conftest import make_source

needs_pandoc = pytest.mark.skipif(
    shutil.which("pandoc") is None, reason="pandoc missing"
)


def test_placeholders_restore_math_blocks_and_code():
    store = PlaceholderStore()
    text = (
        f"a {store.inline_math('x_1')} b {store.inline_math('|x|', in_table=True)}\n\n"
        f"{store.display_math('y')}\n\n{store.code_block('```inner```')}"
    )

    restored = store.restore(text)

    assert "$x_1$" in restored and "$\\vert x\\vert $" in restored
    assert "\n$$\ny\n$$\n" in restored
    assert "````\n```inner```\n````" in restored


def test_tidy_cleans_whitespace_and_needless_escapes_but_not_code_or_math():
    raw = (
        "Cost\u00a0is \\<0.5 and \\[12\\] and \u200bok  \n\n\n\n"
        "1.  item\n\n```\n\\<keep\\> \\[x\\]\n```\n\n$$\n\\[a\\]\n$$\n\n$\\[b\\]$ \\[c\\]\n"
    )

    text = tidy(raw)

    assert "Cost is <0.5 and [12] and ok\n\n1. item" in text
    assert "\\<keep\\> \\[x\\]" in text
    assert "\\[a\\]" in text and "$\\[b\\]$ [c]" in text


def test_tidy_fixes_double_escaped_entities():
    assert tidy("*Discrete \\&amp; Geometry* &lt;ok\n") == "*Discrete & Geometry* <ok\n"


def test_tidy_does_not_touch_adjacent_emphasis():
    assert "*a* **b**" in tidy("*a* **b**")


def test_markdown_stats_counts_structure():
    markdown = "# T\n\n## S\n\n| a |\n|---|\n| 1 |\n\n$$\nx\n$$\n\n![f](u)\n"

    stats = markdown_stats(markdown)

    assert (
        stats["headings"],
        stats["tables"],
        stats["display_equations"],
        stats["images"],
    ) == (2, 1, 1, 1)


def test_front_matter_is_valid_yaml_without_empty_fields():
    source = make_source(title="Colons: and 'quotes'", priority=3, tags=("OT",))
    original = FileRecord("html", "paper.html", "https://x", 1, "ab", "text/html")

    block = build_front_matter(
        source, original, "arxiv/key2020", "c@1", datetime(2026, 1, 1, tzinfo=UTC)
    )

    assert block.startswith("---\n") and block.endswith("---\n\n")
    data = yaml.safe_load(block.strip().strip("-"))
    assert data["title"] == "Colons: and 'quotes'"
    assert data["original_file"] == "arxiv/key2020/paper.html"
    assert "doi" not in data and data["tags"] == ["OT"]


def test_readme_drops_badges_and_comments(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(
        "# Tool\n<!-- hidden -->\n[![CI](https://img.shields.io/x.svg)](https://ci)\n\nUse it.\n"
    )

    markdown = ReadmeConverter().convert(path, "").markdown

    assert markdown == "# Tool\n\nUse it.\n"


@needs_pandoc
def test_rst_readme_is_converted_with_pandoc(tmp_path):
    path = tmp_path / "README.rst"
    path.write_text("Title\n=====\n\nSome *text*.\n")

    markdown = ReadmeConverter().convert(path, "").markdown

    assert markdown.startswith("# Title")


def test_pdf_converter_keeps_page_markers(tmp_path):
    path = tmp_path / "doc.pdf"
    document = pymupdf.open()
    for number in (1, 2):
        page = document.new_page()
        for line in range(12):
            text = f"Page number {number}: this is body text line {line} of the page."
            page.insert_text((72, 72 + 18 * line), text, fontsize=11)
    document.save(path)

    result = PdfConverter().convert(path, "")

    assert (
        "<!-- page: 1 -->" in result.markdown and "<!-- page: 2 -->" in result.markdown
    )
    assert "Page number 2" in result.markdown


def test_pdf_converter_rejects_garbage(tmp_path):
    from source.parsing.errors import ConversionError

    path = tmp_path / "bad.pdf"
    path.write_bytes(b"not a pdf")

    with pytest.raises(ConversionError):
        PdfConverter().convert(path, "")


def test_web_page_converter_extracts_main_content(tmp_path):
    path = tmp_path / "page.html"
    body = "".join(
        f"<p>Paragraph {i} about optimal transport and its many uses.</p>"
        for i in range(8)
    )
    path.write_text(
        f"<html><body><nav>menu menu</nav><article><h1>Course</h1>{body}</article></body></html>"
    )

    markdown = WebPageConverter().convert(path, "https://x.test/").markdown

    assert "Paragraph 3" in markdown and "menu menu" not in markdown


def test_registry_maps_file_roles_to_converters():
    assert build_converter("pdf").name.startswith("pymupdf4llm")
    assert build_converter("readme").name.startswith("readme")
    assert build_converter("html").name.startswith("latexml")


def _readme_record(store: EntryStore, text: str) -> EntryRecord:
    source = make_source(
        SourceKind.GITHUB, key="repo", title="Repo", url="https://github.com/a/b"
    )
    directory = store.entry_dir(source)
    directory.mkdir(parents=True)
    (directory / "README.md").write_text(text)
    file = FileRecord(
        "readme", "README.md", "https://api", len(text), "sha-1", "text/plain"
    )
    return EntryRecord(source=source).with_file(file)


def test_conversion_pipeline_writes_document_and_skips_when_up_to_date(tmp_path):
    store = EntryStore(tmp_path)
    record = _readme_record(store, "# Repo\n\nHello.\n")

    first = run_conversion([record], store, workers=1)[0]
    document = store.entry_dir(record.source) / DOCUMENT_FILENAME
    stamp = document.stat().st_mtime_ns
    second = run_conversion([first], store, workers=1)[0]

    assert first.conversion["status"] == "ok" and first.conversion["headings"] == 1
    assert document.read_text().startswith("---\nkey: repo\n")
    assert (
        document.stat().st_mtime_ns == stamp and second.conversion == first.conversion
    )
    assert store.load(record.source).conversion["status"] == "ok"


def test_conversion_pipeline_reconverts_when_source_changes_or_forced(tmp_path):
    store = EntryStore(tmp_path)
    record = _readme_record(store, "# Repo\n")
    done = run_conversion([record], store, workers=1)[0]
    changed = done.with_file(
        FileRecord("readme", "README.md", "u", 1, "other-sha", "t")
    )

    redone = run_conversion([changed], store, workers=1)[0]
    forced = run_conversion([redone], store, workers=1, force=True)[0]

    assert redone.conversion["source_sha256"] == "other-sha"
    assert forced.conversion["converted_at"] >= redone.conversion["converted_at"]


def test_conversion_failure_is_recorded_and_other_entries_continue(tmp_path):
    store = EntryStore(tmp_path)
    good = _readme_record(store, "# Fine\n")
    bad_source = make_source(key="broken", title="Broken")
    bad_dir = store.entry_dir(bad_source)
    bad_dir.mkdir(parents=True)
    (bad_dir / "paper.pdf").write_bytes(b"garbage")
    bad = EntryRecord(source=bad_source).with_file(
        FileRecord("pdf", "paper.pdf", "u", 7, "sha", "application/pdf")
    )

    results = run_conversion([bad, good], store, workers=1)

    assert results[0].conversion["status"] == "error"
    assert results[1].conversion["status"] == "ok"


def test_entries_without_files_are_skipped(tmp_path):
    record = EntryRecord(source=make_source(SourceKind.DOI))

    result = run_conversion([record], EntryStore(tmp_path), workers=1)[0]

    assert result.conversion == {}


def test_manifest_links_markdown_for_converted_entries(tmp_path):
    store = EntryStore(tmp_path)
    record = run_conversion([_readme_record(store, "# R\n")], store, workers=1)[0]

    path = store.write_manifest([record])

    entry = __import__("json").loads(Path(path).read_text())["entries"][0]
    assert entry["markdown"] == "github/repo/document.md"
