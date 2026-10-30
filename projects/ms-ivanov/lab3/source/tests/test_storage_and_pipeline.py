import json

from source.bibliography.models import SourceKind
from source.cli import build_parser
from source.pipeline import run_pipeline, select_sources
from source.report import format_report
from source.storage import EntryRecord, EntryStore, FileRecord
from source.tests.conftest import make_source


def _file(name="html", path="paper.html"):
    return FileRecord(name, path, "https://x", 10, "ab" * 32, "text/html")


def test_record_updates_are_immutable():
    original = EntryRecord(source=make_source())

    updated = original.with_file(_file()).with_missing("pdf", "404")

    assert original.files == {} and original.missing == {}
    assert updated.status == "ok" and updated.missing == {"pdf": "404"}


def test_new_outcome_for_a_name_replaces_the_old_one():
    record = EntryRecord(source=make_source()).with_missing("html", "404")

    record = record.with_file(_file())

    assert record.missing == {}


def test_status_reflects_files_and_errors():
    empty = EntryRecord(source=make_source())

    assert empty.status == "unavailable"
    assert empty.with_error("html", "timeout").status == "error"
    assert empty.with_file(_file()).status == "ok"


def test_primary_prefers_html_over_pdf():
    record = EntryRecord(source=make_source()).with_file(_file("pdf", "paper.pdf"))

    assert record.primary.name == "pdf"
    assert record.with_file(_file()).primary.name == "html"


def test_store_roundtrip(tmp_path):
    source = make_source()
    store = EntryStore(tmp_path)
    record = EntryRecord(source=source).with_file(_file()).with_missing("pdf", "n/a")

    store.save(record)
    loaded = store.load(source)

    assert loaded.files == record.files and loaded.missing == record.missing
    assert store.entry_dir(source) == tmp_path / "arxiv" / "key2020"


def test_store_ignores_corrupt_metadata(tmp_path):
    source = make_source()
    store = EntryStore(tmp_path)
    store.entry_dir(source).mkdir(parents=True)
    (store.entry_dir(source) / "meta.json").write_text("{broken")

    assert store.load(source) is None


def test_manifest_uses_paths_relative_to_corpus_root(tmp_path):
    store = EntryStore(tmp_path)
    record = EntryRecord(source=make_source()).with_file(_file())

    path = store.write_manifest([record])

    entry = json.loads(path.read_text())["entries"][0]
    assert entry["files"]["html"]["path"] == "arxiv/key2020/paper.html"
    assert entry["primary"] == "html"


def test_select_sources_orders_by_kind_then_priority():
    sources = [
        make_source(key="a_low", priority=1),
        make_source(key="a_core", priority=3),
        make_source(SourceKind.GITHUB, key="g"),
        make_source(SourceKind.WEB, key="w"),
    ]

    ordered = [s.key for s in select_sources(sources)]

    assert ordered == ["g", "w", "a_core", "a_low"]


def test_select_sources_filters():
    sources = [make_source(key="a"), make_source(SourceKind.WEB, key="w")]

    assert [s.key for s in select_sources(sources, kinds=[SourceKind.WEB])] == ["w"]
    assert [s.key for s in select_sources(sources, keys=["a"])] == ["a"]
    assert len(select_sources(sources, limit=1)) == 1


class _StubDownloader:
    def __init__(self, fail_for=()):
        self.fail_for = fail_for

    def download(self, source, record, entry_dir):
        if source.key in self.fail_for:
            raise RuntimeError("boom")
        return record.with_file(_file())


def test_pipeline_continues_after_unexpected_failure(tmp_path):
    sources = [make_source(key="bad"), make_source(key="good")]
    store = EntryStore(tmp_path)

    records = run_pipeline(sources, store, {SourceKind.ARXIV: _StubDownloader({"bad"})})

    assert [r.status for r in records] == ["error", "ok"]
    assert "RuntimeError" in records[0].errors["unexpected"]
    assert store.load(sources[0]).status == "error"


def test_pipeline_keeps_timestamp_when_nothing_changed(tmp_path):
    source = make_source()
    store = EntryStore(tmp_path)
    downloaders = {SourceKind.ARXIV: _StubDownloader()}

    first = run_pipeline([source], store, downloaders)[0]
    second = run_pipeline([source], store, downloaders)[0]

    assert first.updated_at == second.updated_at


def test_report_lists_entries_without_files():
    ok = EntryRecord(source=make_source(key="ok")).with_file(_file())
    bad = EntryRecord(source=make_source(SourceKind.DOI, key="bad")).with_missing(
        "pdf", "paywalled"
    )

    text = format_report([ok, bad], expected_total=3)

    assert "2 / 3" in text and "[doi] bad - pdf: paywalled" in text


def test_cli_parser_defaults():
    args = build_parser().parse_args(["download", "--kinds", "web", "--pdf", "always"])

    assert args.kinds == [SourceKind.WEB] and args.pdf.value == "always"
    assert not args.force


def test_report_summarises_conversion_status():
    done = EntryRecord(source=make_source(key="a")).with_file(_file())
    done = done.with_conversion({"status": "ok", "converter": "latexml-html@1"})
    failed = EntryRecord(source=make_source(key="b")).with_file(_file())
    failed = failed.with_conversion(
        {"status": "error", "converter": "x", "error": "boom"}
    )
    pending = EntryRecord(source=make_source(key="c")).with_file(_file())

    text = format_report([done, failed, pending])

    assert "ok / latexml-html@1" in text and "not converted / -" in text
    assert "b: boom" in text
