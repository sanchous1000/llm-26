"""Extra sources (rag_additions.bib / RAG_ADDITIONS.md) next to the main catalog."""

from source.bibliography import SourceKind, load_sources
from source.bibliography.additions_parser import parse_additions_table
from source.bibliography.bib_parser import parse_bib_file
from source.tests.conftest import (
    PDF_BYTES,
    SAMPLE_BIB,
    SAMPLE_CATALOG,
    make_source,
)

ADDITIONS_BIB = """\
@article{fed1959,
  title = {Curvature measures},
  author = {Herbert Federer},
  year = {1959},
  url = {https://doi.org/10.2307/1993504},
  doi = {10.2307/1993504},
  journal = {Trans. AMS},
  note = {Open access: https://www.ams.org/journals/tran/1959/fed.pdf (open access)}
}

@misc{book2005,
  title = {A Book},
  author = {A. Author},
  year = {2005},
  url = {https://doi.org/10.1/book},
  doi = {10.1/book},
  note = {no open copy known}
}

@article{carlsson2009topology,
  title = {Topology and data},
  author = {Gunnar Carlsson},
  year = {2009},
  url = {https://doi.org/10.1090/S0273-0979-09-01249-X},
  doi = {10.1090/S0273-0979-09-01249-X},
  journal = {Bulletin of the AMS 46},
  note = {Open access: https://www.ams.org/journals/bull/2009/carlsson.pdf}
}
"""

ADDITIONS_TABLE = """\
# Documents to add

| Group | Priority | Key | Title | Year | Why it is needed | Open access |
|---|---|---|---|---|---|---|
| B | ★★★ | `fed1959` | Curvature measures | 1959 | Federer: sets of positive reach. | https://x.test/a.pdf |
| B | ★★ | `book2005` | A Book | 2005 | A standard text. | none |
| A | ★★★ | `carlsson2009topology` | Topology and data | 2009 | Survey. | https://x.test/c.pdf |
"""


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_bib_note_with_open_access_url_is_extracted(tmp_path):
    sources = {
        s.key: s for s in parse_bib_file(_write(tmp_path, "a.bib", ADDITIONS_BIB))
    }

    assert sources["fed1959"].open_access_url == (
        "https://www.ams.org/journals/tran/1959/fed.pdf"
    )
    assert sources["book2005"].open_access_url is None


def test_additions_table_gives_priority_and_summary(tmp_path):
    table = parse_additions_table(_write(tmp_path, "A.md", ADDITIONS_TABLE))

    assert table["fed1959"].priority == 3 and table["book2005"].priority == 2
    assert table["carlsson2009topology"].priority == 3
    assert table["fed1959"].summary == "Federer: sets of positive reach."
    assert "group B" in table["fed1959"].section


def test_additions_next_to_the_main_files_are_merged(tmp_path):
    bib = _write(tmp_path, "sources.bib", SAMPLE_BIB)
    catalog = _write(tmp_path, "SOURCES.md", SAMPLE_CATALOG)
    _write(tmp_path, "rag_additions.bib", ADDITIONS_BIB)
    _write(tmp_path, "RAG_ADDITIONS.md", ADDITIONS_TABLE)

    sources = {s.key: s for s in load_sources(bib, catalog)}

    assert {"fed1959", "book2005"} <= set(sources)
    assert sources["fed1959"].priority == 3
    assert sources["fed1959"].kind is SourceKind.DOI


def test_duplicate_key_keeps_the_main_entry_and_gains_the_open_access_url(tmp_path):
    bib = _write(tmp_path, "sources.bib", SAMPLE_BIB)
    catalog = _write(tmp_path, "SOURCES.md", SAMPLE_CATALOG)
    _write(tmp_path, "rag_additions.bib", ADDITIONS_BIB)
    _write(tmp_path, "RAG_ADDITIONS.md", ADDITIONS_TABLE)

    sources = load_sources(bib, catalog)
    carlsson = [s for s in sources if s.key == "carlsson2009topology"]

    assert len(carlsson) == 1
    assert (
        carlsson[0].open_access_url
        == "https://www.ams.org/journals/bull/2009/carlsson.pdf"
    )
    assert carlsson[0].priority == 1  # from the main catalog, not the additions table


def test_loading_without_additions_files_is_unchanged(tmp_path):
    bib = _write(tmp_path, "sources.bib", SAMPLE_BIB)

    assert {s.key for s in load_sources(bib)} >= {"peyre2018computational"}


def test_doi_downloader_tries_the_listed_open_access_url_first(make_context, tmp_path):
    from source.downloaders import build_downloaders
    from source.storage import EntryRecord

    source = make_source(SourceKind.DOI, open_access_url="https://oa.test/paper.pdf")
    context, client = make_context({"https://oa.test/paper.pdf": PDF_BYTES})

    record = build_downloaders(context)[SourceKind.DOI].download(
        source, EntryRecord(source=source), tmp_path
    )

    assert record.files["pdf"].url == "https://oa.test/paper.pdf"
    assert client.requested == ["https://oa.test/paper.pdf"]


def test_arxiv_abstract_link_becomes_a_pdf_link(make_context, tmp_path):
    from source.downloaders import build_downloaders
    from source.storage import EntryRecord

    source = make_source(
        SourceKind.DOI, open_access_url="https://arxiv.org/abs/1312.2128"
    )
    context, client = make_context({"https://arxiv.org/pdf/1312.2128": PDF_BYTES})

    record = build_downloaders(context)[SourceKind.DOI].download(
        source, EntryRecord(source=source), tmp_path
    )

    assert record.files["pdf"].url == "https://arxiv.org/pdf/1312.2128"


def test_entry_without_doi_but_with_open_access_url_is_still_fetched(
    make_context, tmp_path
):
    from source.downloaders import build_downloaders
    from source.storage import EntryRecord

    source = make_source(
        SourceKind.DOI, doi=None, open_access_url="https://oa.test/b.pdf"
    )
    context, _ = make_context({"https://oa.test/b.pdf": PDF_BYTES})

    record = build_downloaders(context)[SourceKind.DOI].download(
        source, EntryRecord(source=source), tmp_path
    )

    assert record.status == "ok"


def test_listed_landing_page_is_followed_to_its_advertised_pdf(make_context, tmp_path):
    from source.downloaders import build_downloaders
    from source.storage import EntryRecord

    landing = (
        b'<html><head><meta name="citation_pdf_url" '
        b'content="https://oa.test/files/p.pdf"></head><body>'
        + b"x" * 600
        + b"</body></html>"
    )
    source = make_source(SourceKind.DOI, open_access_url="https://oa.test/abs/1")
    context, client = make_context(
        {"https://oa.test/abs/1": landing, "https://oa.test/files/p.pdf": PDF_BYTES}
    )

    record = build_downloaders(context)[SourceKind.DOI].download(
        source, EntryRecord(source=source), tmp_path
    )

    assert record.files["pdf"].url == "https://oa.test/files/p.pdf"
    assert not list(tmp_path.glob("*.tmp*"))


def test_landing_page_without_pdf_link_leaves_the_entry_unavailable(
    make_context, tmp_path
):
    from source.downloaders import build_downloaders
    from source.storage import EntryRecord

    source = make_source(SourceKind.DOI, doi=None, open_access_url="https://oa.test/a")
    context, _ = make_context(
        {"https://oa.test/a": b"<html>" + b"y" * 600 + b"</html>"}
    )

    record = build_downloaders(context)[SourceKind.DOI].download(
        source, EntryRecord(source=source), tmp_path
    )

    assert record.status == "unavailable" and not list(tmp_path.glob("*.tmp*"))


def test_several_additions_files_are_merged_with_their_own_tables(tmp_path):
    bib = _write(tmp_path, "sources.bib", SAMPLE_BIB)
    catalog = _write(tmp_path, "SOURCES.md", SAMPLE_CATALOG)
    _write(tmp_path, "rag_additions.bib", ADDITIONS_BIB)
    _write(tmp_path, "RAG_ADDITIONS.md", ADDITIONS_TABLE)
    _write(
        tmp_path,
        "rag_additions_attention.bib",
        "@misc{sink2021,\n  title = {Sinkformers},\n  author = {M. Sander},\n  year = {2021},\n"
        "  eprint = {2110.11773},\n  url = {https://arxiv.org/abs/2110.11773}\n}\n",
    )
    _write(
        tmp_path,
        "RAG_ADDITIONS_attention.md",
        "| B | ★★★ | `sink2021` | Sinkformers | 2021 | Doubly stochastic attention. | https://arxiv.org/abs/2110.11773 |\n",
    )

    sources = {s.key: s for s in load_sources(bib, catalog)}

    assert {"fed1959", "sink2021"} <= set(sources)
    assert sources["sink2021"].priority == 3
    assert sources["sink2021"].kind is SourceKind.ARXIV
    assert sources["fed1959"].priority == 3  # first file keeps its own table


def test_additions_file_without_a_table_is_still_loaded(tmp_path):
    bib = _write(tmp_path, "sources.bib", SAMPLE_BIB)
    _write(tmp_path, "rag_additions_misc.bib", ADDITIONS_BIB)

    sources = {s.key: s for s in load_sources(bib)}

    assert "fed1959" in sources and sources["fed1959"].priority is None
