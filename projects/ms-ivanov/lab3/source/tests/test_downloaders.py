from source.bibliography.models import SourceKind
from source.downloaders import PdfPolicy, build_downloaders
from source.downloaders.doi import (
    pdf_candidates,
    semantic_scholar_candidates,
    unique_pdf_urls,
)
from source.downloaders.github import parse_repository
from source.downloaders.web import find_pdf_link
from source.net import FetchError
from source.storage import EntryRecord
from source.tests.conftest import HTML_BYTES, PDF_BYTES, make_source

AR5IV = "https://ar5iv.labs.arxiv.org/html/2001.00001"
ARXIV_HTML = "https://arxiv.org/html/2001.00001"
ARXIV_PDF = "https://arxiv.org/pdf/2001.00001"


def _run(kind, context, source, tmp_path, record=None):
    downloader = build_downloaders(context)[kind]
    return downloader.download(source, record or EntryRecord(source=source), tmp_path)


def test_arxiv_prefers_ar5iv_html_and_skips_pdf(make_context, tmp_path):
    context, client = make_context({AR5IV: HTML_BYTES, ARXIV_PDF: PDF_BYTES})

    record = _run(SourceKind.ARXIV, context, make_source(), tmp_path)

    assert record.status == "ok" and record.primary.name == "html"
    assert (tmp_path / "paper.html").read_bytes() == HTML_BYTES
    assert client.requested == [AR5IV]


def test_arxiv_falls_back_to_main_site_html(make_context, tmp_path):
    context, client = make_context({ARXIV_HTML: HTML_BYTES})

    record = _run(SourceKind.ARXIV, context, make_source(), tmp_path)

    assert record.files["html"].url == ARXIV_HTML
    assert client.requested == [AR5IV, ARXIV_HTML]


def test_arxiv_downloads_pdf_when_html_is_missing(make_context, tmp_path):
    context, _ = make_context({ARXIV_PDF: PDF_BYTES})

    record = _run(SourceKind.ARXIV, context, make_source(), tmp_path)

    assert set(record.files) == {"pdf"}
    assert "html" in record.missing
    assert record.primary.name == "pdf"


def test_arxiv_pdf_policies(make_context, tmp_path):
    responses = {AR5IV: HTML_BYTES, ARXIV_PDF: PDF_BYTES}
    always, _ = make_context(responses, pdf_policy=PdfPolicy.ALWAYS)
    never, _ = make_context({ARXIV_PDF: PDF_BYTES}, pdf_policy=PdfPolicy.NEVER)

    both = _run(SourceKind.ARXIV, always, make_source(), tmp_path / "a")
    none = _run(SourceKind.ARXIV, never, make_source(), tmp_path / "b")

    assert set(both.files) == {"html", "pdf"}
    assert none.status == "unavailable"


def test_error_page_served_with_200_is_not_stored(make_context, tmp_path):
    context, _ = make_context({AR5IV: b"<html>HTML is not available</html>"})

    record = _run(SourceKind.ARXIV, context, make_source(), tmp_path)

    assert "html" not in record.files
    assert not (tmp_path / "paper.html").exists()


def test_rerun_skips_existing_files_and_known_missing(make_context, tmp_path):
    context, client = make_context({AR5IV: HTML_BYTES})
    first = _run(SourceKind.ARXIV, context, make_source(), tmp_path)
    client.requested.clear()

    second = _run(SourceKind.ARXIV, context, make_source(), tmp_path, first)

    assert client.requested == []
    assert second.files == first.files


def test_known_missing_is_retried_only_on_request(make_context, tmp_path):
    source = make_source()
    record = EntryRecord(source=source).with_missing("html", "404")
    skipping, skip_client = make_context({}, pdf_policy=PdfPolicy.NEVER)
    retrying, retry_client = make_context(
        {}, pdf_policy=PdfPolicy.NEVER, retry_missing=True
    )

    _run(SourceKind.ARXIV, skipping, source, tmp_path, record)
    _run(SourceKind.ARXIV, retrying, source, tmp_path, record)

    assert skip_client.requested == []
    assert retry_client.requested == [AR5IV, ARXIV_HTML]


def test_transient_failure_is_recorded_as_error_not_missing(make_context, tmp_path):
    timeout = FetchError("timeout", permanent=False)
    context, _ = make_context({AR5IV: timeout, ARXIV_HTML: timeout, ARXIV_PDF: timeout})

    record = _run(SourceKind.ARXIV, context, make_source(), tmp_path)

    assert record.status == "error"
    assert "html" in record.errors and "html" not in record.missing


def test_file_placed_by_hand_is_adopted_without_network(make_context, tmp_path):
    (tmp_path / "paper.pdf").write_bytes(PDF_BYTES)
    source = make_source(SourceKind.DOI, doi="10.1/x")
    context, client = make_context({})

    record = _run(SourceKind.DOI, context, source, tmp_path)

    assert record.files["pdf"].url == "manual"
    assert client.requested == []


def test_force_redownloads_existing_file(make_context, tmp_path):
    (tmp_path / "paper.html").write_bytes(b"old")
    context, _ = make_context({AR5IV: HTML_BYTES}, force=True)

    _run(SourceKind.ARXIV, context, make_source(), tmp_path)

    assert (tmp_path / "paper.html").read_bytes() == HTML_BYTES


def test_doi_downloads_first_valid_open_access_pdf(make_context, tmp_path):
    work = {
        "best_oa_location": {"is_oa": True, "pdf_url": "https://oa.test/landing"},
        "locations": [{"is_oa": True, "pdf_url": "https://oa.test/real.pdf"}],
    }
    context, client = make_context(
        {
            "https://api.openalex.org/works/doi:10.1000/xyz": work,
            "https://oa.test/landing": b"<html>no</html>",
            "https://oa.test/real.pdf": PDF_BYTES,
        }
    )

    record = _run(SourceKind.DOI, context, make_source(SourceKind.DOI), tmp_path)

    assert record.files["pdf"].url == "https://oa.test/real.pdf"
    assert client.requested[2:] == [
        "https://oa.test/landing",
        "https://oa.test/real.pdf",
    ]


def test_doi_falls_back_to_semantic_scholar_and_its_arxiv_copy(make_context, tmp_path):
    scholar = {"openAccessPdf": {"url": ""}, "externalIds": {"ArXiv": "1234.5678"}}
    context, _ = make_context(
        {
            "https://api.openalex.org/works/doi:10.1000/xyz": FetchError(
                "HTTP 404", permanent=True, status=404
            ),
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1000/xyz"
            "?fields=openAccessPdf,externalIds": scholar,
            "https://arxiv.org/pdf/1234.5678": PDF_BYTES,
        }
    )

    record = _run(SourceKind.DOI, context, make_source(SourceKind.DOI), tmp_path)

    assert record.files["pdf"].url == "https://arxiv.org/pdf/1234.5678"


def test_doi_lookup_outage_is_transient(make_context, tmp_path):
    outage = FetchError("HTTP 503", permanent=False, status=503)
    context, _ = make_context(
        {
            "https://api.openalex.org/works/doi:10.1000/xyz": outage,
            "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1000/xyz"
            "?fields=openAccessPdf,externalIds": outage,
        }
    )

    record = _run(SourceKind.DOI, context, make_source(SourceKind.DOI), tmp_path)

    assert record.status == "error"


def test_doi_without_open_access_copy_is_unavailable(make_context, tmp_path):
    context, _ = make_context(
        {"https://api.openalex.org/works/doi:10.1000/xyz": {"locations": []}}
    )

    record = _run(SourceKind.DOI, context, make_source(SourceKind.DOI), tmp_path)

    assert record.status == "unavailable"
    assert "no open-access PDF found" in record.missing["pdf"]


def test_pdf_candidates_keep_open_access_locations_only():
    work = {
        "best_oa_location": {"is_oa": True, "pdf_url": "https://a/1.pdf"},
        "locations": [
            {"is_oa": True, "pdf_url": "https://a/1.pdf"},
            {"is_oa": False, "pdf_url": "https://a/closed.pdf"},
            {"is_oa": True, "pdf_url": None},
            None,
        ],
    }

    assert pdf_candidates(work) == ["https://a/1.pdf", "https://a/1.pdf"]


def test_semantic_scholar_candidates_handle_missing_fields():
    assert semantic_scholar_candidates({}) == []
    assert semantic_scholar_candidates(
        {"openAccessPdf": {"url": "https://a/x.pdf"}}
    ) == ["https://a/x.pdf"]


def test_unique_pdf_urls_rewrites_hal_landing_pages_and_dedupes():
    urls = [
        "https://inria.hal.science/inria-00383685",
        "https://a/x.pdf",
        "https://a/x.pdf",
    ]

    assert unique_pdf_urls(urls) == [
        "https://inria.hal.science/inria-00383685/document",
        "https://a/x.pdf",
    ]


def test_github_downloads_readme(make_context, tmp_path):
    api = "https://api.github.com/repos/owner/repo/readme"
    context, _ = make_context({api: b"# Title"})
    source = make_source(SourceKind.GITHUB, url="https://github.com/owner/repo")

    record = _run(SourceKind.GITHUB, context, source, tmp_path)

    assert (tmp_path / "README.md").read_text() == "# Title"
    assert record.primary.name == "readme"


def test_github_unsupported_url_is_reported(make_context, tmp_path):
    context, client = make_context({})
    source = make_source(SourceKind.GITHUB, url="https://github.com/owner")

    record = _run(SourceKind.GITHUB, context, source, tmp_path)

    assert record.status == "unavailable" and client.requested == []


def test_parse_repository_variants():
    assert parse_repository("https://github.com/o/r") == ("o", "r")
    assert parse_repository("https://github.com/o/r.git/") == ("o", "r")
    assert parse_repository("https://github.com/o/r/tree/main") == ("o", "r")
    assert parse_repository("https://example.com/o/r") is None


def test_web_page_with_advertised_pdf_gets_both(make_context, tmp_path):
    page = (
        b"<html><head><meta name='citation_pdf_url' content='/p/paper.pdf'>"
        b"</head></html>"
    )
    context, _ = make_context(
        {"https://j.test/abs": page, "https://j.test/p/paper.pdf": PDF_BYTES}
    )
    source = make_source(SourceKind.WEB, url="https://j.test/abs")

    record = _run(SourceKind.WEB, context, source, tmp_path)

    assert set(record.files) == {"page", "pdf"}
    assert record.primary.name == "pdf"


def test_web_direct_pdf_url(make_context, tmp_path):
    context, _ = make_context({"https://j.test/a.pdf": PDF_BYTES})
    source = make_source(SourceKind.WEB, url="https://j.test/a.pdf")

    record = _run(SourceKind.WEB, context, source, tmp_path)

    assert set(record.files) == {"pdf"}


def test_find_pdf_link_resolves_relative_urls_and_handles_absence():
    html = "<meta name='citation_pdf_url' content='../a.pdf'>"

    assert find_pdf_link(html, "https://j.test/x/y.html") == "https://j.test/a.pdf"
    assert find_pdf_link("<html></html>", "https://j.test/") is None
