from source.bibliography import SourceKind, load_sources
from source.bibliography.bib_parser import parse_bib_file
from source.bibliography.catalog_parser import parse_catalog
from source.tests.conftest import SAMPLE_BIB, SAMPLE_CATALOG


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_bib_entries_are_classified_by_kind(tmp_path):
    sources = {s.key: s for s in parse_bib_file(_write(tmp_path, "s.bib", SAMPLE_BIB))}

    assert sources["peyre2018computational"].kind is SourceKind.ARXIV
    assert sources["peyre2018computational"].arxiv_id == "1803.00567"
    assert sources["carlsson2009topology"].kind is SourceKind.DOI
    assert sources["carlsson2009topology"].doi == "10.1090/S0273-0979-09-01249-X"
    assert sources["argintum2025rtdlite"].kind is SourceKind.GITHUB
    assert sources["optimaltransport"].kind is SourceKind.WEB


def test_arxiv_preprint_wins_over_journal_doi(tmp_path):
    sources = {s.key: s for s in parse_bib_file(_write(tmp_path, "s.bib", SAMPLE_BIB))}

    assert sources["rakotomamonjy2020optimal"].kind is SourceKind.ARXIV


def test_bib_fields_are_normalised(tmp_path):
    source = parse_bib_file(_write(tmp_path, "s.bib", SAMPLE_BIB))[0]

    assert source.authors == ("Gabriel Peyré", "Marco Cuturi")
    assert source.year == 2019
    assert source.venue == "FnT ML"


def test_arxiv_id_is_taken_from_url_without_eprint(tmp_path):
    bib = "@misc{k, title={T}, author={A}, year={2020}, url={https://arxiv.org/abs/math/0301001}}"

    source = parse_bib_file(_write(tmp_path, "s.bib", bib))[0]

    assert (source.kind, source.arxiv_id) == (SourceKind.ARXIV, "math/0301001")


def test_catalog_extracts_priority_tags_section_and_summary(tmp_path):
    catalog = parse_catalog(_write(tmp_path, "S.md", SAMPLE_CATALOG))

    peyre = catalog["peyre2018computational"]
    assert peyre.priority == 3
    assert peyre.tags == ("SURVEY", "OT")
    assert peyre.section == "1. Books & surveys"
    assert peyre.summary == "Standard reference. Second line."
    assert catalog["carlsson2009topology"].priority == 1
    assert catalog["carlsson2009topology"].summary is None
    assert catalog["argintum2025rtdlite"].section == "2. Another section"


def test_load_sources_merges_catalog(tmp_path):
    bib = _write(tmp_path, "s.bib", SAMPLE_BIB)
    catalog = _write(tmp_path, "S.md", SAMPLE_CATALOG)

    sources = {s.key: s for s in load_sources(bib, catalog)}

    assert sources["peyre2018computational"].priority == 3
    assert sources["optimaltransport"].priority is None


def test_load_sources_works_without_catalog(tmp_path):
    bib = _write(tmp_path, "s.bib", SAMPLE_BIB)

    sources = load_sources(bib, tmp_path / "missing.md")

    assert len(sources) == 5
