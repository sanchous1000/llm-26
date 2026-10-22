"""Static configuration: default paths, network politeness and file names."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_BIB_PATH = DATA_DIR / "sources.bib"
DEFAULT_CATALOG_PATH = DATA_DIR / "SOURCES.md"
DEFAULT_OUTPUT_DIR = DATA_DIR / "papers"
DEFAULT_EVAL_DIR = PROJECT_ROOT / "eval"

USER_AGENT = "ITMO-LLM-lab2-corpus-downloader/1.0 (academic coursework; bounded list)"
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 3
BACKOFF_BASE_S = 2.0

# Minimal pause between two requests to the same host. arxiv.org publishes
# "Crawl-delay: 15" in robots.txt, so we honour it; ar5iv has no such limit.
DEFAULT_HOST_DELAY_S = 1.0
HOST_DELAYS_S = {
    "arxiv.org": 15.0,
    "export.arxiv.org": 15.0,
    "ar5iv.labs.arxiv.org": 2.0,
    "api.github.com": 1.0,
    "api.openalex.org": 0.5,
    "api.semanticscholar.org": 1.5,
}

ARXIV_HTML_URLS = (
    "https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
    "https://arxiv.org/html/{arxiv_id}",
)
ARXIV_PDF_URLS = ("https://arxiv.org/pdf/{arxiv_id}",)
OPENALEX_WORK_URL = "https://api.openalex.org/works/doi:{doi}"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf,externalIds"
GITHUB_README_URL = "https://api.github.com/repos/{owner}/{repo}/readme"
MAX_DOI_PDF_CANDIDATES = 6

META_FILENAME = "meta.json"
MANIFEST_FILENAME = "manifest.json"
