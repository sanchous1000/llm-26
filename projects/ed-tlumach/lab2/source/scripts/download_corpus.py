"""Шаг 1. Скачивание корпуса: статьи arXiv по теме или по идентификаторам, при необходимости документация из GitHub.

Примеры:
  python scripts/download_corpus.py --arxiv-query "retrieval augmented generation" --max 25
  python scripts/download_corpus.py --arxiv 2005.11401 2309.06180
  python scripts/download_corpus.py --repo https://github.com/vllm-project/vllm.git --subdir docs

Рядом с каждым PDF сохраняется файл <id>.json с названием, авторами и датой статьи.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # чтобы работал import book_rag_backend
from book_rag_backend.config import ROOT

ATOM = "{http://www.w3.org/2005/Atom}"
HEADERS = {"User-Agent": "lab2-rag/1.0 (student project)"}
PAUSE = 3  # arXiv просит не чаще одного запроса в 3 секунды


def build_query(text: str, category: str | None) -> str:
    """Слова запроса соединяются через AND; если в запросе есть «:», он передаётся в синтаксисе arXiv как есть."""
    q = text if ":" in text else " AND ".join(f"all:{w}" for w in text.split())
    return f"({q}) AND cat:{category}" if category else q


def parse_atom(xml_text: str) -> list[dict]:
    papers = []
    for e in ET.fromstring(xml_text).findall(f"{ATOM}entry"):
        aid = re.sub(r"v\d+$", "", e.findtext(f"{ATOM}id", "").rsplit("/abs/", 1)[-1])
        papers.append({
            "id": aid,
            "title": " ".join(e.findtext(f"{ATOM}title", "").split()),
            "authors": [a.findtext(f"{ATOM}name", "") for a in e.findall(f"{ATOM}author")],
            "published": e.findtext(f"{ATOM}published", ""),
            "abs_url": f"https://arxiv.org/abs/{aid}",
        })
    return papers


def fetch(url: str, retries: int = 2) -> bytes:
    for attempt in range(retries + 1):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=90).read()
        except Exception as e:
            if attempt == retries:
                raise
            print(f"  ошибка {type(e).__name__}, повтор через 10 с")
            time.sleep(10)


def search_arxiv(query: str, n: int, sort: str, category: str | None) -> list[dict]:
    params = urllib.parse.urlencode({"search_query": build_query(query, category), "start": 0, "max_results": n,
                                     "sortBy": sort, "sortOrder": "descending"})
    return parse_atom(fetch(f"https://export.arxiv.org/api/query?{params}").decode("utf-8"))


def lookup_arxiv(ids: list[str]) -> list[dict]:
    params = urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    return parse_atom(fetch(f"https://export.arxiv.org/api/query?{params}").decode("utf-8"))


def download_papers(papers: list[dict], out: Path) -> None:
    dest = out / "papers"
    dest.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(papers, 1):
        stem = p["id"].replace("/", "_")
        pdf, meta = dest / f"{stem}.pdf", dest / f"{stem}.json"
        if pdf.exists() and meta.exists():
            print(f"[{i}/{len(papers)}] {p['id']} уже скачана")
            continue
        time.sleep(PAUSE)
        data = fetch(f"https://arxiv.org/pdf/{p['id']}")
        if not data.startswith(b"%PDF"):
            print(f"[{i}/{len(papers)}] {p['id']}: arXiv вернул не PDF, пропускаю")
            continue
        pdf.write_bytes(data)
        meta.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{i}/{len(papers)}] {p['id']} {p['title'][:70]}")


def download_repo_docs(args, out: Path) -> None:
    tmp = out / ".clone_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(a, check=True, capture_output=True, text=True).stdout.strip()
    run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "--branch", args.ref, args.repo, str(tmp))
    run("git", "-C", str(tmp), "sparse-checkout", "set", args.subdir)
    commit = run("git", "-C", str(tmp), "rev-parse", "HEAD")
    dest = out / args.name
    shutil.rmtree(dest, ignore_errors=True)
    n = 0
    for f in (tmp / args.subdir).rglob("*.md"):
        target = dest / f.relative_to(tmp / args.subdir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        n += 1
    (dest / "SOURCE.txt").write_text(f"{args.repo}\nref: {args.ref}\ncommit: {commit}\nпапка: {args.subdir}\n", encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Документация: {n} файлов md в {dest}, коммит {commit[:10]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--arxiv-query", help="тема для поиска на arXiv, например: retrieval augmented generation")
    ap.add_argument("--max", type=int, default=25, help="сколько статей скачать по запросу")
    ap.add_argument("--arxiv-cat", help="ограничить категорией arXiv, например cs.CL")
    ap.add_argument("--arxiv-sort", default="relevance", choices=["relevance", "submittedDate", "lastUpdatedDate"])
    ap.add_argument("--arxiv", nargs="*", default=[], help="идентификаторы статей arXiv, например 2005.11401")
    ap.add_argument("--repo", help="репозиторий GitHub с документацией (необязательно)")
    ap.add_argument("--ref", default="main", help="ветка или тег репозитория")
    ap.add_argument("--subdir", default="docs", help="папка с документацией внутри репозитория")
    ap.add_argument("--name", default="docs", help="подпапка в data/raw для документации из репозитория")
    args = ap.parse_args()
    out = Path(args.out)
    if not (args.arxiv_query or args.arxiv or args.repo):
        ap.error("укажите --arxiv-query, --arxiv или --repo")

    if args.repo:
        download_repo_docs(args, out)
    papers = []
    if args.arxiv_query:
        papers += search_arxiv(args.arxiv_query, args.max, args.arxiv_sort, args.arxiv_cat)
        print(f"Найдено статей: {len(papers)}")
    if args.arxiv:
        time.sleep(PAUSE if papers else 0)
        papers += lookup_arxiv(args.arxiv)
    if papers:
        download_papers(list({p["id"]: p for p in papers}.values()), out)
