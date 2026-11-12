"""Загрузка конфигурации, переопределения из CLI и идентификаторы версий индекса."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import yaml

SOURCE_DIR = Path(__file__).resolve().parent
LAB_DIR = SOURCE_DIR.parent
DEFAULT_CONFIG = SOURCE_DIR / "config.yaml"


def nfc(text: str) -> str:
    """macOS хранит имена файлов в NFD («й» = «и» + знак) — приводим всё к NFC."""
    return unicodedata.normalize("NFC", text)


def _set_path(cfg: dict, dotted: str, value) -> None:
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def get(cfg: dict, dotted: str, default=None):
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def apply_overrides(cfg: dict, overrides: dict | list | None) -> dict:
    """overrides: {"chunking.chunk_size": 256} или ["chunking.chunk_size=256", ...]."""
    cfg = copy.deepcopy(cfg)
    if not overrides:
        return cfg
    items = overrides.items() if isinstance(overrides, dict) else (
        (s.split("=", 1)[0], yaml.safe_load(s.split("=", 1)[1])) for s in overrides
    )
    for key, value in items:
        _set_path(cfg, key.strip(), value)
    return cfg


def load_config(path: str | Path | None = None, overrides=None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)


def lab_path(cfg: dict, key: str) -> Path:
    p = Path(get(cfg, f"paths.{key}"))
    return p if p.is_absolute() else LAB_DIR / p


# ---------- CLI ----------
def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="путь к config.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="переопределить параметр, напр. --set chunking.chunk_size=256")


def add_index_args(parser: argparse.ArgumentParser) -> None:
    """Флаги параметров индекса (шаг 3) — синонимы для --set."""
    g = parser.add_argument_group("параметры индекса")
    g.add_argument("--splitter", choices=["recursive", "markdown", "hybrid"])
    g.add_argument("--chunk-size", type=int)
    g.add_argument("--overlap", type=int)
    g.add_argument("--headers", dest="include_headers", action="store_true", default=None)
    g.add_argument("--no-headers", dest="include_headers", action="store_false")
    g.add_argument("--dense-model")
    g.add_argument("--vector-type", choices=["dense", "sparse", "hybrid"])
    g.add_argument("--hnsw-m", type=int)
    g.add_argument("--ef-construct", type=int)
    g.add_argument("--quantization", choices=["none", "scalar"])


_FLAG_MAP = {
    "splitter": "chunking.splitter",
    "chunk_size": "chunking.chunk_size",
    "overlap": "chunking.overlap",
    "include_headers": "chunking.include_headers",
    "dense_model": "embeddings.dense_model",
    "vector_type": "embeddings.vector_type",
    "hnsw_m": "vector_store.hnsw.m",
    "ef_construct": "vector_store.hnsw.ef_construct",
    "quantization": "vector_store.quantization",
}


def config_from_args(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config, args.set)
    flags = {dotted: getattr(args, name) for name, dotted in _FLAG_MAP.items()
             if getattr(args, name, None) is not None}
    return apply_overrides(cfg, flags)


# ---------- версии артефактов ----------
def corpus_fingerprint(cfg: dict) -> str:
    """Хэш нормализованного корпуса: если документы перепарсились иначе — индекс пересоберётся."""
    md_dir = lab_path(cfg, "markdown_dir")
    h = hashlib.sha1()
    for p in sorted(md_dir.glob("*.md")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:10]


def index_params(cfg: dict) -> dict:
    """Всё, от чего зависят чанки и векторы (шаг 3)."""
    c, e = cfg["chunking"], cfg["embeddings"]
    return {
        "splitter": c["splitter"],
        "chunk_size": int(c["chunk_size"]),
        "overlap": int(c["overlap"]),
        "include_headers": bool(c["include_headers"]),
        "min_chunk_tokens": int(c.get("min_chunk_tokens", 0)),
        "tokenizer": c.get("tokenizer"),
        "vector_type": e["vector_type"],
        "dense_model": e["dense_model"] if e["vector_type"] != "sparse" else None,
        "bm25": e.get("bm25") if e["vector_type"] != "dense" else None,
    }


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-").lower()


def index_id(cfg: dict) -> str:
    p = index_params(cfg)
    digest = hashlib.sha1(
        (json.dumps(p, sort_keys=True) + corpus_fingerprint(cfg)).encode()
    ).hexdigest()[:8]
    model = slug(p["dense_model"] or "bm25")
    return (f"{p['splitter']}_cs{p['chunk_size']}_ov{p['overlap']}_h{int(p['include_headers'])}"
            f"_{p['vector_type']}_{model}_{digest}")


def collection_name(cfg: dict) -> str:
    """Коллекция = версия индекса + параметры HNSW/квантования (они влияют только на хранилище)."""
    vs = cfg["vector_store"]
    q = "_q8" if vs.get("quantization") == "scalar" else ""
    return f"{vs['collection_prefix']}_{index_id(cfg)}_m{vs['hnsw']['m']}_ef{vs['hnsw']['ef_construct']}{q}"
