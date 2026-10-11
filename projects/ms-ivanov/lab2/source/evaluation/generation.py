"""Candidate question generation: one document at a time, resumable."""

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from source.evaluation import candidates as candidate_lib
from source.evaluation import opencode
from source.evaluation.documents import read_body, split_into_parts, strip_references
from source.evaluation.prompts import build_prompt
from source.storage import EntryRecord, EntryStore

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GenerationSettings:
    """Knobs of one generation run."""

    model: str = opencode.DEFAULT_MODEL
    per_part: int = 5
    max_chars: int = 150_000
    timeout_s: int = 900
    retries: int = 2


def generate_all(
    records: Sequence[EntryRecord],
    store: EntryStore,
    output_dir: Path,
    settings: GenerationSettings,
    *,
    force: bool = False,
) -> None:
    """Generate candidates for every record, strictly one after another."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="opencode-questions-"))
    for number, record in enumerate(records, start=1):
        key = record.source.key
        document = store.entry_dir(record.source) / record.conversion.get("path", "")
        target = output_dir / f"{key}.json"
        body = read_body(document)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if not force and _is_current(target, digest, settings):
            logger.info("[%d/%d] %s: up to date", number, len(records), key)
            continue
        try:
            result = _generate_document(record, body, digest, settings, scratch)
        except opencode.OpencodeError as error:
            logger.error("[%d/%d] %s: giving up: %s", number, len(records), key, error)
            if not error.retryable:
                raise
            continue
        _write_json(target, result)
        ok = sum(1 for c in result["candidates"] if c["status"] == "ok")
        logger.info(
            "[%d/%d] %s: %d candidates, %d verified",
            number,
            len(records),
            key,
            len(result["candidates"]),
            ok,
        )


def _is_current(target: Path, digest: str, settings: GenerationSettings) -> bool:
    if not target.exists():
        return False
    data = json.loads(target.read_text(encoding="utf-8"))
    return (
        data.get("schema") == SCHEMA_VERSION
        and data.get("body_sha256") == digest
        and data.get("model") == settings.model
        and any(c["status"] == "ok" for c in data.get("candidates", []))
    )


def _generate_document(
    record: EntryRecord,
    body: str,
    digest: str,
    settings: GenerationSettings,
    scratch: Path,
) -> dict:
    parts = split_into_parts(strip_references(body), settings.max_chars)
    collected: list[candidate_lib.Candidate] = []
    for part in parts:
        collected.extend(_generate_part(part, body, settings, scratch))
    return {
        "schema": SCHEMA_VERSION,
        "key": record.source.key,
        "title": record.source.title,
        "model": settings.model,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "body_sha256": digest,
        "body_chars": len(body),
        "parts": len(parts),
        "candidates": [c.to_dict() for c in collected],
    }


def _generate_part(part, body: str, settings: GenerationSettings, scratch: Path):
    attachment = scratch / "document.md"
    attachment.write_text(part.text, encoding="utf-8")
    prompt = build_prompt(count=settings.per_part, part=part.index, parts=part.total)
    last_error: Exception | None = None
    for attempt in range(1, settings.retries + 2):
        try:
            answer = opencode.ask(
                prompt,
                attachment,
                model=settings.model,
                timeout_s=settings.timeout_s,
                workdir=scratch,
            )
            return candidate_lib.verify(
                candidate_lib.parse_answer(answer), body, part.index
            )
        except opencode.OpencodeError as error:
            if not error.retryable:
                raise
            last_error = error
        except ValueError as error:  # malformed JSON: ask again
            last_error = error
        logger.warning(
            "part %d/%d attempt %d failed: %s",
            part.index,
            part.total,
            attempt,
            last_error,
        )
    raise opencode.OpencodeError(f"all attempts failed: {last_error}", retryable=True)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
