"""Human-readable summary of the state of the corpus."""

from collections import Counter
from collections.abc import Sequence

from source.storage import EntryRecord
from source.storage.records import STATUS_OK


def format_report(
    records: Sequence[EntryRecord], expected_total: int | None = None
) -> str:
    """Summarise counts per kind/status/format and list entries without text."""
    lines = [f"Entries with a record: {len(records)}"]
    if expected_total is not None:
        lines[0] += f" / {expected_total}"
    lines += _count_lines(
        "By kind and status",
        Counter((record.source.kind.value, record.status) for record in records),
    )
    lines += _count_lines(
        "Primary format",
        Counter(
            (record.primary.name if record.primary else "none",) for record in records
        ),
    )
    lines += _conversion_lines(records)
    lines += _problem_lines(records)
    return "\n".join(lines)


def _count_lines(title: str, counts: Counter) -> list[str]:
    lines = ["", f"{title}:"]
    lines += [
        f"  {' / '.join(key):<28}{count:>5}" for key, count in sorted(counts.items())
    ]
    return lines


def _conversion_lines(records: Sequence[EntryRecord]) -> list[str]:
    counts = Counter(
        (
            record.conversion.get("status", "not converted"),
            record.conversion.get("converter", "-"),
        )
        for record in records
        if record.files
    )
    lines = _count_lines("Markdown conversion (status / converter)", counts)
    failed = [
        f"  {record.source.key}: {record.conversion.get('error')}"
        for record in records
        if record.conversion.get("status") == "error"
    ]
    return lines + (["", "Conversion errors:", *failed] if failed else [])


def _problem_lines(records: Sequence[EntryRecord]) -> list[str]:
    problems = [record for record in records if record.status != STATUS_OK]
    if not problems:
        return ["", "All entries have at least one downloaded file."]
    lines = ["", f"Entries without any downloaded file ({len(problems)}):"]
    for record in sorted(
        problems, key=lambda item: (item.source.kind.value, item.source.key)
    ):
        reasons = "; ".join(
            f"{n}: {r}" for n, r in {**record.missing, **record.errors}.items()
        )
        lines.append(f"  [{record.source.kind.value}] {record.source.key} - {reasons}")
    return lines
