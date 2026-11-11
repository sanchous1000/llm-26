"""Форматирование Markdown-отчёта и автоматических выводов для п.5."""

from __future__ import annotations

from typing import Any

METRIC_KEYS = ("recall@5", "recall@10", "precision@5", "precision@10", "mrr")
QUALITY_KEYS = ("mrr", "recall@10", "recall@5", "precision@10", "precision@5")


def variant_title(row: dict[str, Any]) -> str:
    mode = "dense+BM25" if row["retrieval"] == "hybrid" else "dense"
    return f"{row['splitter']}, {mode}"


def index_title(row: dict[str, Any]) -> str:
    if row["index_type"] == "flat":
        return "flat"
    quant = "" if row["quantization"] == "none" else f", {row['quantization']}"
    return f"HNSW M={row['M']}, efc={row['ef_construction']}, ef={row['ef_search']}{quant}"


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _quality_key(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(row[key] for key in QUALITY_KEYS)


def _variant_sentence(row: dict[str, Any], rrf_k: int) -> str:
    title = variant_title(row)
    if row["retrieval"] == "hybrid":
        return f"`{title}` сливает FAISS и BM25 через RRF (1/({rrf_k}+rank))"
    return f"`{title}` ищет только FAISS"


def search_intro(rows: list[dict[str, Any]], rrf_k: int) -> str:
    seen: list[str] = []
    sentences: list[str] = []
    hybrid = False
    for row in rows:
        title = variant_title(row)
        if title in seen:
            continue
        seen.append(title)
        sentences.append(_variant_sentence(row, rrf_k))
        hybrid = hybrid or row["retrieval"] == "hybrid"
    text = "Сравниваются прогоны: " + "; ".join(sentences) + "."
    if hybrid:
        text += (
            " В плотное плечо HNSW попадают соседи глубины ef_search (не меньше 10). "
            "У flat в RRF входит полное точное ранжирование, BM25 ранжирует все чанки."
        )
    return text


def conclusion_lines(rows: list[dict[str, Any]], questions: list[dict[str, Any]]) -> list[str]:
    paragraphs: list[str] = []
    flat_rows = [row for row in rows if row["index_type"] == "flat"]
    if flat_rows:
        bits = [
            f"{variant_title(row)} ({row['n_chunks']} чанков): "
            f"Recall@5={_fmt(row['recall@5'])}, Recall@10={_fmt(row['recall@10'])}, "
            f"Precision@5={_fmt(row['precision@5'])}, Precision@10={_fmt(row['precision@10'])}, "
            f"MRR={_fmt(row['mrr'])}"
            for row in flat_rows
        ]
        paragraphs.append(
            "Точный flat-поиск отделяет чанкирование и способ слияния от параметров HNSW. "
            + ". ".join(bits)
            + "."
        )

    best = max(rows, key=_quality_key)
    leaders = [row for row in rows if _quality_key(row) == _quality_key(best)]
    near = [
        row for row in rows
        if best["mrr"] - row["mrr"] <= 0.01 and best["recall@10"] - row["recall@10"] <= 0.01
    ]
    cheap = min(near, key=lambda row: (row["index_bytes"], row["search_ms"]))
    if len(leaders) > 1:
        smallest = min(leaders, key=lambda row: (row["index_bytes"], row["search_ms"]))
        paragraphs.append(
            f"Максимум качества совпадает у {len(leaders)} конфигураций: "
            f"MRR={_fmt(best['mrr'])}, Recall@10={_fmt(best['recall@10'])}. "
            f"Меньший индекс среди них у «{variant_title(smallest)}», {index_title(smallest)}: "
            f"{smallest['index_bytes'] / 1024:.0f} КБ."
        )
    else:
        paragraphs.append(
            f"Максимум качества у «{variant_title(best)}», {index_title(best)}: "
            f"MRR={_fmt(best['mrr'])}, Recall@10={_fmt(best['recall@10'])}, "
            f"индекс {best['index_bytes'] / 1024:.0f} КБ."
        )
    if cheap not in leaders:
        paragraphs.append(
            "В пределах 0.01 по MRR и Recall@10 меньший индекс у "
            f"«{variant_title(cheap)}», {index_title(cheap)}: "
            f"{cheap['index_bytes'] / 1024:.0f} КБ, "
            f"MRR={_fmt(cheap['mrr'])}, Recall@10={_fmt(cheap['recall@10'])}."
        )

    by_id = {question["id"]: question["question"] for question in questions}
    missed = []
    for question in questions:
        bits = []
        for row in flat_rows:
            per = next(item for item in row["per_question"] if item["id"] == question["id"])
            if per["recall@10"] < 1:
                bits.append(
                    f"{variant_title(row)} Recall@10={_fmt(per['recall@10'])}, MRR={_fmt(per['mrr'])}"
                )
        if bits:
            missed.append(f"- {question['id']} {by_id[question['id']]}: " + "; ".join(bits))
    if missed:
        paragraphs.append("Вопросы с Recall@10 ниже 1 на flat:\n" + "\n".join(missed))
    elif flat_rows:
        paragraphs.append("На flat у всех вопросов Recall@10 = 1.")
    return paragraphs


def render_report(
    rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    query_embed_seconds: float,
    coverage: float,
    rrf_k: int,
) -> str:
    lines = [
        "# Пункт 5. Метрики retrieval",
        "",
        f"Эталон: {len(questions)} вопросов с дословными цитатами из `src/preprocessing/output/diploma.md`. "
        "Чанк релевантен пассажу, если нормализованная цитата входит в чанк или в нём есть непрерывный "
        f"фрагмент не короче {coverage:.0%} токенов цитаты. "
        "Recall@k — доля таких пассажей, которые задеты top-k. "
        "Precision@k — доля чанков top-k, которые задевают хотя бы один пассаж. "
        "MRR = 1 / ранг первого релевантного чанка.",
        "",
        search_intro(rows, rrf_k),
        "",
        f"Векторы вопросов посчитаны один раз за {query_embed_seconds:.1f} с. "
        "Колонка «мс» — время батча из всех вопросов, делённое на число вопросов, без эмбеддинга.",
        "",
        "| Вариант | Индекс | Recall@5 | Recall@10 | Precision@5 | Precision@10 | MRR | Совпадение с flat@10 | КБ | мс |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {index} | {r5} | {r10} | {p5} | {p10} | {mrr} | {flat} | {kb} | {ms} |".format(
                variant=variant_title(row),
                index=index_title(row),
                r5=_fmt(row["recall@5"]),
                r10=_fmt(row["recall@10"]),
                p5=_fmt(row["precision@5"]),
                p10=_fmt(row["precision@10"]),
                mrr=_fmt(row["mrr"]),
                flat=_fmt(row["flat_recall@10"]),
                kb=f"{row['index_bytes'] / 1024:.0f}",
                ms=f"{row['search_ms']:.2f}",
            )
        )
    lines.extend(["", "## Выводы", ""])
    lines.append("\n\n".join(conclusion_lines(rows, questions)))
    lines.append("")
    return "\n".join(lines)
