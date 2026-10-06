"""Автоматические проверки качества ответа.

Каждая проверка возвращает CheckResult со score в [0, 1]. Средневзвешенный
score по всем проверкам промпта = «качество/адекватность» в отчёте.
Проверки эвристические: они ловят нарушение формата, потерю фактов и
выдуманные сущности, но не заменяют ручную оценку (для неё в отчёте
сохраняются полные тексты ответов).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from llm_lab.evaluation.text import (
    extract_json_object,
    extract_numbers,
    normalize_for_match,
    word_count,
)

# Версия набора проверок. v2 — исправления по итогам ручного разбора ответов
# (словоформы, парсинг чисел, полнота JSON, смешение языков, искажение фактов).
CHECKS_VERSION = 2

# Маркер в тексте детали проверки: по нему отчёт собирает сводку галлюцинаций.
HALLUCINATION_MARKER = "ГАЛЛЮЦИНАЦИЯ"
# Маркер переключения на другой язык посреди русского ответа.
CODE_SWITCH_MARKER = "СМЕШЕНИЕ ЯЗЫКОВ"


@dataclass(frozen=True)
class CheckResult:
    name: str
    score: float
    passed: bool
    detail: str
    weight: float = 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "passed": self.passed,
            "detail": self.detail,
            "weight": self.weight,
        }


class Check:
    """Базовый класс проверки."""

    name: str = "check"
    weight: float = 1.0

    def evaluate(self, answer: str) -> CheckResult:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def _result(self, score: float, detail: str, threshold: float = 0.999) -> CheckResult:
        score = max(0.0, min(1.0, score))
        return CheckResult(
            name=self.name,
            score=score,
            passed=score >= threshold,
            detail=detail,
            weight=self.weight,
        )


@dataclass(frozen=True)
class ContainsAll(Check):
    """Все обязательные фрагменты присутствуют (частичный балл)."""

    values: tuple[str, ...]
    name: str = "contains_all"
    weight: float = 1.0

    def evaluate(self, answer: str) -> CheckResult:
        haystack = normalize_for_match(answer)
        missing = [v for v in self.values if normalize_for_match(v) not in haystack]
        score = 1.0 - len(missing) / max(1, len(self.values))
        detail = "все обязательные фрагменты на месте" if not missing else f"нет: {', '.join(missing)}"
        return self._result(score, detail)


@dataclass(frozen=True)
class ContainsAny(Check):
    """Хотя бы один из вариантов присутствует."""

    values: tuple[str, ...]
    name: str = "contains_any"
    weight: float = 1.0

    def evaluate(self, answer: str) -> CheckResult:
        haystack = normalize_for_match(answer)
        hits = [v for v in self.values if normalize_for_match(v) in haystack]
        detail = f"найдено: {', '.join(hits)}" if hits else f"не найдено ни одного из: {', '.join(self.values[:6])}"
        return self._result(1.0 if hits else 0.0, detail)


@dataclass(frozen=True)
class NotContainsAny(Check):
    """Запрещённых фрагментов нет."""

    values: tuple[str, ...]
    name: str = "not_contains"
    weight: float = 1.0
    marks_hallucination: bool = True

    def evaluate(self, answer: str) -> CheckResult:
        haystack = normalize_for_match(answer)
        hits = [v for v in self.values if normalize_for_match(v) in haystack]
        if not hits:
            return self._result(1.0, "запрещённых фрагментов нет")
        prefix = f"{HALLUCINATION_MARKER} — " if self.marks_hallucination else ""
        return self._result(0.0, f"{prefix}найдены запрещённые фрагменты: {', '.join(hits)}")


@dataclass(frozen=True)
class WordBudget(Check):
    """Соблюдение лимита длины (мягкая деградация за перебор)."""

    max_words: int
    min_words: int = 1
    name: str = "word_budget"
    weight: float = 1.0

    def evaluate(self, answer: str) -> CheckResult:
        count = word_count(answer)
        if count < self.min_words:
            return self._result(
                count / max(1, self.min_words),
                f"{count} слов — короче минимума {self.min_words}",
            )
        if count <= self.max_words:
            return self._result(1.0, f"{count} слов (лимит {self.max_words})")
        overflow = (count - self.max_words) / self.max_words
        return self._result(
            max(0.0, 1.0 - overflow),
            f"{count} слов — превышен лимит {self.max_words} на {count - self.max_words}",
        )


@dataclass(frozen=True)
class RegexAbsent(Check):
    """Ответ НЕ содержит шаблон (запрещённый формат или искажение факта)."""

    pattern: str
    name: str = "regex_absent"
    weight: float = 1.0
    description: str = ""
    marks_hallucination: bool = False

    def evaluate(self, answer: str) -> CheckResult:
        found = re.search(self.pattern, answer or "", re.IGNORECASE | re.MULTILINE)
        label = self.description or self.pattern
        if not found:
            return self._result(1.0, f"нет: {label}")
        prefix = f"{HALLUCINATION_MARKER} — " if self.marks_hallucination else ""
        return self._result(0.0, f"{prefix}найдено «{found.group(0).strip()}»: {label}")


@dataclass(frozen=True)
class NoForeignWords(Check):
    """Нет вкраплений слов латиницей в русском тексте (переключение языка).

    Коды и обозначения с цифрами (5xx, IP54, TK3-220-RU) словами не считаются.
    """

    allowed: tuple[str, ...] = ()
    min_letters: int = 3
    name: str = "only_russian"
    weight: float = 1.0

    def evaluate(self, answer: str) -> CheckResult:
        pattern = rf"(?<![\w\-])[A-Za-z]{{{self.min_letters},}}(?![\w\-])"
        allowed = {a.lower() for a in self.allowed}
        found = [w for w in re.findall(pattern, answer or "") if w.lower() not in allowed]
        if not found:
            return self._result(1.0, "слов на другом языке нет")
        unique = list(dict.fromkeys(found))
        return self._result(0.0, f"{CODE_SWITCH_MARKER} — латиницей: {', '.join(unique[:6])}")


@dataclass(frozen=True)
class RegexPresent(Check):
    """Ответ соответствует регулярному выражению."""

    pattern: str
    name: str = "regex"
    weight: float = 1.0
    description: str = ""

    def evaluate(self, answer: str) -> CheckResult:
        found = re.search(self.pattern, answer or "", re.IGNORECASE | re.MULTILINE)
        label = self.description or self.pattern
        detail = f"шаблон найден: {label}" if found else f"шаблон не найден: {label}"
        return self._result(1.0 if found else 0.0, detail)


@dataclass(frozen=True)
class LabelsMatch(Check):
    """Классификация: разбор строк вида `1: Label` и сверка с эталоном."""

    expected: tuple[str, ...]
    allowed: tuple[str, ...]
    name: str = "labels_accuracy"
    weight: float = 2.0

    _LINE_RE = re.compile(r"^\s*(\d+)\s*[:.)\-–]\s*(.+?)\s*$", re.MULTILINE)

    def evaluate(self, answer: str) -> CheckResult:
        parsed = self._parse(answer or "")
        correct = 0
        problems: list[str] = []
        for index, expected_label in enumerate(self.expected, start=1):
            got = parsed.get(index)
            if got is None:
                problems.append(f"#{index}: нет ответа")
                continue
            if got == normalize_for_match(expected_label):
                correct += 1
            elif got not in {normalize_for_match(a) for a in self.allowed}:
                problems.append(f"#{index}: метка вне списка ({got!r}, ждали {expected_label})")
            else:
                problems.append(f"#{index}: {got!r} вместо {expected_label!r}")
        score = correct / max(1, len(self.expected))
        detail = f"верно {correct}/{len(self.expected)}"
        if problems:
            detail += "; " + "; ".join(problems)
        return self._result(score, detail)

    def _parse(self, answer: str) -> dict[int, str]:
        parsed: dict[int, str] = {}
        for raw_index, raw_label in self._LINE_RE.findall(answer):
            try:
                index = int(raw_index)
            except ValueError:
                continue
            parsed.setdefault(index, normalize_for_match(raw_label))
        return parsed


@dataclass(frozen=True)
class JsonSchemaCheck(Check):
    """Извлечение: валидность JSON, набор полей, значения и обязательные null.

    `null_fields` — поля, которых нет в исходном тексте: модель обязана
    вернуть null. Любое заполненное значение здесь = галлюцинация.
    """

    required_fields: tuple[str, ...]
    # Значение поля или кортеж допустимых вариантов («220» / «220 В»).
    expected_values: dict[str, str | tuple[str, ...]] = field(default_factory=dict)
    null_fields: tuple[str, ...] = ()
    name: str = "json_schema"
    weight: float = 2.0

    def evaluate(self, answer: str) -> CheckResult:
        parsed, error = extract_json_object(answer or "")
        if parsed is None:
            return self._result(0.0, f"JSON не разобран ({error})")

        parts: list[float] = []
        notes: list[str] = []

        missing = [f for f in self.required_fields if f not in parsed]
        parts.append(1.0 - len(missing) / max(1, len(self.required_fields)))
        notes.append("все поля на месте" if not missing else f"нет полей: {', '.join(missing)}")

        if self.expected_values:
            wrong: list[str] = []
            for key, expected in self.expected_values.items():
                variants = (expected,) if isinstance(expected, str) else expected
                got = parsed.get(key)
                accepted = {normalize_for_match(v) for v in variants}
                if got is None or normalize_for_match(str(got)) not in accepted:
                    wrong.append(f"{key}={got!r} (ждали {variants[0]!r})")
            parts.append(1.0 - len(wrong) / len(self.expected_values))
            notes.append("значения верны" if not wrong else f"расхождения: {'; '.join(wrong)}")

        if self.null_fields:
            invented = [
                f for f in self.null_fields
                if f in parsed and parsed[f] not in (None, "", [], {}, "null", "-")
            ]
            parts.append(1.0 - len(invented) / len(self.null_fields))
            notes.append(
                "выдуманных значений нет"
                if not invented
                else f"{HALLUCINATION_MARKER} — заполнены отсутствующие в тексте поля: {', '.join(invented)}"
            )

        return self._result(sum(parts) / len(parts), "; ".join(notes))


@dataclass(frozen=True)
class NumericAnswer(Check):
    """Числовой ответ: сверка последнего/любого числа с эталоном."""

    expected: float
    tolerance: float = 0.5
    name: str = "numeric_answer"
    weight: float = 2.0

    def evaluate(self, answer: str) -> CheckResult:
        numbers = extract_numbers(answer or "")
        if not numbers:
            return self._result(0.0, "числа в ответе не найдены")
        matches = [n for n in numbers if abs(n - self.expected) <= self.tolerance]
        if not matches:
            return self._result(0.0, f"итог неверен: числа {list(numbers)[-5:]}, ждали {self.expected:g}")
        if abs(numbers[-1] - self.expected) <= self.tolerance:
            return self._result(1.0, f"итог верен: {self.expected:g}")
        return self._result(0.6, f"верное число встречается, но итоговое — {numbers[-1]:g}")


@dataclass(frozen=True)
class NumbersFromSource(Check):
    """Суммаризация: все числа ответа должны встречаться в исходном тексте.

    Числа «из ниоткуда» — прямой признак галлюцинации при пересказе.
    """

    source: str
    allowed_extra: tuple[float, ...] = ()
    name: str = "numbers_grounded"
    weight: float = 1.5

    def evaluate(self, answer: str) -> CheckResult:
        answer_numbers = extract_numbers(answer or "")
        if not answer_numbers:
            return self._result(1.0, "чисел в ответе нет — выдумывать нечего")
        allowed = set(extract_numbers(self.source)) | set(self.allowed_extra)
        # Нумерация пунктов (1., 2., 3.) не считается фактом.
        allowed |= {1.0, 2.0, 3.0, 4.0, 5.0}
        invented = [n for n in answer_numbers if not any(abs(n - a) < 1e-6 for a in allowed)]
        score = 1.0 - len(invented) / len(answer_numbers)
        detail = (
            "все числа есть в исходнике"
            if not invented
            else f"{HALLUCINATION_MARKER} — этих чисел нет в исходнике: {', '.join(f'{n:g}' for n in invented)}"
        )
        return self._result(score, detail)


@dataclass(frozen=True)
class RefusalExpected(Check):
    """Ловушка на галлюцинации: модель обязана сообщить о несуществующем объекте.

    Балл 1.0 — есть явное указание на отсутствие/неизвестность.
    Балл 0.0 — модель уверенно «пересказывает» несуществующий документ.
    """

    markers: tuple[str, ...]
    fabrication_markers: tuple[str, ...] = ()
    name: str = "refusal_expected"
    weight: float = 2.0

    def evaluate(self, answer: str) -> CheckResult:
        haystack = normalize_for_match(answer)
        honest = [m for m in self.markers if normalize_for_match(m) in haystack]
        fabricated = [m for m in self.fabrication_markers if normalize_for_match(m) in haystack]
        if honest and not fabricated:
            return self._result(1.0, f"корректный отказ (маркеры: {', '.join(honest[:3])})")
        if honest and fabricated:
            return self._result(0.5, f"оговорка есть, но рядом придуманные детали: {', '.join(fabricated[:3])}")
        if fabricated:
            return self._result(
                0.0,
                f"{HALLUCINATION_MARKER} — изложены детали несуществующего документа: {', '.join(fabricated[:3])}",
            )
        if len(answer or "") < 20:
            return self._result(0.0, "ответ пуст или слишком короток для оценки")
        return self._result(
            0.0,
            f"{HALLUCINATION_MARKER} — нет признания незнания, ответ дан по несуществующему объекту",
        )


EMPTY_ANSWER_DETAIL = "пустой ответ — требование не выполнено"


def evaluate_all(checks: tuple[Check, ...], answer: str) -> tuple[CheckResult, ...]:
    """Все проверки промпта. Пустой ответ проваливает каждую из них: иначе проверки
    «на отсутствие» (нет выдуманных чисел, нет латиницы) засчитывались бы ответу,
    который целиком съеден рассуждениями до лимита max_tokens."""
    if not (answer or "").strip():
        return tuple(
            CheckResult(name=check.name, score=0.0, passed=False, detail=EMPTY_ANSWER_DETAIL, weight=check.weight)
            for check in checks
        )
    return tuple(check.evaluate(answer) for check in checks)


def aggregate_score(results: tuple[CheckResult, ...]) -> float:
    """Средневзвешенный балл качества в [0, 1]."""
    if not results:
        return 0.0
    total_weight = sum(r.weight for r in results) or 1.0
    return sum(r.score * r.weight for r in results) / total_weight
