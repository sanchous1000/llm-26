"""Дизайн эксперимента: пресеты гиперпараметров и профили прогонов.

Все структуры неизменяемые (frozen dataclass): пресет нельзя испортить
в процессе прогона, любое изменение порождает новый объект.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from llm_lab.inference.params import ConfigError, GenerationParams

# Лимит длины единственного пресета, который его задаёт (см. G_budget).
# Переопределяется через --budget-max-tokens.
DEFAULT_BUDGET_MAX_TOKENS = 256


@dataclass(frozen=True)
class ParamPreset:
    """Именованный набор гиперпараметров (режим прогона)."""

    key: str
    label: str
    mode: str  # "baseline" | "tuned"
    params: GenerationParams
    rationale: str

    @property
    def changed_params(self) -> tuple[str, ...]:
        return tuple(sorted(self.params.to_payload().keys()))


BASELINE_PRESET = ParamPreset(
    key="A_baseline",
    label="A. Базовый",
    mode="baseline",
    params=GenerationParams(),
    rationale="Ни один гиперпараметр не передаётся — работают дефолты LM Studio/движка.",
)

# Важно: длину ответа ограничивает ТОЛЬКО пресет F. У reasoning-моделей бюджет
# max_tokens съедают блоки <think>, и жёсткий лимит обрезает рассуждения, не дав
# дойти до ответа — остальные пресеты меняют другие гиперпараметры, а объём ответа
# задаётся текстом самого промпта («не более N слов»).
TUNED_PRESETS: tuple[ParamPreset, ...] = (
    ParamPreset(
        key="B_precise",
        label="B. Детерминированный",
        mode="tuned",
        params=GenerationParams(
            temperature=0.0,
            top_p=0.5,
            top_k=20,
            repeat_penalty=1.05,
        ),
        rationale="Максимальная предсказуемость: жадное декодирование, узкое ядро выборки, штраф за повторы.",
    ),
    ParamPreset(
        key="C_creative",
        label="C. Креативный",
        mode="tuned",
        params=GenerationParams(
            temperature=1.15,
            top_p=0.97,
            top_k=100,
            presence_penalty=0.6,
            frequency_penalty=0.3,
        ),
        rationale="Высокая энтропия + штрафы присутствия и частоты: проверяем рост вариативности и галлюцинаций.",
    ),
    ParamPreset(
        key="D_balanced",
        label="D. Сбалансированный",
        mode="tuned",
        params=GenerationParams(
            temperature=0.3,
            top_p=0.9,
            top_k=40,
        ),
        rationale="Рабочая точка «продакшн»: середина между B и C.",
    ),
    ParamPreset(
        key="E_warm",
        label="E. Тёплый",
        mode="tuned",
        params=GenerationParams(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
        ),
        rationale="Вторая точка температурного свипа T=0.0/0.3/0.7/1.15; отличается от D только температурой.",
    ),
    ParamPreset(
        key="F_min_p",
        label="F. Отсечение по min_p",
        mode="tuned",
        params=GenerationParams(
            temperature=0.8,
            min_p=0.1,
            top_p=1.0,
            repeat_penalty=1.15,
        ),
        rationale="Альтернативная стратегия сэмплирования: вместо top_p — порог min_p плюс сильный штраф за повторы.",
    ),
    ParamPreset(
        key="G_budget",
        label="G. Ограниченный бюджет",
        mode="tuned",
        params=GenerationParams(
            temperature=0.2,
            top_p=0.9,
            max_tokens=DEFAULT_BUDGET_MAX_TOKENS,
        ),
        rationale=(
            "Единственный пресет с лимитом длины: показывает деградацию при обрезке "
            "и то, как reasoning-модели тратят весь бюджет на рассуждения."
        ),
    ),
)

ALL_PRESETS: tuple[ParamPreset, ...] = (BASELINE_PRESET, *TUNED_PRESETS)

# Профили = наборы пресетов. `min` покрывает минимум лабораторной (A + B),
# `core` — рекомендуемый объём, `sweep` — чистый температурный свип,
# `full` — все режимы.
PROFILES: dict[str, tuple[str, ...]] = {
    "min": ("A_baseline", "B_precise"),
    "core": ("A_baseline", "B_precise", "C_creative", "G_budget"),
    "sweep": ("A_baseline", "B_precise", "D_balanced", "E_warm", "C_creative"),
    "full": tuple(p.key for p in ALL_PRESETS),
}

DEFAULT_PROFILE = "core"
DEFAULT_REPEATS = 3


def with_budget(presets: tuple[ParamPreset, ...], budget: int | None) -> tuple[ParamPreset, ...]:
    """Переопределить лимит длины в тех пресетах, которые его задают.

    Пресеты без max_tokens остаются без лимита — это осознанное решение:
    у reasoning-моделей бюджет уходит на блоки рассуждений.
    """
    if budget is None:
        return presets
    if budget <= 0:
        raise ConfigError("--budget-max-tokens должен быть положительным числом")
    return tuple(
        replace(preset, params=replace(preset.params, max_tokens=budget))
        if preset.params.max_tokens is not None
        else preset
        for preset in presets
    )


def preset_by_key(key: str) -> ParamPreset:
    for preset in ALL_PRESETS:
        if preset.key == key:
            return preset
    known = ", ".join(p.key for p in ALL_PRESETS)
    raise ConfigError(f"Неизвестный пресет {key!r}. Доступны: {known}")


def resolve_presets(profile: str, explicit_keys: tuple[str, ...]) -> tuple[ParamPreset, ...]:
    """Список пресетов по имени профиля или по явному перечислению ключей."""
    if explicit_keys:
        return tuple(preset_by_key(key) for key in explicit_keys)
    if profile not in PROFILES:
        known = ", ".join(PROFILES)
        raise ConfigError(f"Неизвестный профиль {profile!r}. Доступны: {known}")
    return tuple(preset_by_key(key) for key in PROFILES[profile])
