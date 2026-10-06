"""Графики для отчёта (PNG в results/figures/).

Требует matplotlib (см. requirements.txt). Данные — из reporting/analysis.py, поэтому
цифры на графиках совпадают с таблицами в results/analysis.md.

Запуск:  python source/main.py plots
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from llm_lab.reporting.analysis import (
    LENGTH_LIMITED_PRESET,
    PRESET_ORDER,
    PROMPT_ORDER,
    WORD_LIMITS,
    build_analysis,
    is_acceptable,
    load_all,
    load_critical_issues,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter
except ImportError:  # pragma: no cover - зависит от окружения
    plt = None

# Палитра: первые три категориальных слота (проверены валидатором на всех парах, светлая тема).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
SERIES_MARKERS = ("o", "s", "^")  # вторичное кодирование: форма не зависит от цвета
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
DPI = 160
FIGURES_DIRNAME = "figures"
REASONING_BUDGET = 8192
LABEL_ANCHORS = {0: "right", 1: "left", 2: "right"}  # Qwen справа, Gemma слева

PROMPT_LABELS = {
    "P1": "P1 письмо",
    "P2": "P2 классификация",
    "P3": "P3 JSON",
    "P4": "P4 резюме",
    "P5": "P5 расчёт",
    "P6": "P6 ловушка",
}


def style_axes(ax: Any, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(which="both", colors=INK_SECONDARY, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def new_figure(width: float, height: float, ncols: int = 1) -> tuple[Any, Any]:
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE)
    return fig, axes


def title(fig: Any, text: str, subtitle: str) -> None:
    fig.text(0.012, 0.975, text, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK)
    fig.text(0.012, 0.915, subtitle, ha="left", va="top", fontsize=9.5, color=INK_SECONDARY)


def legend(fig: Any, models: list[str], names: dict[str, str], y: float = 0.84, bars: bool = False) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SERIES_COLORS[i]) if bars else
        plt.Line2D([], [], color=SERIES_COLORS[i], marker=SERIES_MARKERS[i], markersize=7, linewidth=2,
                   markeredgecolor=SURFACE, markeredgewidth=1)
        for i in range(len(models))
    ]
    fig.legend(handles, [names[m] for m in models], loc="upper left", bbox_to_anchor=(0.005, y), ncol=len(models),
               frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY, handlelength=2.2, columnspacing=1.6)


def save(fig: Any, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------

def plot_acceptability_by_task(a: dict[str, Any], out: Path) -> Path:
    models, names = a["models"], a["short_names"]
    fig, ax = new_figure(10, 4.6)
    style_axes(ax)
    width = 0.24
    for i, model in enumerate(models):
        values = [a["by_prompt"][model][pid] or 0.0 for pid in PROMPT_ORDER]
        xs = [x + (i - 1) * width for x in range(len(PROMPT_ORDER))]
        ax.bar(xs, values, width=width, color=SERIES_COLORS[i], edgecolor=SURFACE, linewidth=1.2)
        for x, v in zip(xs, values):
            if v < 0.995:  # подписываем только то, что отличается от 100 %
                ax.text(x, v + 0.02, f"{v * 100:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xticks(range(len(PROMPT_ORDER)), [PROMPT_LABELS[p] for p in PROMPT_ORDER], color=INK_SECONDARY)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    title(fig, "Доля приемлемых ответов по задачам",
          "18 прогонов на ячейку: пресеты A–F × 3 повтора; G_budget исключён. Приемлемо = ключевые автопроверки + ручной разбор")
    legend(fig, models, names, bars=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.76, bottom=0.1)
    return save(fig, out / "acceptability_by_task.png")


def plot_tokens_by_task(results_root: Path, a: dict[str, Any], out: Path) -> Path:
    by_model = load_all(results_root)
    models, names = a["models"], a["short_names"]
    fig, ax = new_figure(10, 5.0)
    style_axes(ax)
    offset = 0.26
    for i, model in enumerate(models):
        for x, pid in enumerate(PROMPT_ORDER):
            tokens = [r["metrics"]["completion_tokens"] for r in by_model[model]
                      if r["prompt_id"] == pid and r["preset_key"] != LENGTH_LIMITED_PRESET]
            center = x + (i - 1) * offset
            jitter = [center + ((k % 6) - 2.5) * 0.022 for k in range(len(tokens))]
            ax.scatter(jitter, tokens, s=26, color=SERIES_COLORS[i], marker=SERIES_MARKERS[i],
                       edgecolors=SURFACE, linewidths=0.8, alpha=0.9, zorder=3)
            med = statistics.median(tokens)
            ax.plot([center - 0.09, center + 0.09], [med, med], color=INK, linewidth=2, zorder=4,
                    solid_capstyle="round")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    ax.axhline(REASONING_BUDGET, color=INK_MUTED, linewidth=1)
    ax.text(len(PROMPT_ORDER) - 0.5, REASONING_BUDGET * 0.9, "бюджет рассуждений ≈ 8 192",
            ha="right", va="top", fontsize=8.5, color=INK_MUTED)
    ax.set_xticks(range(len(PROMPT_ORDER)), [PROMPT_LABELS[p] for p in PROMPT_ORDER], color=INK_SECONDARY)
    ax.set_ylabel("токенов вывода (лог. шкала)", color=INK_SECONDARY, fontsize=9)
    title(fig, "Сколько токенов тратится на короткий ответ",
          "Точка — прогон (пресеты A–F), черта — медиана. У Qwen и Gemma 96–98 % токенов — рассуждения; "
          "видимый ответ — десятки токенов")
    legend(fig, models, names)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.76, bottom=0.1)
    return save(fig, out / "tokens_by_task.png")


def plot_temperature_sweep(a: dict[str, Any], out: Path) -> Path:
    models, names = a["models"], a["short_names"]
    panels = (
        ("acceptable", "Доля приемлемых ответов", True, (0, 1.05)),
        ("similarity", "Схожесть повторов (0–1)", False, (0, 1.05)),
        ("tokens_cv", "Разброс длины вывода, CV", False, None),
    )
    fig, axes = new_figure(12, 4.3, ncols=3)
    for ax, (key, label, is_share, ylim) in zip(axes, panels):
        style_axes(ax, grid_axis="both")
        for i, model in enumerate(models):
            points = a["temperature_sweep"][model]
            dodge = (i - 1) * 0.025  # совпадающие значения (Qwen и Gemma = 100 %) не должны скрывать друг друга
            ax.plot([p["temperature"] + dodge for p in points], [p[key] or 0.0 for p in points], color=SERIES_COLORS[i],
                    marker=SERIES_MARKERS[i], markersize=7, linewidth=2, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, solid_capstyle="round", zorder=3)
        ax.set_title(label, loc="left", fontsize=10, color=INK, pad=8)
        ax.set_xticks([0, 0.3, 0.7, 1.15], ["0", "0.3", "0.7", "1.15"])
        ax.set_xlabel("температура (точки слегка разнесены)" if key == "acceptable"
                      else "температура", color=INK_MUTED, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        if is_share:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    title(fig, "Температурный свип: пресеты B → D → E → C",
          "B (T=0) также сужает выборку (top_p 0.5, top_k 20), C (T=1.15) добавляет штрафы presence/frequency; "
          "D и E различаются только температурой")
    legend(fig, models, names, y=0.855)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.69, bottom=0.14, wspace=0.22)
    return save(fig, out / "temperature_sweep.png")


def plot_latency_vs_quality(results_root: Path, a: dict[str, Any], out: Path) -> Path:
    by_model = load_all(results_root)
    critical = load_critical_issues(results_root)
    models, names = a["models"], a["short_names"]
    fig, ax = new_figure(10, 5.0)
    style_axes(ax, grid_axis="both")
    for i, model in enumerate(models):
        xs, ys = [], []
        for preset in PRESET_ORDER:
            if preset == LENGTH_LIMITED_PRESET:
                continue
            group = [r for r in by_model[model] if r["preset_key"] == preset]
            xs.append(statistics.median(r["metrics"]["latency_s"] for r in group))
            ys.append(sum(1 for r in group if is_acceptable(r, critical)) / len(group))
        ax.scatter(xs, ys, s=80, color=SERIES_COLORS[i], marker=SERIES_MARKERS[i], edgecolors=SURFACE,
                   linewidths=1.5, zorder=3)
        label_at = LABEL_ANCHORS.get(i, "right")
        k = max(range(len(xs)), key=lambda j: xs[j]) if label_at == "right" else min(range(len(xs)), key=lambda j: xs[j])
        ax.annotate(names[model], (xs[k], ys[k]), xytext=(10, -16) if label_at == "right" else (-8, -18),
                    ha="left" if label_at == "right" else "right", textcoords="offset points",
                    fontsize=9.5, color=INK_SECONDARY)
        if i == len(models) - 1:  # у Ministral разброс по пресетам — подписываем крайние точки
            presets = [p for p in PRESET_ORDER if p != LENGTH_LIMITED_PRESET]
            for preset, note in (("B_precise", "T = 0"), ("C_creative", "T = 1.15")):
                j = presets.index(preset)
                ax.annotate(note, (xs[j], ys[j]), xytext=(9, -3), textcoords="offset points",
                            fontsize=8.5, color=INK_MUTED)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} с"))
    ax.set_xlim(1.5, 250)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("медианная задержка запроса (лог. шкала)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("доля приемлемых ответов", color=INK_MUTED, fontsize=9)
    title(fig, "Цена качества: задержка против доли приемлемых ответов",
          "Точка — пресет A–F (18 прогонов). По медиане Gemma в 9 раз, Qwen в 17 раз медленнее Ministral — и почти не ошибаются")
    legend(fig, models, names)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.12)
    return save(fig, out / "latency_vs_quality.png")


def plot_reasoning_word_limits(a: dict[str, Any], out: Path) -> Path:
    effect = a["word_limit_effect"]
    models = [m for m in a["models"] if m in effect]
    colors = {m: SERIES_COLORS[a["models"].index(m)] for m in models}
    names = a["short_names"]
    fig, ax = new_figure(10, 4.8)
    style_axes(ax)
    width = 0.34
    for i, model in enumerate(models):
        values = [effect[model][pid]["reasoning_median"] for pid in PROMPT_ORDER]
        xs = [x + (i - 0.5) * width for x in range(len(PROMPT_ORDER))]
        ax.bar(xs, values, width=width, color=colors[model], edgecolor=SURFACE, linewidth=1.2)
    ax.axhline(REASONING_BUDGET, color=INK_MUTED, linewidth=1)
    ax.text(len(PROMPT_ORDER) - 0.5, REASONING_BUDGET + 120, "бюджет рассуждений ≈ 8 192",
            ha="right", va="bottom", fontsize=8.5, color=INK_MUTED)
    labels = [
        f"{PROMPT_LABELS[p]}\n" + (f"лимит {WORD_LIMITS[p]} слов" if WORD_LIMITS[p] else "без лимита слов")
        for p in PROMPT_ORDER
    ]
    ax.set_xticks(range(len(PROMPT_ORDER)), labels, color=INK_SECONDARY, fontsize=8.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    ax.set_ylim(0, 9200)
    ax.set_ylabel("медиана токенов рассуждений", color=INK_MUTED, fontsize=9)
    title(fig, "Где в задании есть лимит слов, рассуждения длиннее",
          "В трассах к таким промптам модели пересчитывают слова (Qwen — 94–100 % трасс); "
          "на P4 Gemma упирается в бюджет рассуждений в 10 из 18 прогонов")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[m]) for m in models]
    fig.legend(handles, [names[m] for m in models], loc="upper left", bbox_to_anchor=(0.005, 0.84),
               ncol=len(models), frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.76, bottom=0.15)
    return save(fig, out / "reasoning_word_limits.png")


def render_all(results_root: Path) -> list[Path]:
    """Все графики отчёта в results/figures/. Без matplotlib — RuntimeError с подсказкой."""
    if plt is None:
        raise RuntimeError("Нужен matplotlib: pip install -r source/requirements.txt")
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    analysis = build_analysis(results_root)
    out = results_root / FIGURES_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    return [
        plot_acceptability_by_task(analysis, out),
        plot_tokens_by_task(results_root, analysis, out),
        plot_temperature_sweep(analysis, out),
        plot_latency_vs_quality(results_root, analysis, out),
        plot_reasoning_word_limits(analysis, out),
    ]
