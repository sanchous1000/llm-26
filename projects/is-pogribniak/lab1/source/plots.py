"""Графики к отчёту."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODE_COLORS = {"A": "#2a78d6", "B": "#eb6834"}
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BAR_ERR = {"ecolor": TEXT_2, "elinewidth": 1, "capsize": 3}


def style_axes(ax, title: str, ylabel: str | None = None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_2)


def save(fig, title: str, path: Path):
    fig.suptitle(title, color=TEXT, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def by_prompt(summary: pd.DataFrame, metric: str, err: str | None, ylabel: str, title: str,
              path: Path, ylim=None):
    """Панель на промпт, внутри — модели, столбцы режимов A/B рядом."""
    prompts = sorted(summary["prompt_id"].unique())
    models = list(dict.fromkeys(summary["model"]))
    modes = [m for m in MODE_COLORS if m in set(summary["mode"])]
    width = 0.72 / len(modes)
    fig, axes = plt.subplots(1, len(prompts), figsize=(4.2 * len(prompts), 3.8), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax, pid in zip(axes[0], prompts):
        style_axes(ax, pid)
        sub = summary[summary["prompt_id"] == pid].set_index(["model", "mode"])
        for j, mode in enumerate(modes):
            xs = [i + (j - (len(modes) - 1) / 2) * (width + 0.03) for i in range(len(models))]
            ax.bar(xs, [sub.loc[(m, mode), metric] for m in models], width, color=MODE_COLORS[mode],
                   label=f"режим {mode}", yerr=[sub.loc[(m, mode), err] for m in models] if err else None,
                   error_kw=BAR_ERR)
        ax.set_xticks(range(len(models)), models, color=TEXT)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0][0].set_ylabel(ylabel, color=TEXT_2)
    axes[0][0].legend(frameon=False, labelcolor=TEXT)
    save(fig, title, path)


def speeds(speed: pd.DataFrame, path: Path):
    """Скорость генерации и обработки промпта по моделям — две панели."""
    panels = [("gen_tps", "Генерация")] + ([("prompt_tps", "Обработка промпта (prefill)")]
                                          if "prompt_tps" in speed else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.6), facecolor=SURFACE, squeeze=False)
    for ax, (col, title) in zip(axes[0], panels):
        style_axes(ax, title, "токенов/с")
        ax.bar(speed["model"], speed[col], 0.5, color="#2a78d6", yerr=speed[f"{col}_std"], error_kw=BAR_ERR)
        for i, (v, sd) in enumerate(zip(speed[col], speed[f"{col}_std"].fillna(0))):
            ax.text(i, v + sd, f"{v:.0f}", ha="center", va="bottom", color=TEXT, fontsize=9)
    save(fig, "Скорость, токенов в секунду (среднее ± σ)", path)


def draw_all(summary: pd.DataFrame, speed: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    by_prompt(summary, "completion_mean", "completion_std", "токенов",
              "Длина ответа (completion tokens), среднее ± σ", out / "tokens.png")
    by_prompt(summary, "quality_mean", "quality_std", "доля",
              "Качество под задачу, среднее ± σ", out / "quality.png", ylim=(0, 1.05))
    by_prompt(summary, "text_similarity", None, "сходство",
              "Стабильность: попарное сходство ответов", out / "stability.png", ylim=(0, 1.05))
    speeds(speed, out / "speed.png")
