#!/usr/bin/env python3
"""HTML-отчёт по экспериментам Ollama.

Использование:
    python make_report.py report.parquet -o ollama_report.html

Для рендера markdown-ответов нужен пакет:
    pip install markdown-it-py
Без него ответы показываются как обычный текст.
"""

from __future__ import annotations

import argparse
import html
import numbers
import re

import pandas as pd

try:
    from markdown_it import MarkdownIt

    # html=False: сырой HTML из ответов модели экранируется (безопасно)
    _MD = MarkdownIt("commonmark", {"html": False, "breaks": True}).enable(
        ["table", "strikethrough"]
    )
except ImportError:
    _MD = None

PARAM_KEYS = ["temperature", "repeat_penalty", "top_p", "top_k"]

# <think>...</think> у reasoning-моделей (в т.ч. незакрытый, если ответ обрезан)
THINK_RE = re.compile(r"<think>(.*?)(?:</think>|\Z)", re.S | re.I)


# =============================================================
# Helpers
# =============================================================

def esc(value) -> str:
    return html.escape(str(value))


def fmt(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return f"{value:.{digits}f}"
    return str(value)


def data_v(value) -> str:
    """Значение для data-v: по нему таблица сортируется в браузере."""
    if value is None or pd.isna(value):
        return ""
    return esc(value)


def gid_label(group_id) -> str:
    return f"G{int(group_id):02d}"


def render_markdown(text: str) -> str:
    if _MD is not None:
        return _MD.render(text)
    return "<p>" + esc(text).replace("\n", "<br>") + "</p>"


def render_response(text) -> str:
    """Два представления ответа: отрендеренный markdown и сырой текст."""
    text = "" if text is None else str(text)

    if not text.strip():
        return '<div class="empty">empty response</div>'

    thoughts = [t.strip() for t in THINK_RE.findall(text) if t.strip()]
    answer = THINK_RE.sub("", text).strip()

    rendered = ""
    for thought in thoughts:
        rendered += (
            '<details class="think">'
            f"<summary>Reasoning ({len(thought)} chars)</summary>"
            f"{render_markdown(thought)}"
            "</details>"
        )
    rendered += render_markdown(answer)

    return (
        f'<div class="md">{rendered}</div>'
        f'<pre class="raw">{esc(text)}</pre>'
    )


# =============================================================
# Data
# =============================================================

def build_report_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()

    # Если в исходных данных уже есть experiment_id — используем его
    if "experiment_id" not in df.columns:
        df["experiment_id"] = [f"EXP-{i:05d}" for i in range(1, len(df) + 1)]

    rows = []

    for _, row in df.iterrows():
        ans = row["ans"]
        params = row["params"]

        generation_time = ans.get("generation_time_sec") or 0
        response_tokens = ans.get("response_length_tokens") or 0

        rows.append({
            "experiment_id": row["experiment_id"],
            "model": row["model_name"],
            "prompt": row["prompt"],
            "temperature": params.get("temperature"),
            "repeat_penalty": params.get("repeat_penalty"),
            "top_p": params.get("top_p"),
            "top_k": params.get("top_k"),
            "response": ans.get("response", ""),
            "response_length_chars": ans.get("response_length_chars"),
            "response_length_tokens": response_tokens,
            "response_time_sec": ans.get("response_time_sec"),
            "generation_time_sec": generation_time,
            "load_time_sec": ans.get("load_time_sec"),
            "prompt_eval_time_sec": ans.get("prompt_eval_time_sec"),
            # NaN, а не 0: такие запуски не занижают среднюю скорость
            "tokens_per_sec": (
                response_tokens / generation_time
                if generation_time > 0
                else float("nan")
            ),
        })

    report = pd.DataFrame(rows)

    # Сквозной индекс конфигурации (модель + параметры): G01, G02, ...
    report["group_id"] = (
        report.groupby(["model", *PARAM_KEYS], dropna=False).ngroup() + 1
    )

    return report


# =============================================================
# HTML blocks
# =============================================================

def build_summary_html(report: pd.DataFrame) -> str:
    summary = (
        report
        .groupby(["group_id", "model", *PARAM_KEYS], dropna=False)
        .agg(
            runs=("experiment_id", "count"),
            ids=("experiment_id", list),
            avg_response_time=("response_time_sec", "mean"),
            avg_generation_time=("generation_time_sec", "mean"),
            avg_load_time=("load_time_sec", "mean"),
            avg_tokens=("response_length_tokens", "mean"),
            avg_tokens_sec=("tokens_per_sec", "mean"),
        )
        .reset_index()
    )

    # (колонка, число знаков, суффикс)
    metric_cols = [
        ("avg_response_time", 2, " s"),
        ("avg_generation_time", 2, " s"),
        ("avg_load_time", 2, " s"),
        ("avg_tokens", 0, ""),
        ("avg_tokens_sec", 2, " tok/s"),
    ]

    rows = []

    for _, r in summary.iterrows():
        label = gid_label(r["group_id"])
        links = " ".join(
            f'<a class="chip" href="#{esc(i)}">{esc(i)}</a>' for i in r["ids"]
        )

        cells = [
            f'<td data-v="{int(r["group_id"])}">'
            f'<a class="gid" href="#experiments" data-filter="{label}">'
            f"{label}</a></td>",
            f'<td data-v="{esc(r["model"])}"><b>{esc(r["model"])}</b></td>',
        ]

        for key in PARAM_KEYS:
            cells.append(f'<td data-v="{data_v(r[key])}">{fmt(r[key])}</td>')

        cells.append(
            f'<td data-v="{int(r["runs"])}">'
            f'<details class="ids"><summary>{int(r["runs"])}</summary>'
            f"<div>{links}</div></details></td>"
        )

        for key, digits, suffix in metric_cols:
            value = r[key]
            text = "—" if pd.isna(value) else fmt(value, digits) + suffix
            cells.append(f'<td data-v="{data_v(value)}">{text}</td>')

        rows.append("<tr>" + "".join(cells) + "</tr>")

    return "\n".join(rows)


def build_cards_html(report: pd.DataFrame) -> str:
    metrics = [
        ("response_time_sec", "Response time", 2, " s"),
        ("generation_time_sec", "Generation time", 2, " s"),
        ("prompt_eval_time_sec", "Prompt eval", 2, " s"),
        ("response_length_tokens", "Output tokens", 0, ""),
        ("response_length_chars", "Characters", 0, ""),
        ("tokens_per_sec", "Tokens / sec", 2, ""),
        ("load_time_sec", "Model load", 2, " s"),
    ]

    cards = []

    for r in report.to_dict("records"):
        eid = esc(r["experiment_id"])
        label = gid_label(r["group_id"])

        search = " ".join([
            str(r["experiment_id"]),
            label,
            str(r["model"]),
            *(f"{k}={fmt(r[k])}" for k in PARAM_KEYS),
            str(r["prompt"]),
        ]).lower()

        metrics_html = "".join(
            '<div class="metric">'
            f'<div class="metric-value">{fmt(r[key], digits)}'
            f'{"" if pd.isna(r[key]) else suffix}</div>'
            f'<div class="metric-label">{title}</div>'
            "</div>"
            for key, title, digits, suffix in metrics
        )

        params_html = "".join(
            f"<span>{k}={fmt(r[k])}</span>" for k in PARAM_KEYS
        )

        cards.append(f"""
<details class="experiment" id="{eid}" data-search="{esc(search)}">
  <summary>
    <span class="experiment-id">{eid}</span>
    <a class="gid" href="#experiments" data-filter="{label}">{label}</a>
    <span class="model">{esc(r["model"])}</span>
    <span class="badge">temp={fmt(r["temperature"])}</span>
    <span class="badge">repeat={fmt(r["repeat_penalty"])}</span>
    <span class="speed">{fmt(r["tokens_per_sec"])} tok/s</span>
  </summary>

  <div class="experiment-body">
    <div class="prompt">
      <div class="label">PROMPT</div>
      <div class="prompt-text">{esc(r["prompt"])}</div>
    </div>

    <div class="metrics">{metrics_html}</div>

    <div class="params">{params_html}</div>

    <div class="response">
      <div class="label">RESPONSE</div>
      {render_response(r["response"])}
    </div>
  </div>
</details>
""")

    return "\n".join(cards)


# =============================================================
# Static assets
# =============================================================

CSS = """
:root {
    --bg: #f5f6f8; --card: #fff; --text: #202124; --muted: #6b7280;
    --line: #e5e7eb; --accent: #3730a3; --accent-bg: #eef2ff; --soft: #f8fafc;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.container { max-width: 1500px; margin: 0 auto; padding: 32px; }
h1 { margin-bottom: 6px; }
.subtitle { color: var(--muted); margin-bottom: 20px; }

.toolbar {
    position: sticky; top: 0; z-index: 10; background: var(--bg);
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    padding: 10px 0; margin-bottom: 14px;
}
.toolbar input[type=search] {
    flex: 1 1 280px; padding: 9px 12px; font-size: 14px;
    border: 1px solid var(--line); border-radius: 8px;
}
.toolbar button {
    padding: 8px 12px; font-size: 13px; cursor: pointer;
    background: var(--card); border: 1px solid var(--line); border-radius: 8px;
}
.toolbar button:hover { background: #f3f4f6; }
.toolbar label { font-size: 13px; color: var(--muted); user-select: none; }
#count { font-size: 13px; color: var(--muted); }

.section {
    background: var(--card); border-radius: 14px; padding: 24px;
    margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,.05);
}
.table-wrap { overflow: auto; max-height: 560px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th {
    text-align: left; background: #f3f4f6; padding: 12px;
    position: sticky; top: 0; white-space: nowrap;
}
table.sortable th { cursor: pointer; user-select: none; }
th[data-dir=asc]::after { content: " ▲"; font-size: 10px; }
th[data-dir=desc]::after { content: " ▼"; font-size: 10px; }
td { padding: 11px 12px; border-bottom: 1px solid #eee; vertical-align: top; }
tr:hover { background: #fafafa; }

a { color: var(--accent); }
.gid {
    font-family: var(--mono); font-size: 12px; font-weight: 700;
    color: var(--accent); background: var(--accent-bg);
    padding: 3px 7px; border-radius: 6px; text-decoration: none;
}
details.ids summary { cursor: pointer; list-style: none; }
details.ids summary::-webkit-details-marker { display: none; }
details.ids summary::before { content: "▸ "; }
details.ids[open] summary::before { content: "▾ "; }
details.ids div { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; max-width: 340px; }
.chip {
    font-family: var(--mono); font-size: 11px; color: #4b5563;
    background: #f3f4f6; padding: 2px 6px; border-radius: 5px; text-decoration: none;
}
.chip:hover { background: var(--accent-bg); color: var(--accent); }

.experiment {
    border: 1px solid var(--line); border-radius: 12px; margin-bottom: 10px;
    overflow: hidden; background: var(--card);
}
.experiment[hidden] { display: none; }
.experiment.flash { animation: flash 1.5s ease-out; }
@keyframes flash {
    from { box-shadow: 0 0 0 3px #a5b4fc; } to { box-shadow: 0 0 0 0 transparent; }
}
.experiment summary {
    cursor: pointer; padding: 15px 18px; display: flex;
    align-items: center; gap: 10px; list-style: none;
}
.experiment summary::-webkit-details-marker { display: none; }
.experiment summary:hover { background: #f9fafb; }
.experiment-id {
    font-family: var(--mono); font-size: 12px; font-weight: 700;
    color: #4b5563; background: #f3f4f6; padding: 4px 7px; border-radius: 6px;
}
.model { font-weight: 600; min-width: 160px; }
.badge {
    background: var(--accent-bg); color: var(--accent);
    padding: 4px 8px; border-radius: 6px; font-size: 12px;
}
.speed { margin-left: auto; font-weight: 600; }
.experiment-body { border-top: 1px solid #eee; padding: 20px; }

.prompt { background: var(--soft); border-radius: 8px; padding: 14px; margin-bottom: 18px; }
.prompt-text { white-space: pre-wrap; }
.label { font-size: 11px; font-weight: 700; color: var(--muted); margin-bottom: 8px; letter-spacing: .08em; }

.metrics {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px; margin-bottom: 18px;
}
.metric { background: var(--soft); border-radius: 10px; padding: 14px; }
.metric-value { font-size: 20px; font-weight: 600; }
.metric-label { font-size: 12px; color: var(--muted); margin-top: 4px; }

.params { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
.params span {
    background: #f3f4f6; padding: 5px 9px; border-radius: 6px;
    font-family: var(--mono); font-size: 12px;
}

.response { border-top: 1px solid #eee; padding-top: 18px; }
.empty { color: var(--muted); font-style: italic; }

/* rendered markdown */
.md { line-height: 1.6; overflow-wrap: anywhere; }
.md > :first-child { margin-top: 0; }
.md > :last-child { margin-bottom: 0; }
.md h1, .md h2, .md h3, .md h4 { margin: 1.2em 0 .5em; line-height: 1.3; }
.md h1 { font-size: 1.4em; } .md h2 { font-size: 1.25em; }
.md h3 { font-size: 1.1em; } .md h4 { font-size: 1em; }
.md code { font-family: var(--mono); font-size: .9em; background: #f3f4f6; padding: 2px 5px; border-radius: 4px; }
.md pre { background: #0f172a; color: #e2e8f0; padding: 14px; border-radius: 8px; overflow-x: auto; }
.md pre code { background: none; padding: 0; color: inherit; }
.md blockquote { margin: 1em 0; padding: .2em 1em; border-left: 4px solid #d1d5db; color: #4b5563; }
.md table { width: auto; margin: 1em 0; }
.md th, .md td { border: 1px solid var(--line); padding: 6px 10px; }
.md th { position: static; }
.md img { max-width: 100%; }
.think {
    background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px;
    padding: 8px 12px; margin-bottom: 14px; color: #78350f; font-size: 14px;
}
.think summary { cursor: pointer; font-weight: 600; }

/* raw view */
.raw {
    display: none; margin: 0; padding: 14px; border-radius: 8px;
    background: var(--soft); font-family: var(--mono); font-size: 13px;
    white-space: pre-wrap; overflow-wrap: anywhere;
}
body.show-raw .md { display: none; }
body.show-raw .raw { display: block; }

@media (max-width: 800px) {
    .container { padding: 15px; }
    .section { padding: 15px; }
    .experiment summary { flex-wrap: wrap; }
    .speed { margin-left: 0; width: 100%; }
}
"""

JS = """
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  const cards = $$('details.experiment');
  const input = $('#filter');
  const counter = $('#count');

  // ---- filter ------------------------------------------------
  function applyFilter() {
    const terms = input.value.trim().toLowerCase().split(/\\s+/).filter(Boolean);
    let shown = 0;
    cards.forEach(card => {
      const ok = terms.every(t => card.dataset.search.includes(t));
      card.hidden = !ok;
      if (ok) shown++;
    });
    counter.textContent = shown + ' / ' + cards.length;
  }
  input.addEventListener('input', applyFilter);

  // ---- open experiment by link / hash ------------------------
  function showCard(card) {
    if (card.hidden) { input.value = ''; applyFilter(); }
    card.open = true;
    card.scrollIntoView({ block: 'start' });
    card.classList.add('flash');
    setTimeout(() => card.classList.remove('flash'), 1500);
  }

  document.addEventListener('click', e => {
    const chip = e.target.closest('[data-filter]');
    if (chip) {
      e.preventDefault();
      input.value = chip.dataset.filter;
      applyFilter();
      $('#experiments').scrollIntoView();
      return;
    }
    const link = e.target.closest('a[href^="#EXP"], a.chip');
    if (link) {
      const card = document.getElementById(decodeURIComponent(link.hash.slice(1)));
      if (card && card.matches('details.experiment')) {
        e.preventDefault();
        history.replaceState(null, '', link.hash);
        showCard(card);
      }
    }
  });

  function openFromHash() {
    const id = decodeURIComponent(location.hash.slice(1));
    const card = id && document.getElementById(id);
    if (card && card.matches('details.experiment')) showCard(card);
  }
  window.addEventListener('hashchange', openFromHash);
  openFromHash();

  // ---- toolbar -----------------------------------------------
  $('#expand').addEventListener('click', () =>
    cards.forEach(c => { if (!c.hidden) c.open = true; }));
  $('#collapse').addEventListener('click', () =>
    cards.forEach(c => { c.open = false; }));
  $('#render').addEventListener('change', e =>
    document.body.classList.toggle('show-raw', !e.target.checked));

  // ---- sortable tables ---------------------------------------
  $$('table.sortable').forEach(table => {
    const ths = $$('th', table);
    ths.forEach((th, i) => th.addEventListener('click', () => {
      const dir = th.dataset.dir === 'asc' ? 'desc' : 'asc';
      ths.forEach(t => delete t.dataset.dir);
      th.dataset.dir = dir;

      const key = tr => {
        const cell = tr.children[i];
        const v = (cell.dataset.v ?? cell.textContent).trim();
        if (v === '') return null;
        const n = Number(v);
        return Number.isNaN(n) ? v.toLowerCase() : n;
      };
      const sign = dir === 'asc' ? 1 : -1;
      const tbody = $('tbody', table);
      $$('tr', tbody)
        .sort((a, b) => {
          const x = key(a), y = key(b);
          if (x === y) return 0;
          if (x === null) return 1;   // пустые значения всегда в конце
          if (y === null) return -1;
          return (x > y ? 1 : -1) * sign;
        })
        .forEach(tr => tbody.appendChild(tr));
    }));
  });

  applyFilter();
})();
"""


# =============================================================
# Report
# =============================================================

def make_html_report(df: pd.DataFrame, output_file="ollama_report.html"):
    if _MD is None:
        print(
            "WARNING: markdown-it-py не установлен — markdown не будет "
            "отрендерен (pip install markdown-it-py)"
        )

    report = build_report_frame(df)

    summary_rows = build_summary_html(report)
    experiment_cards = build_cards_html(report)

    n_exp = len(report)
    n_groups = report["group_id"].nunique()
    n_models = report["model"].nunique()

    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ollama Experiments</title>
<style>{CSS}</style>
</head>
<body>

<div class="container">

  <h1>Ollama Experiments</h1>
  <div class="subtitle">
    Model generation benchmark and parameter comparison ·
    {n_exp} experiments · {n_groups} configurations · {n_models} models
  </div>

  <div class="toolbar">
    <input id="filter" type="search"
           placeholder="Filter: model, EXP-00012, G03, temp=0.70, words from the prompt…">
    <span id="count"></span>
    <button id="expand" type="button">Expand all</button>
    <button id="collapse" type="button">Collapse all</button>
    <label><input id="render" type="checkbox" checked> Render markdown</label>
  </div>

  <div class="section">
    <h2>Summary</h2>
    <div class="table-wrap">
      <table class="sortable">
        <thead>
          <tr>
            <th>#</th>
            <th>Model</th>
            <th>Temp.</th>
            <th>Repeat</th>
            <th>Top P</th>
            <th>Top K</th>
            <th>Runs</th>
            <th>Response</th>
            <th>Generation</th>
            <th>Load</th>
            <th>Tokens</th>
            <th>Speed</th>
          </tr>
        </thead>
        <tbody>
{summary_rows}
        </tbody>
      </table>
    </div>
  </div>

  <div class="section" id="experiments">
    <h2>Experiments</h2>
{experiment_cards}
  </div>

</div>

<script>{JS}</script>
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_document)

    print(f"Report saved to: {output_file}")


# =============================================================
# CLI
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTML-отчёт по экспериментам Ollama")
    parser.add_argument("input", nargs="?", default="report.parquet")
    parser.add_argument("-o", "--output", default="ollama_report.html")
    args = parser.parse_args()

    make_html_report(pd.read_parquet(args.input), args.output)