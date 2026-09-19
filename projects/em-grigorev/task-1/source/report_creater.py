import html
import pandas as pd


def make_html_report(df: pd.DataFrame, output_file="ollama_report.html"):
    rows = []

    for _, row in df.iterrows():
        ans = row["ans"]
        params = row["params"]

        generation_time = ans.get("generation_time_sec", 0)
        response_tokens = ans.get("response_length_tokens", 0)

        tokens_per_sec = (
            response_tokens / generation_time
            if generation_time > 0
            else 0
        )

        rows.append({
            "model": row["model_name"],
            "prompt": row["prompt"],

            "temperature": params.get("temperature"),
            "repeat_penalty": params.get("repeat_penalty"),
            "top_p": params.get("top_p"),
            "top_k": params.get("top_k"),

            "response": ans.get("response", ""),

            "response_length_chars":
                ans.get("response_length_chars"),

            "response_length_tokens":
                response_tokens,

            "response_time_sec":
                ans.get("response_time_sec"),

            "generation_time_sec":
                generation_time,

            "load_time_sec":
                ans.get("load_time_sec"),

            "prompt_eval_time_sec":
                ans.get("prompt_eval_time_sec"),

            "tokens_per_sec":
                tokens_per_sec,
        })

    report = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def fmt(value, digits=2):
        if pd.isna(value):
            return "—"

        if isinstance(value, float):
            return f"{value:.{digits}f}"

        return str(value)

    def esc(value):
        return html.escape(str(value))

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = (
        report
        .groupby(
            [
                "model",
                "temperature",
                "repeat_penalty",
                "top_p",
                "top_k",
            ],
            dropna=False,
        )
        .agg(
            runs=("response", "count"),
            avg_response_time=("response_time_sec", "mean"),
            avg_generation_time=("generation_time_sec", "mean"),
            avg_load_time=("load_time_sec", "mean"),
            avg_tokens=("response_length_tokens", "mean"),
            avg_chars=("response_length_chars", "mean"),
            avg_tokens_sec=("tokens_per_sec", "mean"),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Summary HTML
    # ---------------------------------------------------------

    summary_rows = ""

    for _, row in summary.iterrows():

        summary_rows += f"""
        <tr>
            <td><b>{esc(row["model"])}</b></td>
            <td>{fmt(row["temperature"])}</td>
            <td>{fmt(row["repeat_penalty"])}</td>
            <td>{fmt(row["top_p"])}</td>
            <td>{fmt(row["top_k"])}</td>
            <td>{int(row["runs"])}</td>
            <td>{fmt(row["avg_response_time"])} s</td>
            <td>{fmt(row["avg_generation_time"])} s</td>
            <td>{fmt(row["avg_tokens"])}</td>
            <td>{fmt(row["avg_tokens_sec"])} tok/s</td>
        </tr>
        """

    # ---------------------------------------------------------
    # Detailed experiments
    # ---------------------------------------------------------

    experiment_cards = ""

    for i, row in report.iterrows():

        response = esc(row["response"]).replace("\n", "<br>")

        experiment_cards += f"""
        <details class="experiment">
            <summary>
                <span class="model">
                    {esc(row["model"])}
                </span>

                <span class="badge">
                    temp={fmt(row["temperature"])}
                </span>

                <span class="badge">
                    repeat={fmt(row["repeat_penalty"])}
                </span>

                <span class="speed">
                    {fmt(row["tokens_per_sec"])} tok/s
                </span>
            </summary>

            <div class="experiment-body">

                <div class="prompt">
                    <div class="label">PROMPT</div>
                    <div>{esc(row["prompt"])}</div>
                </div>

                <div class="metrics">

                    <div class="metric">
                        <div class="metric-value">
                            {fmt(row["response_time_sec"])} s
                        </div>
                        <div class="metric-label">
                            Response time
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-value">
                            {fmt(row["generation_time_sec"])} s
                        </div>
                        <div class="metric-label">
                            Generation time
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-value">
                            {fmt(row["response_length_tokens"], 0)}
                        </div>
                        <div class="metric-label">
                            Output tokens
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-value">
                            {fmt(row["response_length_chars"], 0)}
                        </div>
                        <div class="metric-label">
                            Characters
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-value">
                            {fmt(row["tokens_per_sec"])}
                        </div>
                        <div class="metric-label">
                            Tokens / sec
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-value">
                            {fmt(row["load_time_sec"])} s
                        </div>
                        <div class="metric-label">
                            Model load
                        </div>
                    </div>

                </div>

                <div class="params">
                    <span>temperature={fmt(row["temperature"])}</span>
                    <span>repeat_penalty={fmt(row["repeat_penalty"])}</span>
                    <span>top_p={fmt(row["top_p"])}</span>
                    <span>top_k={fmt(row["top_k"])}</span>
                </div>

                <div class="response">
                    <div class="label">RESPONSE</div>
                    <div class="response-text">
                        {response}
                    </div>
                </div>

            </div>
        </details>
        """

    # ---------------------------------------------------------
    # HTML
    # ---------------------------------------------------------

    html_document = f"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Ollama Experiments</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: #f5f6f8;
    color: #202124;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}}

.container {{
    max-width: 1500px;
    margin: 0 auto;
    padding: 32px;
}}

h1 {{
    margin-bottom: 6px;
}}

.subtitle {{
    color: #6b7280;
    margin-bottom: 30px;
}}

.section {{
    background: white;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow:
        0 2px 8px rgba(0,0,0,.05);
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}}

th {{
    text-align: left;
    background: #f3f4f6;
    padding: 12px;
    position: sticky;
    top: 0;
}}

td {{
    padding: 11px 12px;
    border-bottom: 1px solid #eee;
}}

tr:hover {{
    background: #fafafa;
}}

.experiment {{
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    margin-bottom: 10px;
    overflow: hidden;
    background: white;
}}

.experiment summary {{
    cursor: pointer;
    padding: 15px 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    list-style: none;
}}

.experiment summary::-webkit-details-marker {{
    display: none;
}}

.experiment summary:hover {{
    background: #f9fafb;
}}

.model {{
    font-weight: 600;
    min-width: 160px;
}}

.badge {{
    background: #eef2ff;
    color: #3730a3;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 12px;
}}

.speed {{
    margin-left: auto;
    font-weight: 600;
}}

.experiment-body {{
    border-top: 1px solid #eee;
    padding: 20px;
}}

.prompt {{
    background: #f8fafc;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 18px;
}}

.label {{
    font-size: 11px;
    font-weight: 700;
    color: #6b7280;
    margin-bottom: 8px;
    letter-spacing: .08em;
}}

.metrics {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin-bottom: 18px;
}}

.metric {{
    background: #f8fafc;
    border-radius: 10px;
    padding: 14px;
}}

.metric-value {{
    font-size: 20px;
    font-weight: 600;
}}

.metric-label {{
    font-size: 12px;
    color: #6b7280;
    margin-top: 4px;
}}

.params {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 18px;
}}

.params span {{
    background: #f3f4f6;
    padding: 5px 9px;
    border-radius: 6px;
    font-family: monospace;
    font-size: 12px;
}}

.response {{
    border-top: 1px solid #eee;
    padding-top: 18px;
}}

.response-text {{
    line-height: 1.6;
    white-space: normal;
}}

@media (max-width: 800px) {{

    .container {{
        padding: 15px;
    }}

    .section {{
        padding: 15px;
    }}

    .experiment summary {{
        flex-wrap: wrap;
    }}

    .speed {{
        margin-left: 0;
        width: 100%;
    }}

}}

</style>

</head>

<body>

<div class="container">

    <h1>Ollama Experiments</h1>

    <div class="subtitle">
        Model generation benchmark and parameter comparison
    </div>

    <div class="section">

        <h2>Summary</h2>

        <div style="overflow-x:auto">

        <table>

            <thead>
                <tr>
                    <th>Model</th>
                    <th>Temp.</th>
                    <th>Repeat</th>
                    <th>Top P</th>
                    <th>Top K</th>
                    <th>Runs</th>
                    <th>Response</th>
                    <th>Generation</th>
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

    <div class="section">

        <h2>Experiments</h2>

        {experiment_cards}

    </div>

</div>

</body>
</html>
"""

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(html_document)

    print(f"Report saved to: {output_file}")


# =============================================================
# Использование
# =============================================================
df = pd.read_parquet('report.parquet')
make_html_report(df)
