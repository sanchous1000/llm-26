"""Prompt for generating candidate evaluation questions."""

QUESTION_TYPES = (
    "definition",
    "method",
    "theory",
    "result",
    "comparison",
    "motivation",
)

_TEMPLATE = """\
You are building an evaluation set for a retrieval-augmented question answering \
assistant over scientific papers (optimal transport, topological data analysis, \
generative models, representation learning). The attached file is {scope} of one \
document in Markdown (math is LaTeX).

Write exactly {count} questions that a researcher or a graduate student could ask \
such an assistant, so that each question is answered by ONE specific contiguous \
passage of the document.

Rules for every question:
1. Self-contained: it must be understandable without having seen the document. Name \
the concept, method, dataset or quantity explicitly. Never write "this paper", \
"the authors", "the proposed method", "Section 3", "Theorem 2", "Figure 1", \
"the above", "in the text".
2. Specific and non-trivial: not answerable from the title or abstract alone and not \
a yes/no question. Prefer asking for definitions, assumptions, algorithm steps, \
formulas, numerical results, conditions of a theorem, limitations, or the reason \
behind a design choice.
3. Cover different parts of the document and different types: {types}.
4. "evidence": copy VERBATIM, character by character, one contiguous passage of \
{min_chars}-{max_chars} characters from the document that contains the full answer \
(1-6 sentences or one definition/theorem/algorithm block). Keep LaTeX and \
formatting exactly as in the file. No paraphrase, no "...", no merging of separated \
passages, do not cross a heading.
5. "answer": a concise reference answer (1-3 sentences) that uses only the evidence.
6. "question_ru": the same question translated into Russian (keep technical terms \
and formulas).

Answer with ONLY a JSON array of {count} objects with the keys "question", \
"question_ru", "type", "answer", "evidence". No code fences, no comments. Do not \
use any tools.
"""


def build_prompt(
    *, count: int, part: int, parts: int, min_chars: int = 200, max_chars: int = 1200
) -> str:
    """Instructions for one request (one document part)."""
    scope = "the full text" if parts == 1 else f"part {part} of {parts}"
    return _TEMPLATE.format(
        count=count,
        scope=scope,
        types=", ".join(QUESTION_TYPES),
        min_chars=min_chars,
        max_chars=max_chars,
    )
