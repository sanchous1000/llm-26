from __future__ import annotations

SYSTEM_PROMPT = (
    "Ты отвечаешь на вопросы по предоставленным фрагментам документов.\n"
    "Правила:\n"
    "1. Используй только информацию из фрагментов, ничего не выдумывай.\n"
    "2. После каждого утверждения указывай номер фрагмента в квадратных скобках, например [1] или [2][3].\n"
    "3. Если во фрагментах нет ответа, прямо скажи: «В документах нет ответа на этот вопрос.»\n"
    "4. Отвечай по-русски, кратко и по существу."
)

NO_ANSWER = "В документах нет ответа на этот вопрос."


def strip_section_prefix(text: str, section: str | None) -> str:
    if section and text.startswith(f"{section}\n\n"):
        return text[len(section) + 2 :]
    return text


def format_context(blocks: list[dict]) -> str:
    parts = []
    for block in blocks:
        section = block.get("section")
        header = f"[{block['n']}] {block['source']}, стр. {block['page']}"
        if section:
            header += f", раздел: {section}"
        parts.append(f"{header}\n{strip_section_prefix(block['text'], section)}")
    return "\n\n".join(parts)


def build_messages(question: str, blocks: list[dict]) -> list[dict]:
    user = f"Фрагменты документов:\n\n{format_context(blocks)}\n\nВопрос: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
