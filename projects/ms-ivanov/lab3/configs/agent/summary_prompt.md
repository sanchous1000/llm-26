You compress the earlier part of a conversation between a user and a research assistant so that the assistant
can continue without the full transcript. Write a compact summary (at most {max_tokens} tokens) in English with
these sections:

USER FACTS: everything the user stated about themselves or their setup (numbers, dimensions, costs, choices),
corrections the user made, and their stated goals. Copy numbers exactly.
FINDINGS: for every question, the answer that was given: paper (authors, year, title or doc_key), result number
(Theorem/Proposition/Definition), the exact formula or statement in LaTeX, and verdicts given. Keep the order in
which topics were discussed and mark the first topic as such.
OPEN POINTS: what was not found or remains unresolved.

Do not add anything that is not in the conversation. Previous summary (if any) must be merged, not dropped.

Conversation to summarise:
{transcript}
