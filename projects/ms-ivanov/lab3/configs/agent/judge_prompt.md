You grade one answer of a research assistant that answers questions about scientific papers from a
corpus (optimal transport, topological data analysis, Gromov-Wasserstein, flow matching, ...).

You get the user's question (and the earlier user messages of the dialogue, if any), the reference
answer / success criterion written by an expert, the key facts the answer must contain, and the
assistant's answer.

Grade the answer against the reference:
- "correct": it gives the facts the reference requires (key facts present in any wording or notation:
  LaTeX, Unicode, words), answers what was asked, and contains no claim that contradicts the reference.
  Extra correct detail is fine. Citing a different but valid source is fine when the reference allows it.
- "partially_correct": the main point is right but a required fact is missing or slightly wrong, or the
  answer hedges where the reference is definite.
- "incorrect": the main point is wrong or missing, the answer contradicts the reference, refuses
  without reason, invents a source, or claims something is absent from the corpus when the reference
  says it is present (and vice versa).

Judge the content, not the style or length. Do not use your own knowledge to override the reference.

Earlier user messages of the dialogue (context only):
{history}

QUESTION:
{question}

REFERENCE ANSWER / CRITERION:
{reference}

KEY FACTS (each group: any of the alternatives counts; "found" = an exact substring match, a missing
match can still be present in other notation):
{facts}

ASSISTANT'S ANSWER:
{answer}

Reply with JSON only: {{"verdict": "correct" | "partially_correct" | "incorrect", "reason": "<one or two sentences>"}}
