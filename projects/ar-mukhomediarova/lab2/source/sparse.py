"""Разрежённые векторы BM25 для Qdrant.

Документ: вес термина = tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl)).
Запрос: вес 1 для каждого уникального термина.
IDF считает сам Qdrant (sparse-вектор с modifier=IDF), поэтому итоговый скор = BM25.
Термины — стеммы Snowball (русский/английский), id термина — crc32 от стема.
"""
from __future__ import annotations

import re
import zlib
from collections import Counter

import snowballstemmer

_RU = snowballstemmer.stemmer("russian")
_EN = snowballstemmer.stemmer("english")
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-_][A-Za-zА-Яа-яЁё0-9]+)*")
_CYR = re.compile(r"[А-Яа-яЁё]")

STOP = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было
вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь
опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам
чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь
этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой
хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою
этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это также который
которые которая которых является
the a an and or of to in on for with is are was were be been by as at from that this these those it its
into than then not no but if so such can could may might will would should do does did has have had
""".split())


def tokenize(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        if len(w) < 2 or w in STOP:
            continue
        if w.isdigit():
            out.append(w)
            continue
        out.append(_RU.stemWord(w) if _CYR.search(w) else _EN.stemWord(w))
    return out


def term_id(term: str) -> int:
    return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF


def _to_sparse(weights: dict[int, float]) -> tuple[list[int], list[float]]:
    idx = sorted(weights)
    return idx, [round(float(weights[i]), 5) for i in idx]


class BM25Encoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75, avgdl: float | None = None):
        self.k1, self.b, self.avgdl = k1, b, avgdl

    def fit(self, texts: list[str]) -> "BM25Encoder":
        lens = [len(tokenize(t)) for t in texts]
        self.avgdl = sum(lens) / max(len(lens), 1)
        return self

    def encode_doc(self, text: str) -> tuple[list[int], list[float]]:
        toks = tokenize(text)
        tf = Counter(toks)
        dl = len(toks)
        norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
        weights: dict[int, float] = {}
        for term, f in tf.items():
            tid = term_id(term)  # коллизии crc32 складываем
            weights[tid] = weights.get(tid, 0.0) + f * (self.k1 + 1) / (f + norm)
        return _to_sparse(weights)

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        return _to_sparse({term_id(t): 1.0 for t in set(tokenize(text))})

    def state(self) -> dict:
        return {"k1": self.k1, "b": self.b, "avgdl": self.avgdl}
