"""Retrieval layer (the "R" in RAG).

A small curated health-tip knowledge base is indexed with TF-IDF vectors.
For each patient we retrieve the top-k tips for their condition, ranked by
cosine similarity to the patient's notes. Retrieval is pure Python: no extra
API call, no embedding latency, deterministic, and easy to unit test.
Swapping in dense embeddings only needs a new `retrieve` implementation.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

GENERAL_KEY = "general"

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or the to with you your "
    "please we our if do not can will this that any when even".split()
)


def _stem(token: str) -> str:
    """Very light stemmer so 'doses/dose', 'taking/take' land on the same token."""
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        token = token[:-1]
    if token.endswith("ing") and len(token) > 5:
        token = token[:-3]
    if token.endswith("e") and len(token) > 3:
        token = token[:-1]
    return token


_ABBREVIATIONS = (
    (r"\bbp\b", "blood pressure"), (r"\bmeds?\b", "medicine"), (r"\bmins?\b", "minutes"),
    (r"\bhrs?\b", "hours"), (r"\bphysio\b", "physiotherapist"), (r"\bhb\b", "haemoglobin"),
    (r"\btablets?\b", "medicine"), (r"\bpills?\b", "medicine"),
)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    for pattern, replacement in _ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)
    words = re.findall(r"[a-z]+", text)
    return [_stem(w) for w in words if w not in _STOPWORDS and len(w) > 1]


@dataclass(frozen=True)
class Tip:
    id: str
    condition: str
    kind: str  # "tip" or "reminder"
    text: str


def tip_coverage(message: str, tip: Tip | str) -> float:
    """Share of a tip's content words that appear in the message (0..1)."""
    tip_text = tip.text if isinstance(tip, Tip) else tip
    tip_tokens = set(tokenize(tip_text))
    if not tip_tokens:
        return 1.0
    return len(tip_tokens & set(tokenize(message))) / len(tip_tokens)


class KnowledgeBase:
    def __init__(self, data: dict[str, dict]):
        if GENERAL_KEY not in data:
            raise ValueError("Knowledge base must define a 'general' entry.")
        self._entries = data
        self._tips: list[Tip] = []
        for key, entry in data.items():
            for i, tip in enumerate(entry["tips"], start=1):
                self._tips.append(Tip(f"{key}-{i}", key, tip.get("kind", "tip"), tip["text"]))

        tokens = [tokenize(t.text) for t in self._tips]
        doc_freq: Counter[str] = Counter()
        for toks in tokens:
            doc_freq.update(set(toks))
        n = len(self._tips)
        self._idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in doc_freq.items()}
        self._vectors = [self._vectorize(t) for t in tokens]

        aliases: list[tuple[str, str]] = []
        for key, entry in data.items():
            for alias in [key.replace("_", " "), entry["label"], *entry.get("aliases", [])]:
                norm = self._normalize(alias)
                if norm:
                    aliases.append((norm, key))
        # Longest alias first so "type 2 diabetes" beats "diabetes", "knee injury" beats "knee".
        aliases.sort(key=lambda pair: len(pair[0]), reverse=True)
        self._aliases = [(re.compile(rf"\b{re.escape(a)}\b"), k) for a, k in aliases]

    @classmethod
    def from_file(cls, path: Path | str) -> KnowledgeBase:
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    # ---- condition handling -------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()

    def resolve(self, text: str) -> str:
        """Map free-text ('BP', 'type 2 diabetes mellitus') to a knowledge-base key."""
        norm = self._normalize(text or "")
        if not norm:
            return GENERAL_KEY
        for pattern, key in self._aliases:
            if pattern.search(norm):
                return key
        return GENERAL_KEY

    def label(self, key: str) -> str:
        return self._entries.get(key, self._entries[GENERAL_KEY])["label"]

    def follow_up_days(self, key: str) -> int:
        return int(self._entries.get(key, self._entries[GENERAL_KEY]).get("follow_up_days", 7))

    @property
    def conditions(self) -> list[str]:
        return list(self._entries)

    # ---- retrieval ----------------------------------------------------------
    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(t for t in tokens if t in self._idf)
        vec = {w: c * self._idf[w] for w, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {w: v / norm for w, v in vec.items()} if norm else {}

    def retrieve(self, condition_key: str, query: str = "", top_k: int = 2) -> list[Tip]:
        """Top-k tips for a condition, ranked by similarity to `query`.

        Always returns at least one tip: if the condition has fewer than `top_k`
        tips (or is unknown) the general tips fill the remainder.
        """
        key = condition_key if condition_key in self._entries else GENERAL_KEY
        pool = [i for i, t in enumerate(self._tips) if t.condition == key]
        if len(pool) < top_k and key != GENERAL_KEY:
            pool += [i for i, t in enumerate(self._tips) if t.condition == GENERAL_KEY]
        query_vec = self._vectorize(tokenize(query))

        def score(position_and_index: tuple[int, int]) -> tuple[float, int]:
            position, idx = position_and_index
            vec = self._vectors[idx]
            cosine = sum(w * vec.get(t, 0.0) for t, w in query_vec.items())
            return (-cosine, position)  # best similarity first, ties keep file order

        ranked = sorted(enumerate(pool), key=score)
        return [self._tips[idx] for _, idx in ranked[:top_k]]


@lru_cache(maxsize=4)
def load_knowledge_base(path: str) -> KnowledgeBase:
    return KnowledgeBase.from_file(path)
