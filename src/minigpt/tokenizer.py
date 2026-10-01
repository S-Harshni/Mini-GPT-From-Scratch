"""Byte-pair encoding, trained from scratch.

Text is split into words (a word keeps its leading space), each word starts as a sequence of bytes,
and the most frequent adjacent pair is merged into a new token until the vocabulary is full. Encoding
replays the merges in the order they were learned, so decode(encode(text)) == text for any input.
"""
import json
import re
from collections import Counter
from pathlib import Path

WORD = re.compile(r"\s*\S+|\s+$")


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = merges or []
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        words = Counter(WORD.findall(text))
        splits = {w: list(w.encode("utf-8")) for w in words}
        merges: list[tuple[int, int]] = []
        while 256 + len(merges) < vocab_size:
            pairs: Counter = Counter()
            for w, ids in splits.items():
                for pair in zip(ids, ids[1:], strict=False):
                    pairs[pair] += words[w]
            if not pairs:
                break
            best = max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))  # deterministic tie-break
            new = 256 + len(merges)
            merges.append(best)
            for w, ids in splits.items():
                splits[w] = _merge(ids, best, new)
        return cls(merges)

    def _encode_word(self, word: str) -> list[int]:
        if word not in self._cache:
            ids = list(word.encode("utf-8"))
            while len(ids) > 1:
                pair = min(zip(ids, ids[1:], strict=False), key=lambda p: self.rank.get(p, float("inf")))
                if pair not in self.rank:
                    break
                ids = _merge(ids, pair, 256 + self.rank[pair])
            self._cache[word] = ids
        return self._cache[word]

    def encode(self, text: str) -> list[int]:
        return [t for word in WORD.findall(text) for t in self._encode_word(word)]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"merges": self.merges}))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        return cls([tuple(m) for m in json.loads(path.read_text())["merges"]])


def _merge(ids: list[int], pair: tuple[int, int], new: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
