"""Two chunking strategies, compared in the evaluation.

1. FixedWindowChunker (baseline): a sliding window of N words with overlap. Ignores structure,
   so it can cut a sentence, a table row or a section in half.
2. StructureChunker: respects the document's own structure. It splits on headings, packs whole
   paragraphs (or whole records for CSV/JSON) up to a size limit, splits oversized paragraphs at
   sentence boundaries, and remembers the section heading so each chunk carries its context.

Sizes are measured in words (a cheap, explainable proxy for tokens; ~1.3 tokens per word for
English, so 220 words stay well under the embedding model's 512-token limit).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class ChunkSpan:
    text: str
    section: str
    position: int


def _strip_heading(line: str) -> str:
    return line.lstrip("#").strip()


class FixedWindowChunker:
    name = "fixed"

    def __init__(self, max_words: int = 220, overlap_words: int = 40):
        if overlap_words >= max_words:
            raise ValueError("overlap must be smaller than the window size")
        self.max_words = max_words
        self.overlap_words = overlap_words

    def split(self, text: str, record_based: bool = False) -> list[ChunkSpan]:
        words = text.replace("#", " ").split()
        if not words:
            return []
        step = self.max_words - self.overlap_words
        spans = []
        for pos, start in enumerate(range(0, max(len(words) - self.overlap_words, 1), step)):
            window = words[start:start + self.max_words]
            spans.append(ChunkSpan(text=" ".join(window), section="", position=pos))
        return spans


class StructureChunker:
    name = "structure"

    def __init__(self, max_words: int = 220, min_words: int = 25, min_section_words: int = 80):
        self.max_words = max_words
        self.min_words = min_words
        # A new section starts a new chunk only if the current chunk already has this many words;
        # otherwise tiny sections ("Escalation: page on-call") are merged with their neighbours.
        self.min_section_words = min_section_words

    def _split_long(self, paragraph: str) -> list[str]:
        """Split an oversized paragraph at sentence boundaries; hard-split giant sentences."""
        pieces: list[str] = []
        current: list[str] = []
        count = 0
        for sentence in SENTENCE_SPLIT_RE.split(paragraph):
            words = sentence.split()
            while len(words) > self.max_words:  # a single "sentence" longer than the limit
                pieces.append(" ".join(words[: self.max_words]))
                words = words[self.max_words:]
            if count + len(words) > self.max_words and current:
                pieces.append(" ".join(current))
                current, count = [], 0
            current.extend(words)
            count += len(words)
        if current:
            pieces.append(" ".join(current))
        return pieces

    def split(self, text: str, record_based: bool = False) -> list[ChunkSpan]:
        # 1) Turn the text into (section, paragraph) units.
        units: list[tuple[str, str]] = []
        section = ""
        for block in re.split(r"\n\s*\n", text):
            block = block.strip()
            if not block:
                continue
            if block.startswith("#") and "\n" not in block:
                section = _strip_heading(block)
                continue
            if block.startswith("#"):  # heading followed directly by text on the next line
                first, rest = block.split("\n", 1)
                section = _strip_heading(first)
                block = rest.strip()
            for piece in (self._split_long(block) if len(block.split()) > self.max_words else [block]):
                units.append((section, piece))

        # 2) Greedily pack units of the same section up to max_words.
        spans: list[ChunkSpan] = []
        buf: list[str] = []
        buf_words = 0
        buf_section = None

        def flush() -> None:
            nonlocal buf, buf_words
            if buf:
                spans.append(ChunkSpan(text="\n".join(buf), section=buf_section or "", position=len(spans)))
            buf, buf_words = [], 0

        last_section = None
        for sec, unit in units:
            n = len(unit.split())
            section_changed = last_section is not None and sec != last_section and not record_based
            if buf and (buf_words + n > self.max_words or (section_changed and buf_words >= self.min_section_words)):
                flush()
            if not buf:
                buf_section = sec
            elif section_changed and sec:
                buf.append(f"{sec}:")  # keep the heading of a merged section as context
            buf.append(unit)
            buf_words += n
            last_section = sec
        flush()

        # 3) Merge a tiny trailing chunk into its predecessor when they share a section.
        if len(spans) >= 2 and len(spans[-1].text.split()) < self.min_words:
            last, prev = spans[-1], spans[-2]
            if last.section == prev.section and len((prev.text + last.text).split()) <= int(self.max_words * 1.2):
                prev.text = prev.text + "\n" + last.text
                spans.pop()
        return spans


def get_chunker(strategy: str, max_words: int = 220, overlap_words: int = 40):
    if strategy == "fixed":
        return FixedWindowChunker(max_words=max_words, overlap_words=overlap_words)
    if strategy == "structure":
        return StructureChunker(max_words=max_words)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
