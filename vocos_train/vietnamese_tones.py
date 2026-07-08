from __future__ import annotations

import re
import unicodedata


TONE_NAMES = ["ngang", "sac", "huyen", "hoi", "nga", "nang"]

_TONE_MARKS = {
    "\u0301": 1,  # acute
    "\u0300": 2,  # grave
    "\u0309": 3,  # hook above
    "\u0303": 4,  # tilde
    "\u0323": 5,  # dot below
}

_WORD_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def tone_id(text: str) -> int:
    """Return Vietnamese tone id for one syllable/word using Unicode combining marks."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    for char in decomposed:
        if char in _TONE_MARKS:
            return _TONE_MARKS[char]
    return 0


def strip_tone(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    without_tone = "".join(char for char in decomposed if char not in _TONE_MARKS)
    return unicodedata.normalize("NFC", without_tone)


def iter_vietnamese_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def utterance_tone_histogram(text: str) -> list[int]:
    hist = [0] * len(TONE_NAMES)
    for word in iter_vietnamese_words(text):
        hist[tone_id(word)] += 1
    return hist
