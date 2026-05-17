from __future__ import annotations

import re


def tokenize_words(text: str, *, min_length: int = 1) -> list[str]:
    """Tokenize text into lowercase Unicode words."""
    min_length = max(1, int(min_length or 1))
    tokens = re.findall(r"[^\W_]+", (text or "").lower())
    return [token for token in tokens if len(token) >= min_length]
