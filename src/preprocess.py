"""Text cleaning shared by training and inference.

Both pipelines must import clean_text from here, so the model sees text cleaned
the same way in the dashboard as it did during training.
"""

import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w.\-]+")
SPACE_RE = re.compile(r"\s+")
HANGUL_RE = re.compile(r"[가-힣ㄱ-ㆎ]")
LATIN_RE = re.compile(r"[A-Za-z]")


def clean_text(text):
    """Light cleaning: replace links and @mentions with placeholders, collapse whitespace.

    Emojis, casing and punctuation are kept on purpose; they carry tone
    ("great..." vs "GREAT!!!"), and transformer tokenizers handle them.
    The "http" and "@user" placeholders match the Cardiff NLP sentiment model's preprocessing.
    """
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub("http", text)
    text = MENTION_RE.sub("@user", text)
    return SPACE_RE.sub(" ", text).strip()


def detect_language(text):
    """Very small language check: "ko" if the text is mostly Hangul, else "en".

    Good enough to route a comment to the Korean or English model. Comments with
    neither script (only emojis or numbers) count as English.
    """
    hangul = len(HANGUL_RE.findall(text or ""))
    latin = len(LATIN_RE.findall(text or ""))
    if hangul == 0:
        return "en"
    return "ko" if hangul >= latin * 0.5 else "en"
