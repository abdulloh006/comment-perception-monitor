"""Fast tests that need no API key, GPU or model download. Run with: pytest"""

import pytest

from src.fetch.youtube import extract_video_id
from src.predict import is_question, sentiment_label
from src.preprocess import clean_text, detect_language


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=abc123XYZ00",
    "https://youtu.be/abc123XYZ00",
    "https://www.youtube.com/shorts/abc123XYZ00",
    "https://www.youtube.com/watch?v=abc123XYZ00&t=30s",
    "https://m.youtube.com/watch?v=abc123XYZ00",
    "https://youtu.be/abc123XYZ00?si=share",
    "https://www.youtube.com/live/abc123XYZ00",
    "youtube.com/watch?v=abc123XYZ00",
])
def test_extract_video_id(url):
    assert extract_video_id(url) == "abc123XYZ00"


@pytest.mark.parametrize("url", ["https://google.com", "", None, "https://www.youtube.com/"])
def test_extract_video_id_rejects(url):
    assert extract_video_id(url) is None


def test_clean_text():
    assert clean_text("  look @MKBHD  https://x.co/abc  wow ") == "look @user http wow"
    assert clean_text(None) == ""


@pytest.mark.parametrize("text,lang", [
    ("This is great", "en"),
    ("이 영상 진짜 좋아요", "ko"),
    ("아이폰 new design 좋다", "ko"),
    ("🔥🔥🔥", "en"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


@pytest.mark.parametrize("text,lang,expected", [
    ("What phone is that?", "en", True),
    ("does anyone know the song at 2:10", "en", True),
    ("Great video, thanks", "en", False),
    ("see https://example.com/?q=1", "en", False),
    ("이거 어디서 살 수 있나요", "ko", True),
    ("왜 이렇게 비싸", "ko", True),
    ("영상 잘 봤습니다", "ko", False),
])
def test_is_question(text, lang, expected):
    assert is_question(text, lang) is expected


def test_sentiment_label():
    assert sentiment_label(0.1, 0.2, 0.7) == "positive"
    assert sentiment_label(0.4, 0.2, 0.4) == "mixed"
    assert sentiment_label(0.1, 0.8, 0.1) == "neutral"
