"""Load the classifiers and label comments.

For every (task, language) pair, MODEL_CANDIDATES lists models in order of
preference. The first one that is available is used:
  - a folder under models/ means your own fine-tuned model (copied from Drive);
  - a Hugging Face ID means a ready-made model, downloaded on first use.

So the dashboard works today with ready-made models, and switches to your own
models automatically as soon as you put them into models/.
"""

from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocess import clean_text, detect_language

ROOT = Path(__file__).resolve().parents[1]

MODEL_CANDIDATES = {
    ("toxicity", "en"): ["models/toxicity-en", "unitary/unbiased-toxic-roberta"],
    ("toxicity", "ko"): ["models/toxicity-ko"],
    ("sentiment", "en"): ["models/sentiment-en", "cardiffnlp/twitter-roberta-base-sentiment-latest"],
    ("sentiment", "ko"): ["models/sentiment-ko", "cardiffnlp/twitter-xlm-roberta-base-sentiment"],
}

TOXICITY_THRESHOLD = 0.5
MIXED_THRESHOLD = 0.3   # both positive and negative at least this high -> "mixed"
MAX_LENGTH = 128        # tokens; long comments are cut


def _resolve(candidate):
    """Return a loadable path/ID for a candidate, or None if a local folder is missing."""
    if candidate.startswith("models/"):
        path = ROOT / candidate
        return str(path) if (path / "config.json").exists() else None
    return candidate


def resolve_model(task, lang):
    for candidate in MODEL_CANDIDATES.get((task, lang), []):
        resolved = _resolve(candidate)
        if resolved:
            return resolved
    return None


class Classifier:
    """A Hugging Face sequence classifier plus the logic to read its outputs."""

    def __init__(self, name_or_path, task):
        self.name = name_or_path
        self.task = task
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(name_or_path)
        self.model.to(self.device).eval()

        labels = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]
        self.labels = labels
        # Toxicity models with many labels (toxicity, insult, threat, ...) are multi-label:
        # each label gets its own sigmoid. Two-label models use softmax.
        self.multi_label = (
            self.model.config.problem_type == "multi_label_classification"
            or (task == "toxicity" and len(labels) > 2)
        )
        if task == "toxicity":
            self.target_index = self._toxic_index(labels)
        else:
            self.sentiment_order = self._sentiment_order(labels)

    @staticmethod
    def _toxic_index(labels):
        for name in ("toxicity", "toxic"):
            if name in labels:
                return labels.index(name)
        for i, name in enumerate(labels):
            if "toxic" in name and not re.search(r"non|not", name):
                return i
        return 1  # plain LABEL_0 / LABEL_1 models (like ours): 1 = toxic

    @staticmethod
    def _sentiment_order(labels):
        """Indices of (negative, neutral, positive) in the model's outputs."""
        if all(any(key in l for l in labels) for key in ("neg", "neu", "pos")):
            return [next(i for i, l in enumerate(labels) if key in l) for key in ("neg", "neu", "pos")]
        return [0, 1, 2]  # LABEL_0/1/2 convention used by Cardiff and TweetEval

    def scores(self, texts, batch_size=32):
        """Return an array of probabilities, one row per text."""
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                     truncation=True, max_length=MAX_LENGTH).to(self.device)
                logits = self.model(**enc).logits
                probs = torch.sigmoid(logits) if self.multi_label else torch.softmax(logits, dim=-1)
                out.append(probs.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def toxicity(self, texts):
        return self.scores(texts)[:, self.target_index]

    def sentiment(self, texts):
        """Return an (n, 3) array of negative / neutral / positive probabilities."""
        return self.scores(texts)[:, self.sentiment_order]


@lru_cache(maxsize=None)
def get_classifier(task, lang):
    """Load (once) and return the classifier for a task and language, or None if none is available."""
    name = resolve_model(task, lang)
    return Classifier(name, task) if name else None


# ---------------------------------------------------------------- questions

EN_OPENERS = (
    "anyone know", "does anyone", "did anyone", "can someone", "can anyone", "could someone",
    "how do", "how does", "how did", "how can", "how to", "how much", "how many", "how long",
    "what is", "what's", "what are", "what was", "what does", "which", "why is", "why are",
    "why does", "why did", "why do", "where can", "where is", "when is", "when will", "when does",
    "who is", "is there", "is it", "are there", "are you", "can you", "could you", "will there",
    "will you", "do you", "should i",
)
KO_OPENERS = ("왜", "어떻게", "언제", "어디", "누가", "누구", "뭐가", "무슨", "몇", "혹시")
KO_ENDINGS = re.compile(r"(까요|나요|가요|는지|을까|ㄹ까|니|냐|죠|인가|건가)$")


def is_question(text, lang="en"):
    """Rule-based question detector, the baseline for this task.

    Known weakness: rhetorical questions ("who even watches this?") count as questions.
    """
    if not text:
        return False
    text_no_urls = re.sub(r"https?://\S+", "", text)
    if "?" in text_no_urls or "？" in text_no_urls:
        return True
    stripped = re.sub(r"[\s.!~ㅋㅎㅠㅜ…]+$", "", text_no_urls.strip())
    lower = stripped.lower()
    if lang == "ko":
        return lower.startswith(KO_OPENERS) or bool(KO_ENDINGS.search(lower))
    return lower.startswith(EN_OPENERS)


# ---------------------------------------------------------------- full analysis

def sentiment_label(neg, neu, pos):
    if pos >= MIXED_THRESHOLD and neg >= MIXED_THRESHOLD:
        return "mixed"
    return ("negative", "neutral", "positive")[int(np.argmax([neg, neu, pos]))]


def analyze(df, toxicity_threshold=TOXICITY_THRESHOLD, progress=None):
    """Add model outputs to a comments DataFrame (needs a "text" column).

    Returns (df, info). info maps "task/lang" to the model used, or None if no
    model exists yet for that pair (those rows get NaN). `progress` is an optional
    callback taking a float between 0 and 1.
    """
    df = df.copy()
    df["clean"] = df["text"].map(clean_text)
    df["lang"] = df["clean"].map(detect_language)
    df["is_question"] = [is_question(t, l) for t, l in zip(df["clean"], df["lang"])]
    for col in ("toxicity", "neg", "neu", "pos"):
        df[col] = np.nan

    info = {}
    jobs = [(task, lang) for lang in sorted(df["lang"].unique()) for task in ("toxicity", "sentiment")]
    for step, (task, lang) in enumerate(jobs):
        rows = df.index[df["lang"] == lang]
        clf = get_classifier(task, lang)
        info[f"{task}/{lang}"] = clf.name if clf else None
        if clf is not None:
            texts = df.loc[rows, "clean"].tolist()
            if task == "toxicity":
                df.loc[rows, "toxicity"] = clf.toxicity(texts)
            else:
                df.loc[rows, ["neg", "neu", "pos"]] = clf.sentiment(texts)
        if progress:
            progress((step + 1) / len(jobs))

    is_toxic = (df["toxicity"] >= toxicity_threshold).astype("boolean")
    is_toxic[df["toxicity"].isna()] = pd.NA   # no model for this language yet
    df["is_toxic"] = is_toxic
    df["sentiment"] = [
        sentiment_label(n, u, p) if not np.isnan(n) else None
        for n, u, p in zip(df["neg"], df["neu"], df["pos"])
    ]
    return df, info
