# Comment Perception Monitor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abdulloh006/comment-perception-monitor/blob/main/notebooks/demo_colab.ipynb)

Paste a YouTube link and get a dashboard showing how the comment section feels: its sentiment, how toxic it is, which comments are real questions, and **why** the model flagged a comment. Built for English and Korean comments.

Deep Learning course project, Kyungdong University, Fall 2026.

## What it does

- **Fetches** up to 1,000 comments from any YouTube video through the YouTube Data API (top or newest).
- **Classifies** each comment with transformer models:
  - toxicity (toxic / not toxic, with an adjustable threshold),
  - sentiment (positive / neutral / negative, plus *mixed* when both positive and negative are high),
  - questions (a rule-based baseline).
- **Explains** toxicity predictions word by word with Integrated Gradients: red words pushed the model toward "toxic", blue words away.
- **Routes** each comment to an English or Korean model based on its script.
- **Exports** the labeled comments as CSV.

## How it works

```
TRAINING (offline, Colab)                 INFERENCE (every time a link is pasted)
labeled datasets                          YouTube link
  → clean_text()  ◄──── shared ────►        → fetch comments (src/fetch/youtube.py)
  → fine-tune transformer                   → clean_text(), detect language (src/preprocess.py)
  → evaluate on held-out test set           → classify (src/predict.py)
  → save to models/                         → explain on request (src/explain.py)
                                            → Streamlit dashboard (app/app.py)
```

Training and inference share one cleaning function, so the model sees the same kind of text in the dashboard that it saw in training.

## Results so far

English toxicity, Civil Comments, the same held-out test set of 20,000 comments (6.9% toxic):

| Model | Accuracy | Toxic precision | Toxic recall | **Toxic F1** | **Macro F1** |
|---|---|---|---|---|---|
| Always "not toxic" | 0.931 | 0.000 | 0.000 | 0.000 | 0.482 |
| TF-IDF + logistic regression | 0.913 | 0.417 | 0.658 | 0.510 | 0.731 |
| DistilBERT (fine-tuned, 40k comments, 2 epochs) | **0.955** | **0.710** | 0.602 | **0.652** | **0.814** |

The useless "always not toxic" model has *higher* accuracy than the real baseline, which is why this project reports F1 instead of accuracy.

### Identity bias

Toxicity probability for harmless sentences that differ in one word:

| Sentence | TF-IDF | DistilBERT |
|---|---|---|
| I love my family. | 0.15 | 0.00 |
| I am a proud Muslim. | 0.50 | 0.05 |
| I am a proud Christian. | 0.16 | 0.00 |
| I am a gay man. | 0.98 | **0.95** |
| I am a straight man. | 0.45 | 0.01 |
| I am a Black woman. | 0.97 | **0.94** |
| I am a white woman. | 0.98 | 0.77 |
| You're killing it, amazing performance! | 0.40 | 0.01 |
| You are an idiot. | 1.00 | 0.99 |

DistilBERT beats the baseline on every overall metric yet still rates harmless identity statements as toxic. Word attributions confirm that the identity word itself drives the score. A dashboard built on this model would wrongly report the audiences of LGBTQ or Black creators as toxic. Higher F1 does not mean fairer.

## Try it in Colab

Click the **Open in Colab** badge at the top, switch the runtime to a T4 GPU, add your YouTube API key under 🔑 Secrets as `YOUTUBE_API_KEY`, and run the cells. The last cell can open the full dashboard through a temporary public link.

## Quick start (Windows)

```bat
git clone https://github.com/abdulloh006/comment-perception-monitor.git
cd comment-perception-monitor
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put your YouTube Data API v3 key into `.env`, then:

```bat
streamlit run app/app.py
```

The first analysis downloads the models (about 1.5 GB). To keep them off the C: drive, set `HF_HOME` first, e.g. `set HF_HOME=E:\hf-cache`.

Run the tests (no API key or GPU needed): `pytest`

## Models

The app uses your own fine-tuned models from `models/` when present and falls back to ready-made ones otherwise (see [`models/README.md`](models/README.md)).

| Task | Default model | Notes |
|---|---|---|
| Toxicity, English | `unitary/unbiased-toxic-roberta` (Detoxify) | RoBERTa trained for the Jigsaw Unintended Bias challenge |
| Sentiment, English | `cardiffnlp/twitter-roberta-base-sentiment-latest` | RoBERTa trained on ~124M tweets, fine-tuned on TweetEval |
| Sentiment, Korean | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Multilingual; not validated on Korean comments yet |
| Toxicity, Korean | your KcELECTRA model in `models/toxicity-ko` | Planned |
| Questions | rule-based baseline | Question marks plus English and Korean question patterns |

## Project structure

```
app/app.py              Streamlit dashboard
src/fetch/youtube.py    link → video ID → comments, with error handling
src/preprocess.py       shared text cleaning and language detection
src/predict.py          model loading, routing and labeling
src/explain.py          Integrated Gradients word attributions
notebooks/              exploration, baselines and training (Colab)
models/                 fine-tuned models (not in Git)
tests/                  unit tests
```

## Roadmap

- [x] YouTube fetching with error handling
- [x] Baselines and fine-tuned DistilBERT for English toxicity
- [x] Bias test and word-level explanations
- [x] Dashboard with ready-made models
- [ ] RoBERTa fine-tune and comparison with Detoxify on the official test split
- [ ] Korean toxicity with KcELECTRA (Korean UnSmile / K-MHaS)
- [ ] ~300 hand-labeled YouTube comments as a real-domain test set
- [ ] Deployment to Hugging Face Spaces

## Limitations

- **Platforms.** Reddit closed new API access in 2026 and X's API is pay-per-use only, so the project uses YouTube's free Data API.
- **Sampling.** "Top comments" follows YouTube's undisclosed relevance ranking; "newest" gives a less filtered slice. The two can give quite different toxicity rates for the same video.
- **Domain shift.** The models are trained on news comments and tweets, not YouTube comments, and sarcasm and jokes are often misread.
- **Bias.** See above; identity terms raise toxicity scores.
- **Questions.** The rule-based detector counts rhetorical questions as questions.
- **Privacy.** Commenter names are not collected.

## License

MIT
