# models/

Trained model folders go here. They are ignored by Git because of their size.

The app looks for these folders first and uses a ready-made Hugging Face model when a folder is missing:

| Folder | Task | Fallback when missing |
|---|---|---|
| `toxicity-en/` | English toxicity | `unitary/unbiased-toxic-roberta` |
| `toxicity-ko/` | Korean toxicity | none (Korean toxicity is left out) |
| `sentiment-en/` | English sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| `sentiment-ko/` | Korean sentiment | `cardiffnlp/twitter-xlm-roberta-base-sentiment` |

A folder is the output of `trainer.save_model(...)` plus `tokenizer.save_pretrained(...)`: it must contain `config.json`, the weights, and the tokenizer files. To use the DistilBERT toxicity model trained in Colab, download `comment-monitor/distilbert-toxic-final` from Google Drive and rename it to `models/toxicity-en/`.

To change the order or add models, edit `MODEL_CANDIDATES` in `src/predict.py`.
