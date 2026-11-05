"""Word-level explanations with Integrated Gradients (Captum).

Idea: start from a "blank" comment (padding tokens), blend the real words in over
n_steps, and credit each word with how much it moved the toxic probability.
Positive score = pushed toward toxic, negative = pushed away.
"""

import html

import torch
from captum.attr import LayerIntegratedGradients


def _forward_fn(clf):
    def forward(input_ids, attention_mask):
        logits = clf.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.sigmoid(logits) if clf.multi_label else torch.softmax(logits, dim=-1)
        return probs[:, clf.target_index]
    return forward


def explain(clf, text, n_steps=32):
    """Return (words, scores, prob) for one comment.

    clf is a toxicity Classifier from src.predict. Scores are summed over the
    sub-word tokens of each word and scaled to [-1, 1].
    """
    tok, model = clf.tokenizer, clf.model
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128,
              return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    word_ids = enc.word_ids(0)
    enc = enc.to(clf.device)
    ids, mask = enc["input_ids"], enc["attention_mask"]

    # Blank baseline: keep the special tokens, replace everything else with padding.
    special = torch.tensor(tok.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True),
                           device=clf.device).bool()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    baseline = torch.where(special, ids[0], torch.full_like(ids[0], pad_id)).unsqueeze(0)

    forward = _forward_fn(clf)
    lig = LayerIntegratedGradients(forward, model.get_input_embeddings())
    attrs = lig.attribute(ids, baselines=baseline, additional_forward_args=(mask,),
                          n_steps=n_steps, internal_batch_size=n_steps)
    token_scores = attrs.sum(dim=-1)[0].detach().cpu().tolist()
    with torch.no_grad():
        prob = forward(ids, mask).item()

    # Merge sub-word tokens into words, using the original text spans.
    words, scores, spans = [], [], []
    for token_index, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        start, end = offsets[token_index]
        if spans and spans[-1][0] == word_id:
            spans[-1][2] = end
            scores[-1] += token_scores[token_index]
        else:
            spans.append([word_id, start, end])
            scores.append(token_scores[token_index])
    words = [text[s:e].strip() for _, s, e in spans]  # RoBERTa-style offsets can include the leading space

    peak = max((abs(s) for s in scores), default=0) or 1.0
    return words, [s / peak for s in scores], prob


def render_html(words, scores, prob=None):
    """Colour words: red pushed toward toxic, blue pushed away; darker = stronger."""
    parts = []
    if prob is not None:
        parts.append(f"<b style='margin-right:8px'>{prob:.2f}</b>")
    for word, score in zip(words, scores):
        alpha = min(abs(score), 1.0) * 0.85
        colour = f"rgba(220,38,38,{alpha:.2f})" if score > 0 else f"rgba(37,99,235,{alpha:.2f})"
        parts.append(
            f"<span title='{score:+.2f}' style='background:{colour};padding:1px 3px;"
            f"margin:1px;border-radius:4px;display:inline-block'>{html.escape(word)}</span>"
        )
    return "<div style='line-height:2;font-size:16px'>" + " ".join(parts) + "</div>"
