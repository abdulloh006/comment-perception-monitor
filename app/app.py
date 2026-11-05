"""Comment Perception Monitor: paste a YouTube link, see how the comment section feels.

Run from the repository root:
    streamlit run app/app.py
"""

import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.explain import explain, render_html  # noqa: E402
from src.fetch.youtube import (  # noqa: E402
    extract_video_id, fetch_video_title, get_client, get_comments_from_url,
)
from src.predict import analyze, get_classifier, resolve_model  # noqa: E402

st.set_page_config(page_title="Comment Perception Monitor", page_icon="💬", layout="wide")

SENTIMENT_ORDER = ["positive", "neutral", "mixed", "negative"]
SENTIMENT_COLOURS = ["#16a34a", "#94a3b8", "#d97706", "#dc2626"]
LANG_NAMES = {"en": "English", "ko": "Korean"}


def api_key():
    key = os.environ.get("YOUTUBE_API_KEY")
    if key:
        return key
    try:
        return st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def load_comments(key, url, max_comments, order):
    youtube = get_client(key)
    df, error = get_comments_from_url(youtube, url, max_comments=max_comments, order=order)
    title = fetch_video_title(youtube, extract_video_id(url)) if df is not None else None
    return df, error, title


@st.cache_resource(show_spinner=False)
def toxicity_model(lang):
    return get_classifier("toxicity", lang)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    key = api_key() or st.text_input("YouTube API key", type="password",
                                     help="Put YOUTUBE_API_KEY in a .env file to skip this.")
    max_comments = st.slider("Comments to analyze", 100, 1000, 300, step=100,
                             help="Each 100 comments costs 1 unit of the free 10,000 daily quota.")
    order_label = st.radio("Which comments", ["Top comments", "Newest first"],
                           help="Top comments show what most viewers saw. Newest gives a less "
                                "filtered slice, which often looks more toxic.")
    order = "relevance" if order_label == "Top comments" else "time"
    threshold = st.slider("Toxicity threshold", 0.1, 0.9, 0.5, step=0.05,
                          help="Lower catches more toxic comments but raises false alarms "
                               "(the precision–recall trade-off).")
    with st.expander("Models in use"):
        for task in ("toxicity", "sentiment"):
            for lang in ("en", "ko"):
                name = resolve_model(task, lang)
                if name is None:
                    label = "not available yet"
                elif Path(name).is_dir():
                    label = f"your model (models/{Path(name).name})"
                else:
                    label = name
                st.caption(f"**{task} · {LANG_NAMES[lang]}**: {label}")
        st.caption("Questions: rule-based baseline")

# ------------------------------------------------------------------ input
st.title("💬 Comment Perception Monitor")
st.caption("Paste a YouTube link to see the tone of its comments, how toxic they are, "
           "what viewers are asking, and why the model flagged a comment.")

with st.form("input"):
    url = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=...")
    submitted = st.form_submit_button("Analyze", type="primary")

if submitted:
    if not key:
        st.error("Add a YouTube API key in the sidebar or in your .env file.")
        st.stop()
    with st.spinner("Fetching comments…"):
        df, error, title = load_comments(key, url, max_comments, order)
    if error:
        st.error(error)
        st.stop()
    bar = st.progress(0.0, text="Running the models (the first run downloads them)…")
    result, info = analyze(df, progress=lambda p: bar.progress(p, text="Running the models…"))
    bar.empty()
    st.session_state["result"] = (result, info, title, url)

if "result" not in st.session_state:
    st.info("Results appear here after you analyze a video.")
    st.stop()

result, info, title, url = st.session_state["result"]
df = result.copy()
scored = df["toxicity"].notna()
df["is_toxic"] = (df["toxicity"] >= threshold).where(scored)

# ------------------------------------------------------------------ summary
st.subheader(title or "Results")
st.caption(url)

with_sentiment = df["sentiment"].notna()
cols = st.columns(5)
cols[0].metric("Comments", len(df))
cols[1].metric("Toxic", f"{df.loc[scored, 'is_toxic'].mean():.0%}" if scored.any() else "–")
cols[2].metric("Positive", f"{(df.loc[with_sentiment, 'sentiment'] == 'positive').mean():.0%}"
               if with_sentiment.any() else "–")
cols[3].metric("Negative", f"{(df.loc[with_sentiment, 'sentiment'] == 'negative').mean():.0%}"
               if with_sentiment.any() else "–")
cols[4].metric("Questions", int(df["is_question"].sum()))

for job, model in info.items():
    if model is None:
        task, lang = job.split("/")
        n = int((df["lang"] == lang).sum())
        st.warning(f"{n} {LANG_NAMES.get(lang, lang)} comments have no {task} model yet, "
                   f"so they are left out of the {task} numbers.")

overview, questions, toxic, everything = st.tabs(
    ["Overview", f"Questions ({int(df['is_question'].sum())})", "Toxic comments", "All comments"])

# ------------------------------------------------------------------ overview
with overview:
    left, right = st.columns(2)
    with left:
        st.markdown("**Sentiment**")
        counts = (df["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
                  .rename_axis("sentiment").reset_index(name="comments"))
        st.altair_chart(
            alt.Chart(counts).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("comments:Q", title="Comments"),
                y=alt.Y("sentiment:N", sort=SENTIMENT_ORDER, title=None),
                color=alt.Color("sentiment:N", scale=alt.Scale(domain=SENTIMENT_ORDER,
                                range=SENTIMENT_COLOURS), legend=None),
                tooltip=["sentiment", "comments"],
            ).properties(height=220),
            use_container_width=True,
        )
    with right:
        st.markdown("**Toxicity scores**")
        if scored.any():
            st.altair_chart(
                alt.Chart(df[scored]).mark_bar(color="#dc2626").encode(
                    x=alt.X("toxicity:Q", bin=alt.Bin(extent=[0, 1], step=0.05), title="Toxicity probability"),
                    y=alt.Y("count():Q", title="Comments"),
                ).properties(height=220)
                + alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(strokeDash=[4, 4]).encode(x="t:Q"),
                use_container_width=True,
            )
        else:
            st.caption("No toxicity model for these comments yet.")
    st.caption("Language mix: " + ", ".join(
        f"{LANG_NAMES.get(l, l)} {n}" for l, n in df["lang"].value_counts().items()))

# ------------------------------------------------------------------ questions
with questions:
    q = df[df["is_question"]].sort_values("likes", ascending=False)
    if q.empty:
        st.caption("No questions found.")
    else:
        st.caption("Questions viewers are asking, most-liked first. Detected by a simple rule, "
                   "so rhetorical questions slip in.")
        st.dataframe(q[["text", "likes", "replies"]], hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ toxic comments
with toxic:
    top = df[scored].sort_values("toxicity", ascending=False).head(15)
    if top.empty:
        st.caption("No toxicity scores for these comments.")
    for i, row in top.iterrows():
        with st.expander(f"{row['toxicity']:.2f} · {row['text'][:90]}"):
            st.write(row["text"])
            st.caption(f"{row['likes']} likes · {row['replies']} replies")
            if st.button("Explain why", key=f"explain_{i}"):
                clf = toxicity_model(row["lang"])
                with st.spinner("Computing word attributions…"):
                    words, scores, prob = explain(clf, row["clean"])
                st.html(render_html(words, scores, prob))
                st.caption("Red words pushed toward toxic, blue words pushed away.")

    st.divider()
    st.markdown("**Try your own sentence**")
    st.caption("Useful for testing bias, e.g. compare “I am a gay man.” with “I am a straight man.”")
    own = st.text_input("Sentence", key="own_sentence")
    if own:
        from src.preprocess import clean_text, detect_language
        clf = toxicity_model(detect_language(own))
        if clf is None:
            st.caption("No toxicity model for this language yet.")
        else:
            with st.spinner("Computing word attributions…"):
                words, scores, prob = explain(clf, clean_text(own))
            st.html(render_html(words, scores, prob))

# ------------------------------------------------------------------ all comments
with everything:
    f1, f2, f3 = st.columns(3)
    chosen = f1.multiselect("Sentiment", SENTIMENT_ORDER, default=SENTIMENT_ORDER)
    only_toxic = f2.checkbox("Toxic only")
    only_questions = f3.checkbox("Questions only")
    view = df[df["sentiment"].isin(chosen) | df["sentiment"].isna()]
    if only_toxic:
        view = view[view["is_toxic"] == True]  # noqa: E712 (NaN rows must drop out)
    if only_questions:
        view = view[view["is_question"]]
    columns = ["text", "likes", "replies", "published_at", "lang", "sentiment", "toxicity", "is_question"]
    st.dataframe(view[columns], hide_index=True, use_container_width=True,
                 column_config={"toxicity": st.column_config.ProgressColumn(
                     "toxicity", min_value=0.0, max_value=1.0, format="%.2f")})
    st.download_button("Download CSV", view[columns].to_csv(index=False).encode("utf-8"),
                       file_name="comments_analysis.csv", mime="text/csv")
