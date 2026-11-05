"""Fetch comments from a YouTube video with the YouTube Data API v3.

Typical use:
    youtube = get_client()                      # reads YOUTUBE_API_KEY from the environment
    df, error = get_comments_from_url(youtube, "https://youtu.be/abc123XYZ00")

Each page of up to 100 comments costs 1 quota unit (10,000 free units per day).
Commenter names are deliberately not collected: the analysis needs the text,
not who wrote it.
"""

import os
from urllib.parse import parse_qs, urlparse

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

PAGE_SIZE = 100  # the maximum YouTube allows per request


def get_client(api_key=None):
    """Create a YouTube API client. Falls back to the YOUTUBE_API_KEY environment variable."""
    key = api_key or os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise ValueError("No YouTube API key. Set YOUTUBE_API_KEY in your .env file.")
    return build("youtube", "v3", developerKey=key, cache_discovery=False)


def extract_video_id(url):
    """Return the 11-character video ID from a YouTube link, or None if it isn't one.

    Handles youtube.com/watch?v=ID, youtu.be/ID, youtube.com/shorts/ID,
    youtube.com/live/ID and mobile (m.youtube.com) links.
    """
    if not url:
        return None
    url = url.strip()
    if "://" not in url:  # people often paste links without https://
        url = "https://" + url

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path

    if host.endswith("youtu.be"):
        video_id = path.lstrip("/").split("/")[0]
        return video_id or None

    if host.endswith("youtube.com"):
        if path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        parts = path.split("/")
        if len(parts) >= 3 and parts[1] in ("shorts", "live", "embed"):
            return parts[2] or None

    return None


def fetch_comments(youtube, video_id, max_comments=500, order="relevance"):
    """Fetch up to `max_comments` top-level comments, following nextPageToken.

    order: "relevance" (YouTube's Top comments ranking) or "time" (newest first).
    Note that "relevance" is not "most likes"; sort the DataFrame yourself for that.
    """
    comments = []
    next_token = None

    while len(comments) < max_comments:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=PAGE_SIZE,
            textFormat="plainText",
            order=order,
            pageToken=next_token,
        ).execute()

        for item in response.get("items", []):
            thread = item["snippet"]
            c = thread["topLevelComment"]["snippet"]
            comments.append({
                "text": c["textDisplay"],
                "likes": c.get("likeCount", 0),
                "replies": thread.get("totalReplyCount", 0),
                "published_at": c["publishedAt"],
            })

        next_token = response.get("nextPageToken")
        if not next_token:  # last page reached
            break

    df = pd.DataFrame(comments[:max_comments], columns=["text", "likes", "replies", "published_at"])
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


def fetch_video_title(youtube, video_id):
    """Return the video's title, or None if it can't be read."""
    try:
        resp = youtube.videos().list(part="snippet", id=video_id).execute()
        items = resp.get("items", [])
        return items[0]["snippet"]["title"] if items else None
    except HttpError:
        return None


def get_comments_from_url(youtube, url, max_comments=500, order="relevance"):
    """Link in, (DataFrame, None) or (None, error message) out. Never raises on API errors."""
    video_id = extract_video_id(url)
    if video_id is None:
        return None, "That doesn't look like a YouTube link."

    try:
        df = fetch_comments(youtube, video_id, max_comments=max_comments, order=order)
    except HttpError as e:
        status = getattr(e.resp, "status", None)
        message = str(e)
        if "commentsDisabled" in message:
            return None, "Comments are turned off for this video."
        if "quotaExceeded" in message:
            return None, "Today's YouTube API quota is used up. It resets at midnight Pacific time."
        if status == 404 or "videoNotFound" in message:
            return None, "Video not found. Check the link."
        if status in (400, 403) and "API key" in message:
            return None, "The YouTube API key was rejected. Check YOUTUBE_API_KEY."
        return None, f"YouTube API error ({status})."

    if df.empty:
        return None, "This video has no comments yet."

    return df, None
