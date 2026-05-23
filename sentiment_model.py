import os
import re
from functools import lru_cache

import requests


POSITIVE_TERMS = {
    "beat", "surge", "gain", "gains", "growth", "rise", "rises", "record",
    "strong", "up", "upgrade", "bullish", "profit", "profits", "expands",
}
NEGATIVE_TERMS = {
    "drop", "falls", "fall", "loss", "losses", "down", "weak", "downgrade",
    "bearish", "miss", "slump", "decline", "cuts", "risk", "lawsuit",
}

_LAST_NEWS_ERROR = None


@lru_cache(maxsize=1)
def get_sentiment_model():
    # Lazy import avoids importing torch/transformers during Streamlit boot.
    from transformers import pipeline

    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def _tokenize(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def _rule_based_avg_score(headlines):
    score_sum = 0.0

    for item in headlines:
        tokens = _tokenize(item)
        if not tokens:
            continue

        positive_hits = sum(token in POSITIVE_TERMS for token in tokens)
        negative_hits = sum(token in NEGATIVE_TERMS for token in tokens)
        total_hits = positive_hits + negative_hits

        if total_hits == 0:
            continue

        score_sum += (positive_hits - negative_hits) / total_hits

    return score_sum / max(len(headlines), 1)


def _get_news_api_key():
    env_key = os.getenv("NEWS_API_KEY", "").strip()
    if env_key:
        return env_key

    # Streamlit Cloud stores user-provided secrets in st.secrets.
    try:
        import streamlit as st

        secret_key = str(st.secrets.get("NEWS_API_KEY", "")).strip()
        if secret_key:
            return secret_key
    except Exception:
        pass

    return ""


def has_news_api_key():
    return bool(_get_news_api_key())


def get_last_news_error():
    return _LAST_NEWS_ERROR


# =========================
# GET NEWS
# =========================
def get_news_articles(stock_name, limit=10):
    global _LAST_NEWS_ERROR

    _LAST_NEWS_ERROR = None

    api_key = _get_news_api_key()
    if not api_key:
        _LAST_NEWS_ERROR = "NEWS_API_KEY is missing. Add this key in your Streamlit app Secrets settings."
        print(f"Warning: {_LAST_NEWS_ERROR}")
        return []

    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": stock_name,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": limit,
                "apiKey": api_key,
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "error":
            _LAST_NEWS_ERROR = data.get("message", "Unknown NewsAPI error")
            print(f"Warning: NewsAPI error for {stock_name}: {_LAST_NEWS_ERROR}")
            return []
        
        # Handle API errors
        if "articles" not in data:
            _LAST_NEWS_ERROR = "No articles field returned by NewsAPI response."
            print(f"Warning: No articles found for {stock_name}")
            return []
            
        articles = data["articles"]
        
        # Handle empty results
        if not articles:
            _LAST_NEWS_ERROR = f"No recent headlines available for {stock_name}."
            print(f"Warning: No news articles available for {stock_name}")
            return []

        normalized = []
        for article in articles[:limit]:
            title = (article.get("title") or "").strip()
            if not title:
                continue

            normalized.append(
                {
                    "title": title,
                    "description": (article.get("description") or "").strip(),
                    "url": (article.get("url") or "").strip(),
                    "published_at": (article.get("publishedAt") or "").strip(),
                    "source": (article.get("source") or {}).get("name", "Unknown"),
                }
            )

        if not normalized:
            _LAST_NEWS_ERROR = f"Headlines were empty after filtering for {stock_name}."
            print(f"Warning: All headlines were empty for {stock_name}")
            return []

        return normalized
    except Exception as e:
        _LAST_NEWS_ERROR = str(e)
        print(f"Error fetching news for {stock_name}: {str(e)}")
        return []


def get_news(stock_name):
    articles = get_news_articles(stock_name, limit=10)
    return [item["title"] for item in articles]


# =========================
# ANALYZE SENTIMENT
# =========================
def analyze_sentiment(headlines):

    # Handle empty headlines
    if not headlines or len(headlines) == 0:
        print("No headlines to analyze. Returning NEUTRAL sentiment.")
        return "NEUTRAL 🤝", 0.0

    try:
        model = get_sentiment_model()
        results = model(headlines)

        score = 0
        for r in results:
            if r['label'] == 'positive':
                score += r['score']
            elif r['label'] == 'negative':
                score -= r['score']

        avg_score = score / len(results)
    except Exception:
        print("Warning: FinBERT unavailable. Using rule-based sentiment fallback.")
        avg_score = _rule_based_avg_score(headlines)

    if avg_score > 0.2:
        sentiment = "POSITIVE 📈"
    elif avg_score < -0.2:
        sentiment = "NEGATIVE 📉"
    else:
        sentiment = "NEUTRAL 🤝"

    return sentiment, avg_score


# =========================
# MAIN FUNCTION
# =========================
def get_stock_sentiment(stock_name):

    headlines = get_news(stock_name)
    sentiment, score = analyze_sentiment(headlines)

    return sentiment, score