import streamlit as st
from html import escape

import numpy as np
import pandas as pd

from sentiment_model import (
    get_last_news_error,
    get_news_articles,
    get_stock_sentiment,
    has_news_api_key,
)
from stock_model import get_current_price, get_model, predict_next_day


SUPPORTED_STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]


def to_scalar_number(value, field_name):
    if isinstance(value, pd.DataFrame):
        if value.empty:
            raise ValueError(f"{field_name} is empty")
        value = value.iloc[-1, -1]

    if isinstance(value, pd.Series):
        if value.empty:
            raise ValueError(f"{field_name} is empty")
        value = value.iloc[-1]

    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError(f"{field_name} is empty")
        value = value.reshape(-1)[-1]

    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{field_name} is not numeric: {type(value)}") from exc


def smart_decision(current_price, predicted_price, sentiment_score):
    change_pct = (predicted_price - current_price) / current_price * 100

    if change_pct >= 3 and sentiment_score >= 0.25:
        return "STRONG BUY", "Both signals are aligned with a strong upside bias."
    if change_pct >= 1 and sentiment_score >= 0.05:
        return "BUY", "Model projects upside and sentiment supports momentum."
    if change_pct <= -3 and sentiment_score <= -0.25:
        return "STRONG SELL", "Downside pressure and sentiment risk are both elevated."
    if change_pct <= -1 and sentiment_score <= -0.05:
        return "SELL", "Weak price outlook with negative sentiment backdrop."

    return "HOLD", "Signals are mixed or weak, so no aggressive action is suggested."


def confidence_index(change_pct, sentiment_score, decision):
    price_strength = min(abs(change_pct) / 5, 1)
    sentiment_strength = min(abs(sentiment_score) / 0.6, 1)

    raw = 100 * (0.65 * price_strength + 0.35 * sentiment_strength)

    if decision == "HOLD":
        return max(40.0, min(72.0, raw))
    return max(48.0, min(95.0, raw))


def signal_tone(value):
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def render_metric_card(title, value, subtitle, tone="neutral"):
    st.markdown(
        f"""
        <div class="metric-card {tone}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_news_card(article, index):
    title = escape(article.get("title", "Untitled"))
    source = escape(article.get("source", "Unknown Source"))
    description = escape(article.get("description", "No description available."))
    published = escape(article.get("published_at", "Time unavailable"))
    url = article.get("url", "").strip()
    safe_url = escape(url, quote=True)

    if "T" in published:
        published = published.replace("T", " ").replace("Z", " UTC")

    link_html = (
        f'<a href="{safe_url}" target="_blank">Read full article</a>'
        if safe_url
        else '<span class="news-muted">Article link unavailable</span>'
    )

    st.markdown(
        f"""
        <div class="news-card">
            <div class="news-meta">#{index} | {source} | {published}</div>
            <div class="news-title">{title}</div>
            <div class="news-description">{description}</div>
            <div class="news-link">{link_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Stock Recommendation Studio",
    page_icon="ST",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    :root {
        --bg-soft: #f1f4f2;
        --bg-alt: #e4eae8;
        --surface: #ffffff;
        --text-main: #25313a;
        --text-muted: #66727a;
        --line: #d2dcda;
        --teal: #4f6f74;
        --slate: #3f5664;
        --positive: #2f6f56;
        --negative: #8f5f5f;
    }

    .stApp {
        font-family: 'Manrope', sans-serif;
        background:
            radial-gradient(circle at 10% -10%, #d9e3df 0%, transparent 40%),
            radial-gradient(circle at 90% 0%, #d6dfdf 0%, transparent 35%),
            linear-gradient(160deg, var(--bg-soft), var(--bg-alt));
        color: var(--text-main);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(120deg, rgba(63, 86, 100, 0.95), rgba(79, 111, 116, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 20px;
        color: #f4f8f8;
        padding: 1.35rem 1.35rem 1.2rem 1.35rem;
        box-shadow: 0 14px 36px rgba(20, 35, 45, 0.18);
        margin-bottom: 1rem;
        animation: reveal 0.55s ease-out;
    }

    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        margin: 0;
        font-size: 2rem;
        letter-spacing: 0.2px;
    }

    .hero p {
        margin-top: 0.45rem;
        margin-bottom: 0.9rem;
        color: #e7f3f3;
        font-size: 0.98rem;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }

    .chip {
        border: 1px solid rgba(255, 255, 255, 0.28);
        border-radius: 999px;
        padding: 0.3rem 0.72rem;
        font-size: 0.76rem;
        color: #f2fbfb;
        background: rgba(255, 255, 255, 0.09);
    }

    .panel {
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(4px);
        box-shadow: 0 8px 24px rgba(34, 46, 54, 0.08);
        padding: 1rem 1rem 0.9rem 1rem;
        margin-bottom: 0.9rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.45rem 0.85rem;
        color: var(--text-main) !important;
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
        opacity: 1 !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(63, 86, 100, 0.15);
        border-color: rgba(63, 86, 100, 0.35);
        color: #b04646 !important;
    }

    .panel h3 {
        margin-top: 0;
        margin-bottom: 0.55rem;
        font-family: 'Space Grotesk', sans-serif;
        color: var(--slate);
    }

    .metric-card {
        border-radius: 14px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.92);
        padding: 0.9rem 0.95rem;
        margin-bottom: 0.7rem;
        min-height: 128px;
    }

    .metric-card.positive {
        border-left: 5px solid var(--positive);
    }

    .metric-card.negative {
        border-left: 5px solid var(--negative);
    }

    .metric-card.neutral {
        border-left: 5px solid var(--slate);
    }

    .metric-title {
        color: var(--text-muted);
        font-size: 0.82rem;
        letter-spacing: 0.18px;
    }

    .metric-value {
        margin-top: 0.3rem;
        margin-bottom: 0.3rem;
        font-size: 1.55rem;
        font-weight: 800;
        color: var(--text-main);
    }

    .metric-subtitle {
        color: var(--text-muted);
        font-size: 0.84rem;
    }

    .use-case {
        border: 1px solid var(--line);
        border-radius: 14px;
        background: #f7fbfa;
        padding: 0.95rem;
        margin-bottom: 0.6rem;
    }

    .news-card {
        border: 1px solid var(--line);
        border-radius: 14px;
        background: #f9fcfb;
        padding: 0.85rem 0.9rem;
        margin-bottom: 0.65rem;
    }

    .news-meta {
        font-size: 0.76rem;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }

    .news-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.4rem;
    }

    .news-description {
        color: #4f5f67;
        font-size: 0.88rem;
        line-height: 1.45;
        margin-bottom: 0.42rem;
    }

    .news-link a {
        font-size: 0.84rem;
        color: var(--teal);
        text-decoration: none;
        font-weight: 600;
    }

    .news-link a:hover {
        text-decoration: underline;
    }

    .news-muted {
        font-size: 0.84rem;
        color: var(--text-muted);
    }

    .footnote {
        font-size: 0.82rem;
        color: var(--text-muted);
    }

    @keyframes reveal {
        from {opacity: 0; transform: translateY(8px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Stock Recommendation Studio</h1>
        <p>
            A two-factor recommendation engine that combines next-session price projection and
            financial-news sentiment analysis for clearer buy, sell, or hold calls.
        </p>
        <div class="chip-row">
            <span class="chip">Factor 1: LSTM Price Projection</span>
            <span class="chip">Factor 2: FinBERT News Sentiment</span>
            <span class="chip">Integrated Recommendation Layer</span>
            <span class="chip">Explainable Output Cards</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Project Overview")
    st.write("This interface is designed for practical recommendation workflows:")
    st.markdown("- Quick single-stock scan")
    st.markdown("- Combined technical plus sentiment signal")
    st.markdown("- Decision transparency for presentation and demos")
    st.markdown("### Supported Trained Symbols")
    st.write(", ".join(SUPPORTED_STOCKS))
    st.markdown("### Market News")
    st.write("News tab supports any stock symbol or company name.")
    st.markdown("### Next Planned Module")
    st.info("Conversational chatbot assistant will be integrated in a future release.")

tab_recommendation, tab_news = st.tabs(["Recommendation Studio", "Market News Feed"])

with tab_recommendation:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Run Recommendation")

    input_col_1, input_col_2 = st.columns([1, 2])

    with input_col_1:
        preset = st.selectbox(
            "Pick quick symbol",
            options=SUPPORTED_STOCKS + ["CUSTOM"],
            index=0,
        )

    default_symbol = preset if preset != "CUSTOM" else ""
    with input_col_2:
        stock_symbol = st.text_input(
            "Stock symbol",
            value=default_symbol,
            placeholder="Example: RELIANCE.NS",
        ).strip().upper()

    analyze = st.button("Run AI Analysis", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if analyze:
        if not stock_symbol:
            st.warning("Please enter a valid stock symbol before running analysis.")
        else:
            try:
                with st.spinner("Collecting market data, running model inference, and scoring sentiment..."):
                    model, scaler = get_model(stock_symbol)
                    current_price = get_current_price(stock_symbol)
                    predicted_price = predict_next_day(stock_symbol, model, scaler)
                    sentiment, sentiment_score = get_stock_sentiment(stock_symbol)

                    current_price = to_scalar_number(current_price, "Current price")
                    predicted_price = to_scalar_number(predicted_price, "Predicted price")
                    sentiment_score = to_scalar_number(sentiment_score, "Sentiment score")

                    change_pct = (predicted_price - current_price) / current_price * 100
                    decision, rationale = smart_decision(
                        current_price,
                        predicted_price,
                        sentiment_score,
                    )
                    confidence = confidence_index(change_pct, sentiment_score, decision)

                    st.session_state["analysis"] = {
                        "stock": stock_symbol,
                        "current_price": current_price,
                        "predicted_price": predicted_price,
                        "sentiment": sentiment,
                        "sentiment_score": sentiment_score,
                        "change_pct": change_pct,
                        "decision": decision,
                        "rationale": rationale,
                        "confidence": confidence,
                    }
            except Exception as exc:
                st.error(f"Analysis failed for symbol {stock_symbol}: {exc}")

    if "analysis" in st.session_state:
        result = st.session_state["analysis"]

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f"### Live Summary: {result['stock']}")

        row_1_col_1, row_1_col_2, row_1_col_3 = st.columns(3)
        row_2_col_1, row_2_col_2, row_2_col_3 = st.columns(3)

        with row_1_col_1:
            render_metric_card(
                "Current Price",
                f"INR {result['current_price']:.2f}",
                "Latest close fetched from market feed",
                "neutral",
            )

        with row_1_col_2:
            tone = signal_tone(result["change_pct"])
            render_metric_card(
                "Predicted Price (Next Session)",
                f"INR {result['predicted_price']:.2f}",
                "Model projection based on recent sequence",
                tone,
            )

        with row_1_col_3:
            tone = signal_tone(result["change_pct"])
            render_metric_card(
                "Projected Move",
                f"{result['change_pct']:+.2f}%",
                "Relative move from current level",
                tone,
            )

        with row_2_col_1:
            sentiment_tone = signal_tone(result["sentiment_score"])
            render_metric_card(
                "News Sentiment",
                result["sentiment"],
                f"Aggregate score: {result['sentiment_score']:+.3f}",
                sentiment_tone,
            )

        with row_2_col_2:
            decision_tone = "neutral"
            if "BUY" in result["decision"]:
                decision_tone = "positive"
            if "SELL" in result["decision"]:
                decision_tone = "negative"
            render_metric_card(
                "Decision",
                result["decision"],
                result["rationale"],
                decision_tone,
            )

        with row_2_col_3:
            render_metric_card(
                "Confidence Index",
                f"{result['confidence']:.1f}/100",
                "Weighted by price-signal strength and sentiment intensity",
                "neutral",
            )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Why This Recommendation Is Explainable")
    st.markdown(
        """
        1. Price Projection Signal: Sequence model estimates the next-session close from recent OHLCV behavior.
        2. News Sentiment Signal: FinBERT classifies financial headlines and creates a directional sentiment score.
        3. Fusion Logic: Recommendation is generated only after both factors are evaluated in a single decision layer.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    use_col_1, use_col_2, use_col_3 = st.columns(3)

    with use_col_1:
        st.markdown(
            """
            <div class="use-case">
                <h4>Pre-Market Screening</h4>
                <p>Shortlist stocks where projected move and sentiment are aligned before market open.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with use_col_2:
        st.markdown(
            """
            <div class="use-case">
                <h4>Swing Trade Support</h4>
                <p>Use recommendation output as a confirmation layer with your technical setup.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with use_col_3:
        st.markdown(
            """
            <div class="use-case">
                <h4>Investor Research Workflow</h4>
                <p>Present a compact and explainable report in demos, viva, and project reviews.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_news:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Live Market News")
    st.write("Search with any stock symbol or company name to get latest headlines.")

    if not has_news_api_key():
        st.error("NEWS_API_KEY is missing. Add it in Streamlit Cloud under Manage App > Settings > Secrets.")

    news_col_1, news_col_2 = st.columns([3, 1])

    with news_col_1:
        news_query = st.text_input(
            "Stock or company",
            value=st.session_state.get("news_query", ""),
            placeholder="Examples: HDFCBANK.NS, AAPL, Tesla, Infosys",
            key="news_query_input",
        ).strip()

    with news_col_2:
        news_limit = st.selectbox("Headlines", options=[6, 8, 10, 12], index=2)

    fetch_news = st.button("Fetch Latest News", use_container_width=True, type="primary", key="fetch_news")
    st.markdown('</div>', unsafe_allow_html=True)

    if fetch_news:
        if not news_query:
            st.warning("Please enter a stock symbol or company name.")
        else:
            with st.spinner("Fetching latest news feed..."):
                st.session_state["news_query"] = news_query
                st.session_state["news_articles"] = get_news_articles(news_query, limit=news_limit)
                st.session_state["news_error"] = get_last_news_error()

    if "news_articles" in st.session_state:
        active_query = st.session_state.get("news_query", "Market")
        articles = st.session_state.get("news_articles", [])

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f"### Results for: {active_query}")

        if not articles:
            details = st.session_state.get("news_error", "")
            if details:
                st.warning(f"News fetch issue: {details}")
            else:
                st.info("No recent headlines found for this query. Try a different stock symbol or company name.")
        else:
            for idx, article in enumerate(articles, start=1):
                render_news_card(article, idx)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### News Tab Use Cases")
    st.markdown("- Track headline flow for any stock before analysis")
    st.markdown("- Compare narrative momentum across different companies")
    st.markdown("- Collect references for project demos and reports")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("### Product Roadmap")
st.write("Planned next capability for this project:")
st.markdown("- Chatbot assistant to answer stock-specific why/how questions")
st.markdown("- Interactive what-if simulation with custom sentiment assumptions")
st.markdown("- Portfolio-level recommendation dashboard")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<p class="footnote">Note: This system is for educational and research use. Do not treat output as financial advice.</p>',
    unsafe_allow_html=True,
)