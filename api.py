from fastapi import FastAPI
from stock_model import get_model, predict_next_day, get_current_price
from sentiment_model import get_stock_sentiment

app = FastAPI()


def smart_decision(current_price, predicted_price, sentiment_score):

    change = (predicted_price - current_price) / current_price * 100

    if change > 2 and sentiment_score > 0:
        return "STRONG BUY"
    elif change < -2 and sentiment_score < 0:
        return "STRONG SELL"
    else:
        return "HOLD"


@app.get("/")
def home():
    return {"message": "AI Stock Assistant Running 🚀"}


@app.get("/analyze")
def analyze(stock: str):

    model, scaler = get_model(stock)

    current_price = get_current_price(stock)
    predicted_price = predict_next_day(stock, model, scaler)

    sentiment, score = get_stock_sentiment(stock)

    decision = smart_decision(current_price, predicted_price, score)

    return {
        "stock": stock,
        "current_price": float(current_price),
        "predicted_price": float(predicted_price),
        "sentiment": sentiment,
        "score": float(score),
        "decision": decision
    }