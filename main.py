from stock_model import get_model, predict_next_day, get_current_price
from sentiment_model import get_stock_sentiment


# =========================
# SMART DECISION ENGINE
# =========================
def smart_decision(current_price, predicted_price, sentiment_score):

    change = (predicted_price - current_price) / current_price * 100

    if change > 2 and sentiment_score > 0:
        return "STRONG BUY"

    elif change > 1 and sentiment_score > 0:
        return "BUY"

    elif change < -2 and sentiment_score < 0:
        return "STRONG SELL"

    elif change < -1 and sentiment_score < 0:
        return "SELL"

    else:
        return "HOLD"


# =========================
# MAIN ANALYSIS FUNCTION
# =========================
def analyze_stock(stock_name):

    model, scaler = get_model(stock_name)

    current_price = get_current_price(stock_name)
    predicted_price = predict_next_day(stock_name, model, scaler)

    sentiment, score = get_stock_sentiment(stock_name)

    decision = smart_decision(current_price, predicted_price, score)

    print("\n========================")
    print(f"Stock: {stock_name}")
    print(f"Current Price: {current_price}")
    print(f"Predicted Price: {predicted_price}")
    print(f"Sentiment: {sentiment} ({score:.2f})")
    print(f"Final Decision: {decision}")
    print("========================")


# =========================
# RUN
# =========================
if __name__ == "__main__":

    stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

    for stock in stocks:
        analyze_stock(stock)