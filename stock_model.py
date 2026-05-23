import yfinance as yf
import numpy as np
import joblib
import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def _load_tf_components():
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.models import Sequential, load_model

    return Sequential, load_model, LSTM, Dense


def _tensorflow_available():
    try:
        _load_tf_components()
        return True
    except Exception:
        return False


def _safe_float(value):
    if isinstance(value, pd.Series):
        if value.empty:
            raise ValueError("Cannot convert empty series to float")
        value = value.iloc[-1]

    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("Cannot convert empty array to float")
        value = value.reshape(-1)[-1]

    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Unable to convert value to float: {type(value)}") from exc


def _extract_series(data, column_name):
    if column_name not in data:
        raise ValueError(f"Column {column_name} not found in market data")

    values = data[column_name]

    # yfinance may return a DataFrame for one logical column when columns are multi-indexed.
    if isinstance(values, pd.DataFrame):
        if values.empty:
            raise ValueError(f"Column {column_name} has no values")
        values = values.iloc[:, 0]

    if not isinstance(values, pd.Series):
        values = pd.Series(values)

    return values


def _extract_ohlcv(data):
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    extracted = {}

    for col in required_cols:
        extracted[col] = _extract_series(data, col)

    frame = pd.DataFrame(extracted).dropna()

    if frame.empty:
        raise ValueError("OHLCV frame is empty after cleanup")

    return frame


# =========================
# TRAIN MODEL
# =========================
def train_model(stock_name):

    if not _tensorflow_available():
        raise RuntimeError("TensorFlow is not available in this environment.")

    Sequential, _, LSTM, Dense = _load_tf_components()

    print(f"Training model for {stock_name}...")

    data = yf.download(stock_name, start="2010-01-01", end="2024-01-01")
    data = _extract_ohlcv(data)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []

    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i][3])  # Close

    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 5)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save model & scaler
    model.save(f"{stock_name}_model.h5")
    joblib.dump(scaler, f"{stock_name}_scaler.pkl")

    return model, scaler


# =========================
# LOAD MODEL
# =========================
def load_saved_model(stock_name):

    if not _tensorflow_available():
        raise RuntimeError("TensorFlow is not available in this environment.")

    _, load_model, _, _ = _load_tf_components()

    print(f"Loading saved model for {stock_name}...")

    model = load_model(f"{stock_name}_model.h5")
    scaler = joblib.load(f"{stock_name}_scaler.pkl")

    return model, scaler


# =========================
# AUTO LOAD / TRAIN
# =========================
def get_model(stock_name):

    has_saved_model = os.path.exists(f"{stock_name}_model.h5") and os.path.exists(f"{stock_name}_scaler.pkl")
    tf_ready = _tensorflow_available()

    if has_saved_model and tf_ready:
        return load_saved_model(stock_name)

    if not tf_ready:
        print(f"TensorFlow unavailable. Using fallback predictor for {stock_name}.")
        return None, None

    return train_model(stock_name)


# =========================
# CURRENT PRICE
# =========================
def get_current_price(stock_name):
    data = yf.download(stock_name, period="5d")
    if data.empty:
        raise ValueError(f"No market data found for {stock_name}")

    close = _extract_series(data, "Close").dropna()
    if close.empty:
        raise ValueError(f"No close prices found for {stock_name}")

    return _safe_float(close.iloc[-1])


def predict_next_day_fallback(stock_name):
    data = yf.download(stock_name, period="6mo")
    if data.empty:
        raise ValueError(f"No market data found for {stock_name}")

    close = _extract_series(data, "Close").dropna()
    if close.empty:
        raise ValueError(f"No close price series found for {stock_name}")

    latest = _safe_float(close.iloc[-1])
    if len(close) < 25:
        return latest

    lookback_idx = max(0, len(close) - 6)
    ref_price = _safe_float(close.iloc[lookback_idx])
    if ref_price <= 0:
        return latest

    ret_5d = (latest - ref_price) / ref_price
    ma_5 = _safe_float(close.tail(5).mean())
    ma_20 = _safe_float(close.tail(20).mean())
    trend = (ma_5 - ma_20) / ma_20 if ma_20 else 0.0

    expected_change = 0.35 * ret_5d + 0.65 * trend
    expected_change = _safe_float(np.clip(expected_change, -0.05, 0.05))

    return latest * (1 + expected_change)


# =========================
# PREDICTION
# =========================
def predict_next_day(stock_name, model, scaler):

    if model is None or scaler is None:
        return predict_next_day_fallback(stock_name)

    try:
        data = yf.download(stock_name, period="3mo")
        data = _extract_ohlcv(data)

        scaled_data = scaler.transform(data)

        last_60_days = scaled_data[-60:]
        X_test = np.array([last_60_days])

        pred = model.predict(X_test)

        # inverse transform
        dummy = np.zeros((1, 5))
        dummy[0][3] = pred

        predicted_price = scaler.inverse_transform(dummy)[0][3]

        # Convert to Python float scalar
        return _safe_float(predicted_price)
    except Exception:
        return predict_next_day_fallback(stock_name)


# =========================
# TEST RUN
# =========================
if __name__ == "__main__":

    stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

    for stock in stocks:

        model, scaler = get_model(stock)

        current_price = get_current_price(stock)
        predicted_price = predict_next_day(stock, model, scaler)

        print("\n========================")
        print(f"Stock: {stock}")
        print(f"Current Price: {current_price}")
        print(f"Predicted Price: {predicted_price}")
        print("========================")