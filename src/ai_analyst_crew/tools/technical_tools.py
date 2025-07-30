from crewai.tools import tool

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def compute_technical_indicators(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Volatility_10'] = df['Return'].rolling(window=10).std()
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)
    return df

def normalize_features(df):
    features_to_scale = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_5', 'SMA_10', 'SMA_20', 'EMA_10',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Std',
        'Momentum_10', 'Volatility_10',
        'MACD', 'MACD_Signal',
        'RSI_14', 'Return', 'LogReturn'
    ]
    features_to_scale = [col for col in features_to_scale if col in df.columns]
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])
    return df_scaled

def fetch_and_prepare_data(symbol: str, years_back: int = 5):
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years_back)
    df = yf.Ticker(symbol).history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval="1d", auto_adjust=True)
    if df.empty:
        raise ValueError(f"Symbol {symbol} not found or no data available.")
    df = df.reset_index()  # Get 'Date' column
    df['Symbol'] = symbol
    # Rename columns if needed to match exactly
    df.rename(columns={
        'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
    }, inplace=True)
    df = compute_technical_indicators(df)
    df = normalize_features(df)
    # סדר עמודות
    ordered_columns = [
        'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'SMA_5', 'SMA_10', 'SMA_20',
        'EMA_10', 'BB_Middle', 'BB_Std', 'BB_Upper', 'BB_Lower', 'Momentum_10', 'Volatility_10',
        'MACD', 'MACD_Signal', 'RSI_14', 'LogReturn'
    ]
    # Only keep columns that exist (for robustness)
    df = df[[c for c in ordered_columns if c in df.columns]]
    return df

@tool("Multi-Model Technical Forecast")
def forecast_with_multiple_models(symbol: str, days_ahead: int = 5) -> str:
    """
    Runs multiple models (XGBoost, Naive forecast, ARIMA) and compares results.
    Returns best prediction and model performance.
    """
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings("ignore")

    print(f"\n🧠 DEBUG: Running multi-model forecast for {symbol} (t+{days_ahead})")

    # Load data
    # data_path = Path(__file__).resolve().parent.parent / "data" / "Normalized_Data.csv"
    # df = pd.read_csv(data_path)
    df = fetch_and_prepare_data(symbol)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df["Symbol"] == symbol].sort_values("Date").copy()

    if df.empty:
        return f"No data found for {symbol}"

    last_close = df["Close"].iloc[-1]

    # ---------- Model 1: XGBoost ----------
    try:
        df["Target"] = df["Close"].shift(-days_ahead)
        features = [
            "Return", "SMA_5", "SMA_10", "SMA_20", "EMA_10",
            "BB_Middle", "BB_Std", "Momentum_10", "Volatility_10",
            "MACD", "MACD_Signal", "RSI_14", "LogReturn"
        ]
        df_model = df[features + ["Target"]].dropna()
        X = df_model[features]
        y = df_model["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(df_model[features].iloc[[-1]])[0]
        xgb_rmse = sqrt(mean_squared_error(y_test, xgb_model.predict(X_test)))
    except Exception as e:
        xgb_pred, xgb_rmse = None, float("inf")
        print(f"XGBoost failed: {e}")

    # ---------- Model 2: Naive ----------
    try:
        mean_return = df['Return'].tail(5).mean()
        naive_pred = last_close * (1 + mean_return * days_ahead)
        naive_rmse = abs(naive_pred - last_close)
    except Exception as e:
        naive_pred, naive_rmse = None, float("inf")
        print(f"Naive failed: {e}")

    # ---------- Model 3: ARIMA ----------
    try:
        close_series = df['Close'].dropna()
        if len(close_series) < (days_ahead + 10):
            raise ValueError("Not enough data for ARIMA")

        arima_model = ARIMA(close_series, order=(5, 1, 0)).fit()
        forecast_values = arima_model.forecast(steps=days_ahead)
        arima_forecast = forecast_values.iloc[-1]

        fitted_vals = arima_model.fittedvalues
        if len(fitted_vals) >= 100:
            arima_rmse = sqrt(mean_squared_error(close_series.iloc[-100:], fitted_vals[-100:]))
        else:
            arima_rmse = sqrt(mean_squared_error(close_series, fitted_vals))
    except Exception as e:
        arima_forecast, arima_rmse = None, float("inf")
        print(f"ARIMA failed: {e}")

    # ---------- Best Model ----------
    models = {
        "XGBoost": (xgb_pred, xgb_rmse),
        "Naive": (naive_pred, naive_rmse),
        "ARIMA": (arima_forecast, arima_rmse)
    }
    best_model = min(models.items(), key=lambda kv: kv[1][1])
    best_pred = best_model[1][0]
    change_percent = ((best_pred - last_close) / last_close * 100) if best_pred else None

    # ---------- Format Output ----------
    print("📊 Model Comparison:")
    for name, (pred, rmse) in models.items():
        pred_str = f"{pred:.4f}" if pred is not None else "N/A"
        rmse_str = f"{rmse:.4f}" if rmse != float("inf") else "N/A"
        print(f" - {name}: Prediction = {pred_str} | RMSE ≈ {rmse_str}")

    lines = [
        f"📊 Technical Forecast for {symbol} (t+{days_ahead} days):",
        f"- Last Close Price: {last_close:.4f}\n"
    ]
    for name, (pred, rmse) in models.items():
        pred_str = f"{pred:.4f}" if pred is not None else "N/A"
        rmse_str = f"{rmse:.4f}" if rmse != float("inf") else "N/A"
        lines.append(f"- {name}: {pred_str} (RMSE: {rmse_str})")

    lines.append("")
    lines.append(f"✅ Best Model: {best_model[0]}")
    if best_pred:
        lines.append(f"📈 Final Forecast: {best_pred:.4f}")
        lines.append(f"📊 Forecast change from last close: {change_percent:+.2f}% {'📈' if change_percent > 0 else '📉'}")
    else:
        lines.append("📈 Final Forecast: N/A")

    return "\n".join(lines)
