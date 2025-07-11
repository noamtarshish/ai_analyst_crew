# # src/ai_analyst_crew/tools/technical_tool.py
#
# import pandas as pd
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# from crewai.tools import tool
#
#
# @tool("Technical Stock Forecast Tool")
# def forecast_price(symbol: str, days_ahead: int = 1) -> str:
#     """
#     Uses historical normalized data and technical indicators to forecast the stock price
#     for the given symbol (e.g., NVDA, MSFT, ^GSPC) X days ahead.
#     """
#
#     # Load the cleaned and normalized dataset
#     data_path = Path(__file__).resolve().parent.parent / "data" / "Normalized_Data.csv"
#     df = pd.read_csv(data_path)
#     df['Date'] = pd.to_datetime(df['Date'])
#
#     # Filter by symbol
#     df = df[df['Symbol'] == symbol].sort_values("Date")
#
#     if df.empty:
#         return f"No data found for symbol {symbol}"
#
#     # Simple naive forecast based on average return
#     recent_returns = df['Return'].tail(5).mean()
#     last_close = df['Close'].iloc[-1]
#     forecast = last_close * (1 + recent_returns * days_ahead)
#
#     return (
#         f"📈 Forecast for {symbol}:\n"
#         f"- Last close: {last_close:.4f}\n"
#         f"- Average return (5-day): {recent_returns:.4f}\n"
#         f"- Predicted close in {days_ahead} day(s): {forecast:.4f}"
#     )
#
# @tool("Advanced Technical Forecast with XGBoost")
# def forecast_with_model(symbol: str, days_ahead: int = 5) -> str:
#     """
#     Trains an XGBoost model using technical indicators to forecast the Close price
#     for the given symbol X days ahead.
#     """
#
#     print(f"\n🛠️ DEBUG: Starting XGBoost forecast for {symbol} (t+{days_ahead})")
#
#     # Load dataset
#     data_path = Path(__file__).resolve().parent.parent / "data" / "Normalized_Data.csv"
#     df = pd.read_csv(data_path)
#     df['Date'] = pd.to_datetime(df['Date'])
#     df = df[df["Symbol"] == symbol].sort_values("Date").copy()
#
#     if df.empty:
#         print("❌ DEBUG: No data found after filtering by symbol")
#         return f"No data found for symbol: {symbol}"
#
#     print(f"✅ DEBUG: Loaded {len(df)} rows for symbol {symbol}")
#     print(f"📅 DEBUG: Data range from {df['Date'].min().date()} to {df['Date'].max().date()}")
#
#     # Create the target column
#     df["Target"] = df["Close"].shift(-days_ahead)
#     df = df.dropna(subset=["Target"])
#
#     # Feature selection
#     features = [
#         "Return", "SMA_5", "SMA_10", "SMA_20", "EMA_10",
#         "BB_Middle", "BB_Std", "Momentum_10", "Volatility_10",
#         "MACD", "MACD_Signal", "RSI_14", "LogReturn"
#     ]
#     df_model = df[features + ["Target"]].dropna()
#
#     # Train/Test split
#     X = df_model[features]
#     y = df_model["Target"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
#
#     print(f"📊 DEBUG: Training size: {len(X_train)}, Test size: {len(X_test)}")
#
#     # Model training
#     model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
#     model.fit(X_train, y_train)
#
#     # Prediction
#     last_row = df_model[features].iloc[[-1]]
#     predicted_price = model.predict(last_row)[0]
#     last_close = df["Close"].iloc[-1]
#
#     print(f"🔮 DEBUG: Last close: {last_close:.4f}")
#     print(f"📈 DEBUG: Predicted close in {days_ahead} days: {predicted_price:.4f}")
#
#     # Evaluate
#     y_pred_test = model.predict(X_test)
#     rmse = sqrt(mean_squared_error(y_test, y_pred_test))
#
#     print(f"📉 DEBUG: RMSE on test set: {rmse:.4f}")
#
#     return (
#         f"🤖 XGBoost Forecast for {symbol} (t+{days_ahead} days):\n"
#         f"- Last Close: {last_close:.4f}\n"
#         f"- Predicted Close: {predicted_price:.4f}\n"
#         f"- RMSE on test data: {rmse:.4f}\n"
#         f"- Model used features like RSI, MACD, Momentum and more."
#     )

from crewai.tools import tool

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
    data_path = Path(__file__).resolve().parent.parent / "data" / "Normalized_Data.csv"
    df = pd.read_csv(data_path)
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
        print(f"❌ XGBoost failed: {e}")

    # ---------- Model 2: Naive ----------
    try:
        mean_return = df['Return'].tail(5).mean()
        naive_pred = last_close * (1 + mean_return * days_ahead)
        naive_rmse = abs(naive_pred - last_close)
    except Exception as e:
        naive_pred, naive_rmse = None, float("inf")
        print(f"❌ Naive failed: {e}")

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
        print(f"❌ ARIMA failed: {e}")

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
