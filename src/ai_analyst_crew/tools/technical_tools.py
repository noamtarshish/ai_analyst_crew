# src/ai_analyst_crew/tools/technical_tool.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from crewai.tools import tool


@tool("Technical Stock Forecast Tool")
def forecast_price(symbol: str, days_ahead: int = 1) -> str:
    """
    Uses historical normalized data and technical indicators to forecast the stock price
    for the given symbol (e.g., NVDA, MSFT, ^GSPC) X days ahead.
    """

    # Load the cleaned and normalized dataset
    data_path = Path(__file__).resolve().parent.parent / "data" / "Normalized_Data.csv"
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter by symbol
    df = df[df['Symbol'] == symbol].sort_values("Date")

    if df.empty:
        return f"No data found for symbol {symbol}"

    # Simple naive forecast based on average return
    recent_returns = df['Return'].tail(5).mean()
    last_close = df['Close'].iloc[-1]
    forecast = last_close * (1 + recent_returns * days_ahead)

    return (
        f"📈 Forecast for {symbol}:\n"
        f"- Last close: {last_close:.4f}\n"
        f"- Average return (5-day): {recent_returns:.4f}\n"
        f"- Predicted close in {days_ahead} day(s): {forecast:.4f}"
    )

@tool("Advanced Technical Forecast with XGBoost")
def forecast_with_model(symbol: str, days_ahead: int = 5) -> str:
    """
    Trains an XGBoost model using technical indicators to forecast the Close price
    for the given symbol X days ahead.
    """

    print(f"\n🛠️ DEBUG: Starting XGBoost forecast for {symbol} (t+{days_ahead})")

    # Load dataset
    data_path = Path(__file__).resolve().parent.parent / "data" / "Normalized_Data.csv"
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df["Symbol"] == symbol].sort_values("Date").copy()

    if df.empty:
        print("❌ DEBUG: No data found after filtering by symbol")
        return f"No data found for symbol: {symbol}"

    print(f"✅ DEBUG: Loaded {len(df)} rows for symbol {symbol}")
    print(f"📅 DEBUG: Data range from {df['Date'].min().date()} to {df['Date'].max().date()}")

    # Create the target column
    df["Target"] = df["Close"].shift(-days_ahead)
    df = df.dropna(subset=["Target"])

    # Feature selection
    features = [
        "Return", "SMA_5", "SMA_10", "SMA_20", "EMA_10",
        "BB_Middle", "BB_Std", "Momentum_10", "Volatility_10",
        "MACD", "MACD_Signal", "RSI_14", "LogReturn"
    ]
    df_model = df[features + ["Target"]].dropna()

    # Train/Test split
    X = df_model[features]
    y = df_model["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    print(f"📊 DEBUG: Training size: {len(X_train)}, Test size: {len(X_test)}")

    # Model training
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    # Prediction
    last_row = df_model[features].iloc[[-1]]
    predicted_price = model.predict(last_row)[0]
    last_close = df["Close"].iloc[-1]

    print(f"🔮 DEBUG: Last close: {last_close:.4f}")
    print(f"📈 DEBUG: Predicted close in {days_ahead} days: {predicted_price:.4f}")

    # Evaluate
    y_pred_test = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"📉 DEBUG: RMSE on test set: {rmse:.4f}")

    return (
        f"🤖 XGBoost Forecast for {symbol} (t+{days_ahead} days):\n"
        f"- Last Close: {last_close:.4f}\n"
        f"- Predicted Close: {predicted_price:.4f}\n"
        f"- RMSE on test data: {rmse:.4f}\n"
        f"- Model used features like RSI, MACD, Momentum and more."
    )
