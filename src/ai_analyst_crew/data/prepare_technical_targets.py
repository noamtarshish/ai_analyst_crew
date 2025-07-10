import pandas as pd
from pathlib import Path


def prepare_technical_targets() -> pd.DataFrame:
    """
    Loads the dataset and creates target columns for future close prices (1, 5, 21 days ahead)
    for each stock symbol (NVDA, MSFT, S&P500).
    """
    # Load the data
    data_path = Path(__file__).resolve().parent.parent / "data" / "Normalized_Data.csv"
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort by symbol and date
    df = df.sort_values(by=["Symbol", "Date"]).copy()

    # Create future targets for each symbol separately
    dfs = []
    for symbol in df["Symbol"].unique():
        df_sym = df[df["Symbol"] == symbol].copy()
        df_sym["Close_t+1"] = df_sym["Close"].shift(-1)
        df_sym["Close_t+5"] = df_sym["Close"].shift(-5)
        df_sym["Close_t+21"] = df_sym["Close"].shift(-21)
        dfs.append(df_sym)

    df_full = pd.concat(dfs)

    # Remove rows where future target is missing (e.g. end of dataset)
    df_clean = df_full.dropna(subset=["Close_t+1", "Close_t+5", "Close_t+21"])

    return df_clean


df_with_targets = prepare_technical_targets()
for symbol in df_with_targets["Symbol"].unique():
    print(f"\n=== {symbol} ===")
    print(df_with_targets[df_with_targets["Symbol"] == symbol].tail(3))