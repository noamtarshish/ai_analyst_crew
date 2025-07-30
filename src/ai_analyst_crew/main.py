#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from ai_analyst_crew.crew import AiAnalystCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew.
    """
    print("Welcome to the AI Stock Analyst Recommender System! 🧑‍💻")
    symbol = input("Enter stock symbol (e.g. NVDA, MSFT, AAPL): ").strip().upper()
    days_ahead = int(input("Enter forecast horizon (1, 5, or 21): "))

    inputs = {
        'symbol': symbol,
        'days_ahead': days_ahead
    }

    try:
        result = AiAnalystCrew().crew().kickoff(inputs=inputs)
        print("\n🟢 Final Output:\n", result)

    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == '__main__':
    run()