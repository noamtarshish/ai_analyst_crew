# ai_analyst_crew/tools/fundamental_tools.py

from crewai.tools import tool
import yfinance as yf
import openai
import os
from dotenv import load_dotenv
load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@tool("Fetch and Analyze Fundamental Data")
def analyze_fundamentals(symbol: str) -> str:
    """
    Fetches fundamental financial metrics for a stock and analyzes its health.
    Returns a financial health classification (Good / Neutral / Poor).
    """
    try:
        print(f"\n🛠️ DEBUG: Fetching fundamentals for {symbol}...")
        stock = yf.Ticker(symbol)
        info = stock.info

        try:
            last_updated = yf.download(symbol, period="1d").index[-1].strftime("%Y-%m-%d")
        except:
            last_updated = "N/A"

        metrics = {
            "PE Ratio": info.get("trailingPE", "N/A"),
            "EPS": info.get("trailingEps", "N/A"),
            "Debt/Equity": info.get("debtToEquity", "N/A"),
            "ROE": info.get("returnOnEquity", "N/A"),
            "Profit Margin": info.get("profitMargins", "N/A"),
            "Revenue Growth": info.get("revenueGrowth", "N/A")
        }

        print("📊 DEBUG: Extracted metrics:")
        for k, v in metrics.items():
            print(f" - {k}: {v}")
        print(f"📅 DEBUG: Last updated date: {last_updated}")

        prompt = f"""
You are a financial analyst. Given these financial metrics, assess the company's overall health
as Good, Neutral, or Poor. Explain briefly and summarize as markdown bullet points.

### Symbol: {symbol}
### Date: {last_updated}

### Metrics:
- PE Ratio: {metrics["PE Ratio"]}
- EPS: {metrics["EPS"]}
- Debt/Equity: {metrics["Debt/Equity"]}
- ROE: {metrics["ROE"]}
- Profit Margin: {metrics["Profit Margin"]}
- Revenue Growth: {metrics["Revenue Growth"]}

### Format your response like:
- **Rating:** Good / Neutral / Poor
- **Key Metrics Summary:** ...
- **Explanation:** ...
- **As of:** <date>
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.4,
            messages=[{"role": "user", "content": prompt}]
        )

        final_output = response.choices[0].message.content
        print("DEBUG: GPT Output:\n", final_output)
        return final_output

    except Exception as e:
        print("ERROR in fundamental analysis:", e)
        return f"Error during fundamental analysis: {e}"
