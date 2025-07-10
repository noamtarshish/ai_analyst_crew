from crewai.tools import tool
import random


# Helper Functions

from datetime import datetime, timedelta
import os, requests
import re

from dotenv import load_dotenv
load_dotenv()


def fetch_headlines_from_serper(symbol: str, num_results: int = 5):
    url = "https://google.serper.dev/news"
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json"
    }
    payload = {"q": f"{symbol} stock"}
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()

    now = datetime.utcnow()
    headlines = []

    for item in data.get("news", [])[:num_results]:
        raw_date = item.get("date", "")
        title = item.get("title", "")
        source = item.get("source", "")
        url = item.get("link", "")

        # Try to parse relative date
        try:
            if "minute" in raw_date:
                pub_date = now - timedelta(minutes=int(raw_date.split()[0]))
            elif "hour" in raw_date:
                pub_date = now - timedelta(hours=int(raw_date.split()[0]))
            elif "day" in raw_date:
                pub_date = now - timedelta(days=int(raw_date.split()[0]))
            else:
                pub_date = now
        except:
            pub_date = now

        headlines.append({
            "title": title,
            "source": source,
            "date": pub_date.strftime("%Y-%m-%d"),
            "url": url
        })

    return headlines


from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_sentiment_with_openai(headlines: list[dict]):
    formatted = "\n".join(
        f"- {h['title']} (Source: {h['source']}, Date: {h['date']}, URL: {h['url']})"
        for h in headlines
    )

    prompt = f"""
You are a financial sentiment analyst.

Your task is to analyze the following news headlines related to a stock. 
Classify the overall sentiment as Positive, Negative, or Neutral, and assign a sentiment score between -1.0 and +1.0.

Also:
- Select the most impactful headline.
- Justify your sentiment label briefly.

### Headlines:
{formatted}

### Return in this format:
Headline: <most impactful headline>
Source: <source name>
Date: <YYYY-MM-DD>
URL: <original link to the article>
Sentiment: Positive/Negative/Neutral
Score: +X.XX
Explanation: <short justification for the sentiment>
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content




@tool("Sentiment Analysis of News Headlines")
def analyze_sentiment_from_news(symbol: str) -> str:
    """Retrieves news headlines using Serper and analyzes their sentiment using OpenAI."""
    try:
        print(f"\n🛠️ DEBUG: Running sentiment analysis for {symbol}")
        headlines = fetch_headlines_from_serper(symbol)
        if not headlines:
            return f"No headlines found for {symbol}."
        print("📰 Headlines:", headlines)
        result = analyze_sentiment_with_openai(headlines)
        return result
    except Exception as e:
        return f"Error during sentiment analysis: {e}"








# MOCKED SENTIMENT ANALYSIS TOOL

# @tool("Sentiment Analysis of News Headlines")
# def analyze_sentiment_from_news(symbol: str) -> str:
#     """
#     Analyzes recent (mocked) news headlines for a given stock symbol and returns
#     sentiment score (-1 to 1), top headline, and final sentiment label.
#     """
#
#     print(f"\n🛠️ DEBUG: Running sentiment analysis for {symbol}")
#
#     # Mocked headlines per stock
#     headlines_map = {
#         "NVDA": [
#             "NVIDIA posts record revenue in Q2",
#             "New GPUs spark excitement in gaming market",
#             "Analysts predict strong growth for NVIDIA"
#         ],
#         "MSFT": [
#             "Microsoft AI investments worry competitors",
#             "Teams integration drives Office 365 adoption",
#             "Microsoft's cloud growth exceeds expectations"
#         ],
#         "^GSPC": [
#             "S&P500 slips as inflation data weighs in",
#             "Fed decision keeps markets on edge",
#             "Mixed earnings season drags the index"
#         ]
#     }
#
#     headlines = headlines_map.get(symbol.upper(), [
#         f"No major news found for {symbol}. Market seems neutral."
#     ])
#
#     # Choose top headline
#     top_headline = random.choice(headlines)
#
#     # Mock sentiment logic
#     positive_words = ["record", "growth", "strong", "exceeds", "excitement", "adoption"]
#     negative_words = ["worry", "slips", "drags", "inflation", "edge"]
#
#     score = 0
#     for word in positive_words:
#         if word in top_headline.lower():
#             score += 1
#     for word in negative_words:
#         if word in top_headline.lower():
#             score -= 1
#
#     score = max(min(score / 2, 1), -1)  # Normalize
#
#     if score > 0.2:
#         label = "Positive"
#     elif score < -0.2:
#         label = "Negative"
#     else:
#         label = "Neutral"
#
#     print(f"📰 Top Headline: {top_headline}")
#     print(f"📊 Sentiment Score: {score:.2f} → {label}")
#
#     return (
#         f"💬 Sentiment Analysis for {symbol}:\n"
#         f"- Top Headline: {top_headline}\n"
#         f"- Sentiment Score: {score:.2f}\n"
#         f"- Final Sentiment: {label}"
#     )
