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





