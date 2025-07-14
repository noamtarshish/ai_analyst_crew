from crewai.tools import tool

@tool("Generate Final Recommendation")
def generate_final_recommendation(
    technical_model: str,
    forecast_price: float,
    last_close_price: float,
    sentiment_score: float,
    fundamental_rating: str
) -> str:
    """
    Generates a recommendation using tiered scoring logic per factor:
    sentiment, price change (computed internally), and fundamentals.
    """

    # --- Calculate actual % price change ---
    price_change = (forecast_price - last_close_price) / last_close_price
    price_pct = price_change * 100

    print("\n🧠 DEBUG: [Final Recommendation Tool] Inputs received:")
    print(f" - Technical Model: {technical_model}")
    print(f" - Forecast Price: {forecast_price}")
    print(f" - Last Close Price: {last_close_price}")
    print(f" - Price Change: {price_change:.4f} ({price_pct:.2f}%)")
    print(f" - Sentiment Score: {sentiment_score}")
    print(f" - Fundamental Rating: {fundamental_rating}")

    # --- Sentiment Tier ---
    if sentiment_score >= 0.7:
        sentiment_points = 2
    elif sentiment_score >= 0.4:
        sentiment_points = 1
    elif sentiment_score <= -0.7:
        sentiment_points = -2
    elif sentiment_score <= -0.4:
        sentiment_points = -1
    else:
        sentiment_points = 0

    # --- Price Change Tier ---
    if price_pct >= 2.0:
        price_points = 2
    elif price_pct >= 0.5:
        price_points = 1
    elif price_pct <= -2.0:
        price_points = -2
    elif price_pct <= -0.5:
        price_points = -1
    else:
        price_points = 0

    # --- Fundamental Tier ---
    fr = fundamental_rating.lower()
    if fr == "good":
        fundamentals_points = 2
    elif fr == "poor":
        fundamentals_points = -2
    else:
        fundamentals_points = 0

    # --- Final Score ---
    score = sentiment_points + price_points + fundamentals_points

    print(f"🧠 DEBUG: Score Breakdown → Sentiment: {sentiment_points}, Price: {price_points}, Fundamentals: {fundamentals_points}")
    print(f"🧠 DEBUG: Total Score = {score}")

    # --- Recommendation Logic ---
    if score >= 4:
        recommendation = "Strong BUY"
    elif score >= 2:
        recommendation = "BUY"
    elif score == 1:
        recommendation = "Speculative BUY"
    elif score == 0:
        recommendation = "HOLD - Monitor Closely"
    elif score == -1:
        recommendation = "Speculative SELL"
    elif score >= -3:
        recommendation = "SELL"
    else:
        recommendation = "Strong SELL"

    return (
        f"📊 Final Recommendation Engine\n"
        f"- Technical model: {technical_model}\n"
        f"- Price change: {price_pct:.2f}%\n"
        f"- Sentiment score: {sentiment_score:.2f}\n"
        f"- Fundamental rating: {fundamental_rating.capitalize()}\n\n"
        f"✅ Tiered Score: {score} → Recommendation: {recommendation}"
    )
