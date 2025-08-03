import pandas as pd
import math
import logging
from fuzzywuzzy import fuzz
from helpers import calculate_price_change, calculate_volume_change
from config import (  # Importing relevant constants from the config file
    HIGH_VOLATILITY_THRESHOLD, MEDIUM_VOLATILITY_THRESHOLD,
    surge_words, FEAR_GREED_THRESHOLD,LOW_VOLUME_THRESHOLD_LARGE, LOW_VOLUME_THRESHOLD_MID, LOW_VOLUME_THRESHOLD_SMALL, analyzer
)
from datetime import datetime, timedelta
from api_clients import client, api_call_with_retries,fetch_historical_ticker_data,fetch_santiment_data_for_coin,fetch_twitter_data,fetch_fear_and_greed_index

def analyze_volume_change(volume_data, market_cap, volatility):
    """
    Analyze the volume changes of a cryptocurrency over three time periods.

    Parameters:
        volume_data (pd.DataFrame): Volume data for the cryptocurrency.
        market_cap (int): Market capitalization of the cryptocurrency.
        volatility (float): Volatility of the cryptocurrency.

    Returns:
        tuple: The volume change score and an explanation string detailing
               which periods had significant volume changes.
    """
    # Classify the market capitalization and volatility
    market_cap_class = classify_market_cap(market_cap)
    volatility_class = classify_volatility(volatility)

    # Get volume thresholds based on the market cap and volatility classification
    short_term_threshold, short_term_max, medium_term_threshold, medium_term_max, long_term_threshold, long_term_max = get_volume_thresholds(market_cap_class, volatility_class)

    # Analyze short-term, medium-term, and long-term volume changes
    short_term_change = calculate_volume_change(volume_data, period="short")
    medium_term_change = calculate_volume_change(volume_data, period="medium")
    long_term_change = calculate_volume_change(volume_data, period="long")

    volume_score = 0
    explanation_parts = []

    if short_term_change > short_term_threshold and short_term_change < short_term_max:
        volume_score += 1
        explanation_parts.append(f"Short-term volume change of {short_term_change*100:.2f}% exceeded the threshold of {short_term_threshold*100:.2f}%")
    
    if medium_term_change > medium_term_threshold and medium_term_change < medium_term_max:
        volume_score += 1
        explanation_parts.append(f"Medium-term volume change of {medium_term_change*100:.2f}% exceeded the threshold of {medium_term_threshold*100:.2f}%")
    
    if long_term_change > long_term_threshold and long_term_change < long_term_max:
        volume_score += 1
        explanation_parts.append(f"Long-term volume change of {long_term_change*100:.2f}% exceeded the threshold of {long_term_threshold*100:.2f}%")

    # Combine the explanation parts into a single string
    explanation = " | ".join(explanation_parts) if explanation_parts else "No significant volume changes detected."

    return volume_score, explanation

def analyze_price_change(price_data, market_cap, volatility):
    """
    Analyze the price changes of a cryptocurrency over three time periods.

    Parameters:
        price_data (pd.DataFrame): Price data for the cryptocurrency.
        market_cap (int): Market capitalization of the cryptocurrency.
        volatility (float): Volatility of the cryptocurrency.

    Returns:
        tuple: The price change score and an explanation string detailing
               which periods had significant price changes.
    """
    market_cap_class = classify_market_cap(market_cap)
    volatility_class = classify_volatility(volatility)

    # Unpack only the lower threshold values
    short_term_threshold, medium_term_threshold, long_term_threshold = get_price_change_thresholds(market_cap_class, volatility_class)

    # Analyze short-term, medium-term, and long-term price changes
    short_term_change = calculate_price_change(price_data, period="short")
    medium_term_change = calculate_price_change(price_data, period="medium")
    long_term_change = calculate_price_change(price_data, period="long")

    price_change_score = 0
    explanation_parts = []

    if short_term_change > short_term_threshold:
        price_change_score += 1
        explanation_parts.append(f"Short-term change of {short_term_change*100:.2f}% exceeded the threshold of {short_term_threshold*100:.2f}%")
    
    if medium_term_change > medium_term_threshold:
        price_change_score += 1
        explanation_parts.append(f"Medium-term change of {medium_term_change*100:.2f}% exceeded the threshold of {medium_term_threshold*100:.2f}%")
    
    if long_term_change > long_term_threshold:
        price_change_score += 1
        explanation_parts.append(f"Long-term change of {long_term_change*100:.2f}% exceeded the threshold of {long_term_threshold*100:.2f}%")

    explanation = " | ".join(explanation_parts) if explanation_parts else "No significant price changes detected."

    return price_change_score, explanation

def get_fuzzy_trending_score(coin_id, coin_name, trending_coins_scores):
    """
    Analyze trending coins scores and return the maximum score if a fuzzy match is found 
    between the coin ID or name and any of the trending coin tickers.

    Parameters:
        coin_id (str): The CoinPaprika ID of the cryptocurrency.
        coin_name (str): The full name of the cryptocurrency.
        trending_coins_scores (dict): A dictionary with tickers as keys and their respective scores.

    Returns:
        int: The maximum score if a fuzzy match is found, otherwise 0.
    """
    max_score = 0
    for ticker, score in trending_coins_scores.items():
        match_id = fuzz.partial_ratio(ticker.lower(), coin_id.lower())
        match_name = fuzz.partial_ratio(ticker.lower(), coin_name.lower())
        if match_id > 80 or match_name > 80:  # Threshold for considering a match
            max_score = max(max_score, score)
    return max_score

def fetch_coin_events(coin_id):
    """
    Fetches events for a given cryptocurrency from the CoinPaprika API.

    Parameters:
        coin_id (str): The ID of the cryptocurrency.

    Returns:
        list: A list of recent events for the cryptocurrency, or an empty list if the API call fails.
    """
    try:
        logging.debug(f"Fetching Event data for {coin_id}.")

        events = api_call_with_retries(client.events, coin_id=coin_id)
        
        if not events:
            logging.debug(f"No events found for {coin_id}.")
            return []

        # Filter events to include only those from the past week and exclude future dates
        one_week_ago = datetime.now() - timedelta(days=7)
        recent_events = []

        for event in events:
            event_date = datetime.strptime(event['date'], "%Y-%m-%dT%H:%M:%SZ")
            if one_week_ago <= event_date <= datetime.now():
                recent_events.append(event)

        logging.debug(f"Events found for {coin_id}: {len(recent_events)} recent events")
        return recent_events

    except Exception as e:
        logging.debug(f"Error fetching events for {coin_id}: {e}")
        return []

def has_consistent_monthly_growth(historical_df):
    """
    Returns True if the given historical data shows consistent monthly growth,
    defined as at least 18 of the last 30 days having a positive price change.
    """
    historical_df['price_change'] = historical_df['price'].pct_change()
    last_month_df = historical_df.tail(30)
    rising_days = last_month_df[last_month_df['price_change'] > 0].shape[0]
    return rising_days >= 18

def analyze_coin(coin_id, coin_name, end_date, news_df, digest_tickers, trending_coins_scores, santiment_slugs_df):
    """
    Analyzes a given cryptocurrency and returns a dictionary with various analysis scores, 
    including a score for whether the coin appears in the Sundown Digest and trending coins list.

    Parameters:
        coin_id (str): The CoinPaprika ID of the cryptocurrency.
        coin_name (str): The full name of the cryptocurrency.
        end_date (str): The end date of the period for which to fetch the historical ticker data (in YYYY-MM-DD format).
        news_df (pd.DataFrame): A DataFrame containing news articles related to the cryptocurrency.
        digest_tickers (list): A list of tickers extracted from the Sundown Digest.
        trending_coins_scores (dict): A dictionary with tickers as keys and their respective scores.
        santiment_slugs_df (pd.DataFrame): DataFrame containing Santiment slugs for various coins.

    Returns:
        dict: A dictionary with the analysis scores, cumulative score, market cap, volume, and detailed explanation.
    """
    short_term_window = 7
    medium_term_window = 30
    long_term_window = 90
    
    start_date_short_term = (datetime.now() - timedelta(days=short_term_window)).strftime('%Y-%m-%d')
    start_date_medium_term = (datetime.now() - timedelta(days=medium_term_window)).strftime('%Y-%m-%d')
    start_date_long_term = (datetime.now() - timedelta(days=long_term_window)).strftime('%Y-%m-%d')

    historical_df_short_term = fetch_historical_ticker_data(coin_id, start_date_short_term, end_date)
    historical_df_medium_term = fetch_historical_ticker_data(coin_id, start_date_medium_term, end_date)
    historical_df_long_term = fetch_historical_ticker_data(coin_id, start_date_long_term, end_date)

    # Match the coin with Santiment slugs
    santiment_slug = match_coins_with_santiment(coin_name, santiment_slugs_df)

    # If a Santiment slug is found, fetch Santiment data for the last 30 days
    if santiment_slug:
        santiment_data = fetch_santiment_data_for_coin(santiment_slug)
    else:
        santiment_data = {"dev_activity_increase": 0, "daily_active_addresses_increase": 0}


    if 'price' not in historical_df_long_term.columns or historical_df_long_term.empty:
        logging.debug(f"No valid price data available for {coin_id}.")
        return {"coin_id": coin_id, "coin_name": coin_name, "explanation": f"No valid price data available for {coin_id}."}

    twitter_df = fetch_twitter_data(coin_id)
    tweet_score = 1 if not twitter_df.empty else 0

    volatility = historical_df_long_term['price'].pct_change().std()

    # Get market cap and volume from the most recent entry)
    most_recent_market_cap = int(historical_df_long_term['market_cap'].iloc[-1])
    most_recent_volume_24h = int(historical_df_long_term['volume_24h'].iloc[-1])

    # Get price change score and detailed explanation
    price_change_score, price_change_explanation = analyze_price_change(historical_df_long_term['price'], most_recent_market_cap, volatility)

    volume_score, volume_explanation = analyze_volume_change(historical_df_long_term['volume_24h'], most_recent_market_cap, volatility)

    consistent_growth = has_consistent_weekly_growth(historical_df_short_term)
    consistent_growth_score = 1 if consistent_growth else 0

    sustained_volume_growth = has_sustained_volume_growth(historical_df_short_term)
    sustained_volume_growth_score = 1 if sustained_volume_growth else 0

    fear_and_greed_index = fetch_fear_and_greed_index()
    fear_and_greed_score = 1 if fear_and_greed_index is not None and fear_and_greed_index > FEAR_GREED_THRESHOLD else 0

    events = fetch_coin_events(coin_id)
    recent_events_count = sum(1 for event in events if datetime.strptime(event['date'], '%Y-%m-%d') <= datetime.now())
    event_score = 1 if recent_events_count > 0 else 0

    consistent_monthly_growth = has_consistent_monthly_growth(historical_df_medium_term)
    consistent_monthly_growth_score = 1 if consistent_monthly_growth else 0

    # Classify market cap
    market_cap_class = classify_market_cap(most_recent_market_cap)

    # Calculate liquidity risk based on the most recent 24-hour volume
    liquidity_risk = classify_liquidity_risk(most_recent_volume_24h, market_cap_class)

    # Integrate sentiment analysis
    if not news_df.empty:
        coin_news = news_df[news_df['coin'] == coin_name]
        sentiment_score = compute_sentiment_for_coin(coin_name, coin_news.to_dict('records'))

        # Integrate surge word analysis
        surge_score, surge_explanation = score_surge_words(coin_news, surge_words)
    else:
        sentiment_score = 0
        surge_score = 0
        surge_explanation = "No significant surge-related news detected."

    # Check if the coin is in the digest tickers
    digest_score = 1 if any(fuzz.partial_ratio(ticker.lower(), coin_id.lower()) > 80 or fuzz.partial_ratio(ticker.lower(), coin_name.lower()) > 80 for ticker in digest_tickers) else 0

    # Check if the coin is trending
    trending_score = get_fuzzy_trending_score(coin_id, coin_name, trending_coins_scores)

    # Incorporate Santiment data into the cumulative score
    santiment_score, santiment_explanation = compute_santiment_score_with_thresholds(santiment_data)

    trend_conflict_score = 1 if consistent_monthly_growth_score and not consistent_growth_score else 0

    # Calculate cumulative score and its percentage of the maximum score
    cumulative_score = (
    volume_score + tweet_score + consistent_growth_score + sustained_volume_growth_score + 
    fear_and_greed_score + event_score + price_change_score + sentiment_score + surge_score +
    digest_score + trending_score + santiment_score + consistent_monthly_growth_score +
    trend_conflict_score
    )

    # Maximum possible score (adjust as necessary)
    max_possible_score = 16

    # Calculate the cumulative score as a percentage
    cumulative_score_percentage = (cumulative_score / max_possible_score) * 100

    # Build the explanation string, including Santiment data and price change explanation
    explanation = f"{coin_name} ({coin_id}) analysis: "
    explanation += f"Liquidity Risk: {liquidity_risk}, "
    explanation += f"Price Change Score: {'Significant' if price_change_score else 'No significant change'} ({price_change_explanation}), "
    explanation += f"Volume Change Score: {'Significant' if volume_score else 'No significant change'} ({volume_explanation}), "
    explanation += f"Tweets: {'Yes' if tweet_score else 'None'}, "
    explanation += f"Consistent Price Growth: {'Yes' if consistent_growth_score else 'No'}, "
    explanation += f"Sustained Volume Growth: {'Yes' if sustained_volume_growth_score else 'No'}, "
    explanation += f"Fear and Greed Index: {fear_and_greed_index if isinstance(fear_and_greed_index, int) else 'N/A'}, "    
    explanation += f"Recent Events: {recent_events_count}, "
    explanation += f"Sentiment Score: {sentiment_score}, "
    explanation += f"Surge Keywords Score: {surge_score} ({surge_explanation}), "
    explanation += f"Santiment Score: {santiment_score} ({santiment_explanation}), "
    explanation += f"News Digest Score: {digest_score}, "
    explanation += f"Trending Score: {trending_score}, "
    explanation += f"Market Cap: {most_recent_market_cap}, "
    explanation += f"Volume (24h): {most_recent_volume_24h}, "
    explanation += f"Cumulative Surge Score: {cumulative_score} ({cumulative_score_percentage:.2f}%)"
    explanation += f"Consistent Monthly Growth: {'Yes' if consistent_monthly_growth_score else 'No'}, "
    explanation += f"Trend Conflict: {'Yes' if trend_conflict_score else 'No'} (Monthly growth without short-term support), "
    if trend_conflict_score:
        explanation += "⚠️ Potential breakout opportunity: consistent monthly growth detected without short-term trend confirmation. "
    
    # Add top 3 news headlines
    if not news_df.empty:
        coin_news = news_df[news_df['coin'] == coin_name]
        news_headlines = coin_news['title'].tolist()[:3]
        explanation += f", Top News: " + "; ".join(news_headlines)
    else:
        explanation += ", Top News: No recent news found."
   
  
    return {
        "coin_id": coin_id,
        "coin_name": coin_name,
        "market_cap": most_recent_market_cap,  # Add market cap to output
        "volume_24h": most_recent_volume_24h,  # Add volume (24h) to output
        "price_change_score": f"{price_change_score}",
        "volume_change_score": f"{volume_score}",
        "tweets": len(twitter_df) if tweet_score else "None",
        "consistent_growth": "Yes" if consistent_growth_score else "No",
        "sustained_volume_growth": "Yes" if sustained_volume_growth_score else "No",
        "fear_and_greed_index": int(fear_and_greed_index) if fear_and_greed_index is not None else None,
        "events": recent_events_count,
        "sentiment_score": sentiment_score,
        "surging_keywords_score": surge_score,
        "news_digest_score": digest_score,
        "trending_score": trending_score,
        "liquidity_risk": liquidity_risk,
        "santiment_score": santiment_score,
        "cumulative_score": cumulative_score,
        "cumulative_score_percentage": round(cumulative_score_percentage, 2),  # Rounded to 2 decimal places
        "explanation": explanation,
        "coin_news": coin_news.to_dict('records') if not news_df.empty else [],
        "trend_conflict": "Yes" if trend_conflict_score else "No",
    }


def match_coins_with_santiment(coin_name, santiment_slugs_df):
    """
    Matches a given coin name with the Santiment slugs dataframe.
    
    Parameters:
    coin_name (str): The name of the coin to match.
    santiment_slugs_df (pd.DataFrame): The dataframe containing Santiment slugs and normalized names.
    
    Returns:
    str: The Santiment slug if a match is found, else None.
    """    
    # Check if the 'name_normalized' column exists in the dataframe
    if 'name_normalized' not in santiment_slugs_df.columns:
        logging.warning(f"'name_normalized' column not found in the Santiment slugs dataframe.")
        return None

    # Look for exact matches in the normalized names
    match = santiment_slugs_df[santiment_slugs_df['name_normalized'] == coin_name]
    
    if not match.empty:
        return match['slug'].values[0]  # Return the first matching slug
    else:
        logging.info(f"No match found for {coin_name} in Santiment slugs.")
    
    return None

def get_price_change_thresholds(market_cap_class, volatility_class):
    """
    Returns the price change thresholds for a given market capitalization class and volatility class.

    Parameters:
        market_cap_class (str): The market capitalization class, one of "Large", "Mid", or "Small".
        volatility_class (str): The volatility class, one of "High", "Medium", or "Low".

    Returns:
        tuple: A tuple of three floats, representing the price change thresholds for short-term, medium-term, and long-term periods, respectively.
    """
    thresholds = {
        ("Large", "High"): (0.03, 0.02, 0.01),
        ("Large", "Low"): (0.015, 0.01, 0.005),
        ("Mid", "High"): (0.05, 0.03, 0.02),
        ("Mid", "Medium"): (0.03, 0.02, 0.015),
        ("Mid", "Low"): (0.02, 0.015, 0.01),
        ("Small", "High"): (0.07, 0.05, 0.03),
        ("Small", "Medium"): (0.05, 0.03, 0.02),
        ("Small", "Low"): (0.03, 0.02, 0.015)
    }
    return thresholds.get((market_cap_class, volatility_class), (0.03, 0.02, 0.01))


def has_sustained_volume_growth(historical_df):
    # Calculate daily volume changes
    """
    Returns True if the given historical data shows sustained volume growth,
    which is defined as at least 4 out of the last 7 days having a positive
    volume change.

    Parameters:
        historical_df (pd.DataFrame): A pandas DataFrame containing historical
            data for the cryptocurrency, with columns for 'date', 'price', and
            'volume_24h'.

    Returns:
        bool: True if the volume has been growing for at least 4 of the last 7
            days, False otherwise.
    """
    historical_df['volume_change'] = historical_df['volume_24h'].pct_change()

    # Filter the last 7 days
    last_week_df = historical_df.tail(7)
    
    # Count how many days had a positive volume change
    rising_volume_days = last_week_df[last_week_df['volume_change'] > 0].shape[0]
    
    # Consider sustained growth if at least 4 out of 7 days had rising volume
    return rising_volume_days >= 4

def classify_liquidity_risk(volume_24h, market_cap_class):
    """
    Classify liquidity risk based on trading volume.

    Parameters:
        volume_24h (float): The 24-hour trading volume of the cryptocurrency.
        market_cap_class (str): The market capitalization class, one of "Large", "Mid", or "Small".

    Returns:
        str: The classification of liquidity risk ('Low', 'Medium', 'High').
    """
    if market_cap_class == "Large":
        if volume_24h < LOW_VOLUME_THRESHOLD_LARGE:
            return "High"
        elif volume_24h < LOW_VOLUME_THRESHOLD_LARGE * 2:
            return "Medium"
        else:
            return "Low"
    elif market_cap_class == "Mid":
        if volume_24h < LOW_VOLUME_THRESHOLD_MID:
            return "High"
        elif volume_24h < LOW_VOLUME_THRESHOLD_MID * 2:
            return "Medium"
        else:
            return "Low"
    else:  # Small market cap
        if volume_24h < LOW_VOLUME_THRESHOLD_SMALL:
            return "High"
        elif volume_24h < LOW_VOLUME_THRESHOLD_SMALL * 2:
            return "Medium"
        else:
            return "Low"

def has_consistent_weekly_growth(historical_df):
    # Calculate daily price changes
    """
    Returns True if the given historical data shows consistent weekly growth,
    which is defined as at least 4 out of the last 7 days having a positive
    price change.

    Parameters:
        historical_df (pd.DataFrame): A pandas DataFrame containing historical
            data for the cryptocurrency, with columns for 'date', 'price', and
            'volume_24h'.

    Returns:
        bool: True if the price has been growing for at least 4 of the last 7
            days, False otherwise.
    """
    historical_df['price_change'] = historical_df['price'].pct_change()

    # Filter the last 7 days
    last_week_df = historical_df.tail(7)
    
    # Count how many days had a positive price change
    rising_days = last_week_df[last_week_df['price_change'] > 0].shape[0]
    
    # Consider consistent growth if at least 4 out of 7 days were rising
    return rising_days >= 4

def get_volume_thresholds(market_cap_class, volatility_class):
    """
    Returns the volume thresholds for a given market capitalization class and volatility class.

    Parameters:
        market_cap_class (str): The market capitalization class, one of "Large", "Mid", or "Small".
        volatility_class (str): The volatility class, one of "High", "Medium", or "Low".

    Returns:
        tuple: A tuple of six floats, representing the volume change thresholds for short-term, medium-term, and long-term periods, respectively.
    """
    thresholds = {
        ("Large", "High"): (2, 4, 1.5, 3, 1.2, 2),
        ("Large", "Medium"): (1.5, 3, 1.2, 2, 1.1, 1.5),
        ("Large", "Low"): (1.2, 2, 1.1, 1.5, 1, 1.2),
        ("Mid", "High"): (3, 6, 2, 4, 1.5, 2.5),
        ("Mid", "Medium"): (2, 4, 1.5, 3, 1.2, 2),
        ("Mid", "Low"): (1.5, 3, 1.2, 2, 1, 1.5),
        ("Small", "High"): (5, 10, 3, 6, 2, 4),
        ("Small", "Medium"): (3, 6, 2, 4, 1.5, 2.5),
        ("Small", "Low"): (2, 4, 1.5, 3, 1.2, 2)
    }
    return thresholds.get((market_cap_class, volatility_class), (2, 4, 1.5, 3, 1.2, 2))

def score_surge_words(news_df, surge_words):
    """
    Analyze news articles and score the presence of surge words.

    Parameters:
        news_df (pd.DataFrame): A pandas DataFrame containing news articles.
        surge_words (list): A list of words indicating a surge in the market.

    Returns:
        tuple: A tuple containing the average surge score across all news articles 
               and a detailed explanation of which articles contributed to the score.
    """
    total_surge_score = 0
    news_count = 0
    explanation = []

    if not news_df.empty:
        for _, news_item in news_df.iterrows():
            description = news_item.get('description', '')

            # Ensure description is a string and is not None
            if isinstance(description, str) and description.strip():
                surge_score = 0
                article_explanation = []

                for word in surge_words:
                    # Fuzzy match each surge word with the description
                    match_score = fuzz.partial_ratio(word.lower(), description.lower())

                    # Add to surge_score based on the match score (scale 0 to 100)
                    if match_score > 75:  # Threshold for a significant match
                        surge_score += match_score / 100.0  # Normalize the score to 0-1 range
                        article_explanation.append(f"Matched word '{word}' with score {match_score}%")

                if surge_score > 0:
                    explanation.append(f"Article: '{news_item.get('title', '')}' contributed to surge score with details: {', '.join(article_explanation)}")

                total_surge_score += surge_score
                news_count += 1

    if news_count > 0:
        average_surge_score = total_surge_score / news_count
    else:
        average_surge_score = 0.0

    return int(math.ceil(average_surge_score)), explanation

def classify_volatility(volatility):
    """
    Classify a volatility value as "High", "Medium", or "Low".

    Parameters:
        volatility (float): The volatility of the cryptocurrency.

    Returns:
        str: The classification of the volatility.
    """
    if volatility > HIGH_VOLATILITY_THRESHOLD:
        return "High"
    elif volatility > MEDIUM_VOLATILITY_THRESHOLD:
        return "Medium"
    else:
        return "Low"

def classify_market_cap(market_cap):
    """
    Classify a market capitalization as "Large", "Mid", or "Small".

    Parameters:
        market_cap (int): The market capitalization of the cryptocurrency.

    Returns:
        str: The classification of the market capitalization.
    """
    if market_cap > 10_000_000_000:
        return "Large"
    elif market_cap > 1_000_000_000:
        return "Mid"
    else:
        return "Small"

def compute_sentiment_for_coin(coin_name, news_data):
    """
    Computes the sentiment score for a given coin based on its news data.

    Parameters:
        coin_name (str): The name of the coin.
        news_data (list): A list of news items related to the coin, where each item is a dict with 'title' and 'description' keys.

    Returns:
        int: 1 if the average sentiment score is very positive (e.g., greater than 0.5), otherwise 0.
    """
    sentiments = []
    for news_item in news_data:
        description = news_item.get('description', '')
        
        # Ensure description is a string and is not empty
        if isinstance(description, str) and description.strip():
            sentiment_score = analyzer.polarity_scores(description)['compound']
            sentiments.append(sentiment_score)
    
    if sentiments:
        average_sentiment = sum(sentiments) / len(sentiments)
    else:
        average_sentiment = 0

    # Return 1 if the average sentiment is very positive (e.g., greater than 0.5), otherwise return 0
    return 1 if average_sentiment > 0.5 else 0

def compute_santiment_score_with_thresholds(santiment_data):
    """
    Computes a binary score using Santiment data by applying thresholds for each metric and provides explanations for each score.

    Parameters:
        santiment_data (dict): A dictionary with Santiment metrics data.

    Returns:
        tuple: A final score based on whether each metric exceeds its threshold and an explanation detailing the scoring.
    """
    # Define thresholds for each metric (binary scoring: 0 or 1)
    thresholds = {
        'dev_activity': 10,             # Development activity increase must be greater than 10%
        'daily_active_addresses': 5,    # Daily active addresses increase must be greater than 5%
    }

    # Extract metric values, defaulting to 0 if not available
    dev_activity = santiment_data.get('dev_activity_increase', 0)
    daily_active_addresses = santiment_data.get('daily_active_addresses_increase', 0)

    # Apply thresholds to compute binary scores (0 or 1) and explanations
    explanations = []
 
    if dev_activity > thresholds['dev_activity']:
        dev_activity_score = 1
        explanations.append(f"Development activity increase is significant: {dev_activity}% (Threshold: {thresholds['dev_activity']}%)")
    else:
        dev_activity_score = 0
        explanations.append(f"Development activity increase is low: {dev_activity}% (Threshold: {thresholds['dev_activity']}%)")

    if daily_active_addresses > thresholds['daily_active_addresses']:
        daily_active_addresses_score = 1
        explanations.append(f"Daily active addresses show growth: {daily_active_addresses}% (Threshold: {thresholds['daily_active_addresses']}%)")
    else:
        daily_active_addresses_score = 0
        explanations.append(f"Daily active addresses growth is weak: {daily_active_addresses}% (Threshold: {thresholds['daily_active_addresses']}%)")
    
    # Sum up the scores to get a total score
    total_santiment_score = (
        dev_activity_score +
        daily_active_addresses_score )

    explanation = " | ".join(explanations)  # Combine explanations into a single string

    return total_santiment_score, explanation
