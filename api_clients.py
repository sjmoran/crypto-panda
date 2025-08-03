import requests
import logging
import san  # Add the missing sanpy import for Santiment API
from config import (
    BACKOFF_FACTOR,
    MAX_RETRIES,  # For the Santiment API key
)
import re  # Add the missing import for regular expressions
from helpers import normalize_score
import time
import os
import re
import pandas as pd
import datetime 
import traceback
from datetime import datetime, timedelta
from config import COIN_PAPRIKA_API_KEY,SAN_API_KEY
from coinpaprika import client as Coinpaprika

# Set Santiment API key for sanpy
os.environ["SANAPIKEY"] = SAN_API_KEY
san.ApiConfig.api_key = SAN_API_KEY

client = Coinpaprika.Client(api_key=COIN_PAPRIKA_API_KEY)

def api_call_with_retries(api_function, *args, **kwargs):
    """
    Calls an API function with retries in case of failure.

    Args:
        api_function (function): The API function to call.
        *args: Arguments to pass to the API function.
        **kwargs: Keyword arguments to pass to the API function.

    Returns:
        The result of the API function call.

    Raises:
        Exception: If the API function call fails after MAX_RETRIES retries.
    """
    retries = 0
    wait_time = 60  # Initial wait time (seconds)
    time.sleep(25)

    while retries < MAX_RETRIES:
        try:
            return api_function(*args, **kwargs)
        except Exception as e:
            retries += 1
            if retries < MAX_RETRIES:
                logging.debug(f"Error during API call: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= BACKOFF_FACTOR
            else:
                logging.debug(f"Max retries reached. Failed to complete API call.")
                raise

def fetch_twitter_data(coin_id):
    """
    Fetches tweets for a given cryptocurrency from the CoinPaprika API.

    Parameters:
        coin_id (str): The ID of the cryptocurrency.

    Returns:
        pd.DataFrame: A pandas DataFrame containing tweets for the cryptocurrency from the past week, or an empty DataFrame if the API call fails or no tweets are found.
    """
    tweets = api_call_with_retries(client.twitter, coin_id)
    
    if not tweets:
        logging.debug(f"No tweets found for {coin_id}.")
        return pd.DataFrame()

    df = pd.DataFrame(tweets)

    if 'status' not in df.columns or 'date' not in df.columns:
        logging.debug(f"'status' or 'date' column not found in Twitter data for {coin_id}. Available columns: {df.columns.tolist()}")
        return pd.DataFrame()

    # Filter tweets from the past week
    one_week_ago = datetime.now() - timedelta(days=7)  # Naive datetime
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # Convert to naive datetime
    df = df[df['date'] >= one_week_ago]

    if df.empty:
        logging.debug(f"No recent tweets (past week) found for {coin_id}.")
        return pd.DataFrame()

    logging.debug(f"Tweets found for {coin_id} in the past week: {len(df)} tweets")

    return df

def fetch_santiment_data_for_coin(coin_slug):
    """
    Fetches relevant Santiment data for a specific coin, including:
    - Development activity increase
    - Daily active addresses change
    - Exchange inflow and outflow
    - Whale transaction count
    - Transaction volume change
    - Weighted sentiment score

    If the API call fails, returns default values to allow the rest of the pipeline to proceed.

    Parameters:
        coin_slug (str): The Santiment slug of the cryptocurrency.

    Returns:
        dict: A dictionary with fetched Santiment metrics data or default values if unavailable.
    """
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        logging.debug(f"Fetching Santiment data for {coin_slug} from {start_date} to {end_date}")

        def safe_fetch(metric_name):
            value = fetch_santiment_metric(metric_name, coin_slug, start_date, end_date)
            if value is None or not isinstance(value, (float, int)):
                logging.debug(f"Metric {metric_name} for {coin_slug} is invalid or missing. Defaulting to 0.0")
                return 0.0
            return float(value)

        return {
            "dev_activity_increase": safe_fetch('30d_moving_avg_dev_activity_change_1d'),
            "daily_active_addresses_increase": safe_fetch('active_addresses_24h_change_30d'),
            "exchange_inflow_usd": safe_fetch('exchange_inflow_usd'),
            "exchange_outflow_usd": safe_fetch('exchange_outflow_usd'),
            "whale_transaction_count_100k_usd_to_inf": safe_fetch('whale_transaction_count_100k_usd_to_inf_1d'),
            "transaction_volume_usd_change_1d": safe_fetch('transaction_volume_usd_change_1d'),
            "sentiment_weighted_total": safe_fetch('sentiment_weighted_total_1d'),
        }

    except Exception as e:
        logging.error(f"Error fetching Santiment data for {coin_slug}: {e}")
        # Return default values if there's an error
        return {
            "dev_activity_increase": 0.0,
            "daily_active_addresses_increase": 0.0,
            "exchange_inflow_usd": 0.0,
            "exchange_outflow_usd": 0.0,
            "whale_transaction_count_100k_usd_to_inf": 0.0,
            "transaction_volume_usd_change_1d": 0.0,
            "sentiment_weighted_total": 0.0,
        }
    
def get_sundown_digest():
    """
    Fetches the Sundown Digest from CryptoNews API.

    Returns:
        dict: A dictionary containing the Sundown Digest data.
    """
    url = f"https://cryptonews-api.com/api/v1/sundown-digest?page=1&token={os.getenv('CRYPTO_NEWS_API_KEY')}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        sundown_digest = response.json()
        return sundown_digest.get('data', [])
    except requests.exceptions.HTTPError as http_err:
        logging.debug(f"HTTP error occurred: {http_err}")
    except Exception as err:
        logging.debug(f"An error occurred: {err}")
    return []

def filter_active_and_ranked_coins(coins, max_coins, rank_threshold=1000):
    """
    Filters the list of coins by rank, activity status, and new status, selecting up to max_coins.

    Parameters:
        coins (list): List of coins with rank, activity status, and new status information.
        max_coins (int): The maximum number of coins to return.
        rank_threshold (int): The maximum rank a coin must have to be included (e.g., rank <= 1000).

    Returns:
        list: Filtered list of coins that are active, not new, and ranked within the rank_threshold.
    """
    # Filter out coins that are not active, are new, or have a rank above the rank_threshold
    active_ranked_coins = [coin for coin in coins if coin.get('is_active', False) and not coin.get('is_new', True) and coin.get('rank', None) <= rank_threshold]

    # Limit the list to max_coins
    return active_ranked_coins[:max_coins]

def fetch_santiment_slugs():
    """
    Fetch the available slugs from Santiment using the sanpy API.

    Returns:
        pd.DataFrame: DataFrame containing Santiment slugs and project information.
    """
    try:
        # Fetch available slugs using sanpy API
        all_projects = san.get(
            "projects/all",
            interval="1d",
            columns=["slug", "name", "ticker", "infrastructure", "mainContractAddress"]
        )
        projects_df = pd.DataFrame(all_projects)

        # Normalize the coin names for matching
        projects_df['name_normalized'] = projects_df['name'].apply(lambda x: re.sub(r'\W+', '', x.lower()))

        logging.info(f"Fetched {len(projects_df)} Santiment slugs")
        return projects_df

    except Exception as e:
        logging.error(f"Error fetching Santiment slugs: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def fetch_news_for_past_week(tickers_dict):
    """
    Fetches news for each coin in the given tickers dictionary for the past week.

    Parameters:
        tickers_dict (dict): A dictionary mapping the coin names to their tickers.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the news articles fetched from the API, with columns 'coin', 'date', 'title', 'description', 'url', and 'source'.
    """
    all_news = []

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=7)

    for coin_name, coin_ticker in tickers_dict.items():
        logging.debug(f"Fetching news for {coin_name} ({coin_ticker})...")

        formatted_date = end_date.strftime('%Y%m%d')
        week_start = (end_date - timedelta(days=7)).strftime('%m%d%Y')
        week_end = end_date.strftime('%m%d%Y')
        date_str = f"{week_start}-{week_end}"

        url = f"https://cryptonews-api.com/api/v1?tickers={coin_ticker}&items=1&date={date_str}&sortby=rank&token={os.getenv('CRYPTO_NEWS_API_KEY')}"
        response = requests.get(url)
        if response.status_code == 200:
                news_data = response.json()
                if 'data' in news_data and news_data['data']:
                    for article in news_data['data']:
                        all_news.append({
                            "coin": coin_name,
                            "date": formatted_date,  # Log the specific day
                            "title": article["title"],
                            "description": article.get("text", ""),
                            "url": article["news_url"],
                            "source": article["source_name"]
                        })
                else:
                    logging.debug(f"No news for {coin_name} between {week_start} and {week_end}.")
        else:
                logging.debug(f"Failed to fetch news for {coin_name} between {week_start} and {week_end}. Status Code: {response.status_code}")
            
        time.sleep(1)
        end_date -= timedelta(days=1)

    return pd.DataFrame(all_news)

def fetch_santiment_metric(metric, coin_slug, start_date, end_date):
    """
    Fetches Santiment metric if it's part of the free metrics plan.
    If fetching fails, returns None, allowing the rest of the code to continue.

    Parameters:
        metric (str): The metric to fetch.
        coin_slug (str): The Santiment slug of the cryptocurrency.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        float or None: The fetched data if available, or None if the API call fails.
    """
    try:
        logging.debug(f"Fetching Santiment metric: {metric} for coin: {coin_slug}")
        result = san.get(metric, slug=coin_slug, from_date=start_date, to_date=end_date)
        if not result.empty:
            # Extract the latest value
            return result.iloc[-1]['value']
        else:
            logging.debug(f"No data found for metric {metric} for coin {coin_slug}")
            return None
    except Exception as e:
        logging.error(f"Error fetching Santiment metric: {metric} for {coin_slug}: {e}")
        return None

def fetch_trending_coins_scores():
    """
    Fetches trending coins from CryptoNews API and calculates their sentiment scores.
    Normalizes the scores between 0 and 3 and returns a dictionary with the trending coins and their normalized scores.

    Returns:
        dict: A dictionary with the trending coins as keys and their normalized scores as values.
    """
    url = f"https://cryptonews-api.com/api/v1/top-mention?&date=last7days&token={os.getenv('CRYPTO_NEWS_API_KEY')}"
    response = requests.get(url)
    trending_data = response.json()['data']['all']

    trending_coins_scores = {}
    raw_scores = {}

    # Calculate raw scores
    for item in trending_data:
        ticker = item['ticker'].lower()
        sentiment_score = item['sentiment_score']
        total_mentions = item['total_mentions']
        raw_score = sentiment_score * total_mentions
        raw_scores[ticker] = raw_score

    # Determine min and max raw scores for normalization
    min_raw_score = min(raw_scores.values())
    max_raw_score = max(raw_scores.values())

    # Normalize scores between 0 and 3
    for ticker, raw_score in raw_scores.items():
        trending_coins_scores[ticker] = normalize_score(raw_score, min_raw_score, max_raw_score)

    return trending_coins_scores

def fetch_fear_and_greed_index():
    """
    Fetches the current Fear and Greed Index value from the Alternative.me API.

    Returns:
        int: The current Fear and Greed Index value, or None if the API call fails.
    """
    try:
        response = api_call_with_retries(requests.get, 'https://api.alternative.me/fng/')
        logging.debug(f"Fear and Greed API response status: {response.status_code}")
        logging.debug(f"Fear and Greed API raw response: {response.text}")

        response.raise_for_status()
        data = response.json()

        if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
            value = int(data['data'][0]['value'])
            logging.debug(f"Fear and Greed Index value: {value}")
            return value
        else:
            logging.debug("Fear and Greed Index data missing or malformed in response.")
            return None

    except requests.exceptions.RequestException as e:
        logging.debug(f"HTTP request error fetching Fear and Greed Index: {e}")
    except ValueError as e:
        logging.debug(f"Value error when parsing Fear and Greed Index: {e}")
    except Exception as e:
        logging.debug(f"Unexpected error fetching Fear and Greed Index: {e}")
    return None

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


def fetch_historical_ticker_data(coin_id, start_date, end_date):
    """
    Fetches historical ticker data for the specified coin and date range.

    Parameters:
        coin_id (str): The CoinPaprika ID of the cryptocurrency.
        start_date (str): The start date of the period for which to fetch the ticker data (in YYYY-MM-DD format).
        end_date (str): The end date of the period for which to fetch the ticker data (in YYYY-MM-DD format).

    Returns:
        pandas.DataFrame: A DataFrame containing the historical ticker data, with columns 'date', 'price', 'coin_id', 'volume_24h', and 'market_cap'. 
                          If the API call fails, an empty DataFrame is returned.
    """
    try:
        logging.debug(f"Fetching historical data for {coin_id} from {start_date} to {end_date}")
        
        # API call with retries
        historical_ticker = api_call_with_retries(client.historical, coin_id=coin_id, start=start_date, end=end_date, interval="1d", quote="usd")

        if isinstance(historical_ticker, list) and historical_ticker:
            df = pd.DataFrame(historical_ticker)

            if 'price' in df.columns and 'volume_24h' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                df['coin_id'] = coin_id
                df = df[['date', 'price', 'coin_id', 'volume_24h', 'market_cap']]
                df = df.sort_values(by='date')  # Sort the DataFrame by the 'date' column
                logging.debug(f"Successfully fetched and processed historical data for {coin_id}")
                return df
            else:
                logging.debug(f"Missing expected columns in historical data for {coin_id}. Columns: {df.columns}")
                return pd.DataFrame()
        else:
            logging.debug(f"Unexpected data format returned for {coin_id}: {historical_ticker}")
            return pd.DataFrame()

    except Exception as e:
        logging.debug(f"Error fetching historical data for {coin_id}: {e}")
        logging.debug(traceback.format_exc())  # Include stack trace for further debugging
        return pd.DataFrame()
     
   