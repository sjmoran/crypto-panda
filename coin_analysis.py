#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Mapping, Optional, Tuple, Union
import pandas as pd
from fuzzywuzzy import fuzz, process

from config import (  # Importing relevant constants from the config file
    HIGH_VOLATILITY_THRESHOLD, MEDIUM_VOLATILITY_THRESHOLD,
    MAX_POSSIBLE_SCORE, surge_words, FEAR_GREED_THRESHOLD,
    LOW_VOLUME_THRESHOLD_LARGE, LOW_VOLUME_THRESHOLD_MID, LOW_VOLUME_THRESHOLD_SMALL, analyzer
)

# Production-grade clients (with retries & UTC handling)
from api_clients import (
    call_with_retries,
    fetch_historical_ticker_data,
    fetch_santiment_data_for_coin,
    fetch_twitter_data,
    fetch_fear_and_greed_index,
    fetch_coin_events as fetch_coin_events_api,  # imported, used by wrapper below
)

def calculate_price_change(price_data: pd.Series, period: str = "short", span: int = 7) -> Union[float, None]:
    """
    Calculate percentage change in price over a given period with EMA smoothing.
    """
    smoothed_data = price_data.ewm(span=span, adjust=False).mean()

    if period == "short":
        period_data = smoothed_data.tail(7)
    elif period == "medium":
        period_data = smoothed_data.tail(30)
    else:
        period_data = smoothed_data.tail(90)

    start_price = period_data.iloc[0]
    end_price = period_data.iloc[-1]

    if start_price == 0:
        logger.warning("Start price is 0 → cannot compute percentage change.")
        return None

    change = (end_price - start_price) / start_price
    logger.debug(f"Price change ({period}): {change:.4f}")
    return change


def calculate_volume_change(volume_data: pd.Series, period: str = "short", span: int = 7) -> Union[float, None]:
    """
    Calculate percentage change in volume over a given period with EMA smoothing.
    """
    smoothed_data = volume_data.ewm(span=span, adjust=False).mean()

    if period == "short":
        period_data = smoothed_data.tail(7)
    elif period == "medium":
        period_data = smoothed_data.tail(30)
    else:
        period_data = smoothed_data.tail(90)

    start_volume = period_data.iloc[0]
    end_volume = period_data.iloc[-1]

    if start_volume == 0:
        logger.warning("Start volume is 0 → cannot compute percentage change.")
        return None

    change = (end_volume - start_volume) / start_volume
    logger.debug(f"Volume change ({period}): {change:.4f}")
    return change


# ============================
# Logging (console + single rolling file per script)
# ============================

def setup_logging(name: str,
                  log_dir: Union[str, Path] = None,
                  level: str = None) -> logging.Logger:
    """
    Create a logger that writes to console and a single per-script logfile.
    - File path: <log_dir>/<script_stem>.log (overwrites each run)
    - UTC timestamps, ISO-like format
    """
    base_dir = Path(__file__).resolve().parent
    log_dir = Path(log_dir or (base_dir / "../logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Overwrite same file each run (rolling single file per script)
    log_path = log_dir / f"{Path(__file__).stem}.log"

    # Level from arg or env
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level_val = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level_val)
    logger.propagate = False

    # Clear existing handlers (avoid dupes on re-imports)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = "%(asctime)sZ [%(levelname)s] %(name)s | %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level_val)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (overwrite each run)
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8", delay=False)
    fh.setLevel(level_val)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Logging started → {log_path} (level={level_name})")
    return logger

# Instantiate module logger
logger = setup_logging(__name__)

# ----------------------------
# Small time helper
# ----------------------------

def utcnow() -> datetime:
    """UTC-aware 'now'."""
    return datetime.now(timezone.utc)

# ----------------------------
# ANALYSIS FUNCTIONS
# ----------------------------

def analyze_volume_change(volume_data: pd.Series, market_cap: int, volatility: float) -> Tuple[int, str]:
    """
    Analyze the volume changes of a cryptocurrency over three time periods.

    Returns:
        (volume_score, explanation)
    """
    market_cap_class = classify_market_cap(market_cap)
    volatility_class = classify_volatility(volatility)

    short_thr, short_max, med_thr, med_max, long_thr, long_max = get_volume_thresholds(market_cap_class, volatility_class)

    short_term_change = calculate_volume_change(volume_data, period="short")
    medium_term_change = calculate_volume_change(volume_data, period="medium")
    long_term_change = calculate_volume_change(volume_data, period="long")

    volume_score = 0
    parts: List[str] = []

    if short_thr < short_term_change < short_max:
        volume_score += 1
        parts.append(f"Short-term volume +{short_term_change*100:.2f}% (> {short_thr*100:.2f}%)")
    if med_thr < medium_term_change < med_max:
        volume_score += 1
        parts.append(f"Medium-term volume +{medium_term_change*100:.2f}% (> {med_thr*100:.2f}%)")
    if long_thr < long_term_change < long_max:
        volume_score += 1
        parts.append(f"Long-term volume +{long_term_change*100:.2f}% (> {long_thr*100:.2f}%)")

    explanation = " | ".join(parts) if parts else "No significant volume changes detected."
    return volume_score, explanation


def analyze_price_change(price_data: pd.Series, market_cap: int, volatility: float) -> Tuple[int, str]:
    """
    Analyze the price changes of a cryptocurrency over three time periods.

    Returns:
        (price_change_score, explanation)
    """
    market_cap_class = classify_market_cap(market_cap)
    volatility_class = classify_volatility(volatility)

    short_thr, med_thr, long_thr = get_price_change_thresholds(market_cap_class, volatility_class)

    st = calculate_price_change(price_data, period="short")
    mt = calculate_price_change(price_data, period="medium")
    lt = calculate_price_change(price_data, period="long")

    score = 0
    parts: List[str] = []
    if st > short_thr:
        score += 1
        parts.append(f"Short-term +{st*100:.2f}% (> {short_thr*100:.2f}%)")
    if mt > med_thr:
        score += 1
        parts.append(f"Medium-term +{mt*100:.2f}% (> {med_thr*100:.2f}%)")
    if lt > long_thr:
        score += 1
        parts.append(f"Long-term +{lt*100:.2f}% (> {long_thr*100:.2f}%)")

    explanation = " | ".join(parts) if parts else "No significant price changes detected."
    return score, explanation


def get_fuzzy_trending_score(coin_id: str, coin_name: str, trending_coins_scores: Mapping[str, float]) -> float:
    """
    Fuzzy-match coin_id/name to trending tickers; return max score if matched.
    """
    max_score = 0.0
    cid = coin_id.lower()
    cname = coin_name.lower()
    for ticker, score in trending_coins_scores.items():
        t = str(ticker).strip().lower()
        if not t:
            continue
        if fuzz.partial_ratio(t, cid) > 80 or fuzz.partial_ratio(t, cname) > 80:
            max_score = max(max_score, float(score))
    return max_score


# ---------- Compatibility wrapper (keeps your original function name) ----------

def fetch_coin_events(coin_id: str) -> List[Mapping[str, object]]:
    """
    Wrapper around api_clients.fetch_coin_events, kept for backward compatibility.
    Returns events already filtered to the past 7 days (UTC).
    """
    try:
        logger.debug(f"Fetching Event data for {coin_id}.")
        events = fetch_coin_events_api(coin_id)
        if not events:
            logger.debug(f"No events found for {coin_id}.")
            return []
        logger.debug(f"Events found for {coin_id}: {len(events)} recent events")
        return events
    except Exception as e:
        logger.debug(f"Error fetching events for {coin_id}: {e}")
        return []


def has_consistent_monthly_growth(historical_df: pd.DataFrame) -> bool:
    """
    True if at least 18 of the last 30 days have positive price change.
    """
    df = historical_df.copy()
    df['price_change'] = df['price'].pct_change()
    last_month_df = df.tail(30)
    rising_days = (last_month_df['price_change'] > 0).sum()
    return rising_days >= 18


def compute_santiment_surge_metrics(santiment_data: Mapping[str, float]) -> Tuple[int, str]:
    """
    Computes surge-related scores from selected Santiment metrics.

    Returns:
        (score, explanation)
    """
    thresholds = {
        'exchange_flow_delta': 1_000_000,  # USD
        'active_addresses_increase': 10,   # %
        'dev_activity_increase': 10,       # %
        'whale_tx_count': 5,               # count
        'volume_change': 10,               # %
        'sentiment_score': 0.2,            # absolute compound
    }

    exchange_inflow = float(santiment_data.get("exchange_inflow_usd", 0) or 0)
    exchange_outflow = float(santiment_data.get("exchange_outflow_usd", 0) or 0)
    dev_activity = float(santiment_data.get("dev_activity_increase", 0) or 0)
    active_addresses_change = float(santiment_data.get("daily_active_addresses_increase", 0) or 0)
    whale_tx_count = float(santiment_data.get("whale_transaction_count_100k_usd_to_inf", 0) or 0)
    # FIX: use the correct key for 1d volume change
    volume_change = float(santiment_data.get("transaction_volume_usd_change_1d", 0) or 0)
    sentiment_weighted = float(santiment_data.get("sentiment_weighted_total", 0) or 0)

    exchange_flow_delta = exchange_outflow - exchange_inflow  # Net outflow = bullish

    score = 0
    explanation: List[str] = []

    if exchange_flow_delta > thresholds['exchange_flow_delta']:
        score += 1
        explanation.append(f"🔁 Net exchange outflow of ${exchange_flow_delta:,.0f} signals accumulation")

    if active_addresses_change > thresholds['active_addresses_increase']:
        score += 1
        explanation.append(f"📈 Active addresses increased {active_addresses_change:.2f}%")

    if dev_activity > thresholds['dev_activity_increase']:
        score += 1
        explanation.append(f"🛠️ Dev activity increase = {dev_activity:.2f}%")

    if whale_tx_count > thresholds['whale_tx_count']:
        score += 1
        explanation.append(f"🐋 Whale tx count = {whale_tx_count:.0f}")

    if volume_change > thresholds['volume_change']:
        score += 1
        explanation.append(f"💸 Tx volume surged {volume_change:.2f}%")

    if sentiment_weighted > thresholds['sentiment_score']:
        score += 1
        explanation.append(f"🎯 Weighted sentiment = {sentiment_weighted:.2f} (positive)")

    return score, (" | ".join(explanation) if explanation else "No Santiment surge signals detected.")


def analyze_coin(
    coin_id: str,
    coin_name: str,
    end_date: str,
    news_df: pd.DataFrame,
    digest_tickers: List[str],
    trending_coins_scores: Mapping[str, float],
    santiment_slugs_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Analyzes a given cryptocurrency and returns a dictionary with various analysis scores.
    """
    short_term_window = 7
    medium_term_window = 30
    long_term_window = 90

    # Use UTC-aware 'now' for windows; end_date remains the provided ISO 'YYYY-MM-DD'
    start_date_short_term = (utcnow() - timedelta(days=short_term_window)).date().isoformat()
    start_date_medium_term = (utcnow() - timedelta(days=medium_term_window)).date().isoformat()
    start_date_long_term = (utcnow() - timedelta(days=long_term_window)).date().isoformat()

    historical_df_short_term = fetch_historical_ticker_data(coin_id, start_date_short_term, end_date)
    historical_df_medium_term = fetch_historical_ticker_data(coin_id, start_date_medium_term, end_date)
    historical_df_long_term = fetch_historical_ticker_data(coin_id, start_date_long_term, end_date)

    # Match the coin with Santiment slugs
    santiment_slug = match_coins_with_santiment(coin_name, santiment_slugs_df)

    # Fetch Santiment data if slug available
    if santiment_slug:
        santiment_data = fetch_santiment_data_for_coin(santiment_slug)
    else:
        santiment_data = {
            "dev_activity_increase": 0.0,
            "daily_active_addresses_increase": 0.0,
            "exchange_inflow_usd": 0.0,
            "exchange_outflow_usd": 0.0,
            "whale_transaction_count_100k_usd_to_inf": 0.0,
            "transaction_volume_usd_change_1d": 0.0,
            "sentiment_weighted_total": 0.0,
        }

    santiment_score, santiment_explanation = compute_santiment_score_with_thresholds(santiment_data)
    santiment_surge_score, santiment_surge_explanation = compute_santiment_surge_metrics(santiment_data)

    if historical_df_long_term.empty or 'price' not in historical_df_long_term.columns:
        logger.debug(f"No valid price data available for {coin_id}.")
        return {"coin_id": coin_id, "coin_name": coin_name, "explanation": f"No valid price data available for {coin_id}."}

    twitter_df = fetch_twitter_data(coin_id)
    tweet_score = 1 if not twitter_df.empty else 0

    volatility = historical_df_long_term['price'].pct_change().std()

    most_recent_market_cap = int(historical_df_long_term['market_cap'].iloc[-1])
    most_recent_volume_24h = int(historical_df_long_term['volume_24h'].iloc[-1])

    price_change_score, price_change_explanation = analyze_price_change(historical_df_long_term['price'], most_recent_market_cap, volatility)
    volume_score, volume_explanation = analyze_volume_change(historical_df_long_term['volume_24h'], most_recent_market_cap, volatility)

    consistent_growth = has_consistent_weekly_growth(historical_df_short_term)
    consistent_growth_score = 1 if consistent_growth else 0

    sustained_volume_growth = has_sustained_volume_growth(historical_df_short_term)
    sustained_volume_growth_score = 1 if sustained_volume_growth else 0

    fear_and_greed_index = fetch_fear_and_greed_index()
    fear_and_greed_score = 1 if fear_and_greed_index is not None and fear_and_greed_index > FEAR_GREED_THRESHOLD else 0

    events = fetch_coin_events(coin_id)  # already 7d filtered
    recent_events_count = len(events)
    event_score = 1 if recent_events_count > 0 else 0

    consistent_monthly_growth = has_consistent_monthly_growth(historical_df_medium_term)
    consistent_monthly_growth_score = 1 if consistent_monthly_growth else 0

    market_cap_class = classify_market_cap(most_recent_market_cap)
    liquidity_risk = classify_liquidity_risk(most_recent_volume_24h, market_cap_class)

    # Integrate sentiment analysis & surge words
    if not news_df.empty:
        coin_news = news_df[news_df['coin'] == coin_name].copy()
        sentiment_score = compute_sentiment_for_coin(coin_name, coin_news.to_dict('records'))
        surge_score, surge_explanation_list = score_surge_words(coin_news, surge_words)
        surge_explanation = "; ".join(surge_explanation_list) if surge_explanation_list else "—"
    else:
        coin_news = pd.DataFrame()
        sentiment_score = 0
        surge_score = 0
        surge_explanation = "No significant surge-related news detected."

    # Digest presence (fuzzy)
    digest_score = 1 if any(
        fuzz.partial_ratio(str(t).lower(), coin_id.lower()) > 80
        or fuzz.partial_ratio(str(t).lower(), coin_name.lower()) > 80
        for t in (digest_tickers or [])
    ) else 0

    # Trending
    trending_score = get_fuzzy_trending_score(coin_id, coin_name, trending_coins_scores)

    # Recompute santiment score note: already computed above; just reuse
    trend_conflict_score = 1 if (consistent_monthly_growth_score and not consistent_growth_score) else 0

    cumulative_score = (
        volume_score + tweet_score + consistent_growth_score + sustained_volume_growth_score +
        fear_and_greed_score + event_score + price_change_score + sentiment_score + surge_score +
        digest_score + trending_score + santiment_score + consistent_monthly_growth_score +
        trend_conflict_score + santiment_surge_score
    )

    max_possible_score = MAX_POSSIBLE_SCORE
    cumulative_score_percentage = (cumulative_score / max_possible_score) * 100 if max_possible_score else 0.0

    # Build explanation
    explanation = (
        f"{coin_name} ({coin_id}) analysis: "
        f"Liquidity Risk: {liquidity_risk}, "
        f"Price Change Score: {'Significant' if price_change_score else 'No significant change'} ({price_change_explanation}), "
        f"Volume Change Score: {'Significant' if volume_score else 'No significant change'} ({volume_explanation}), "
        f"Tweets: {'Yes' if tweet_score else 'None'}, "
        f"Consistent Price Growth: {'Yes' if consistent_growth_score else 'No'}, "
        f"Sustained Volume Growth: {'Yes' if sustained_volume_growth_score else 'No'}, "
        f"Fear and Greed Index: {fear_and_greed_index if isinstance(fear_and_greed_index, int) else 'N/A'}, "
        f"Recent Events: {recent_events_count}, "
        f"Sentiment Score: {sentiment_score}, "
        f"Surge Keywords Score: {surge_score} ({surge_explanation}), "
        f"Santiment Score: {santiment_score} ({santiment_explanation}), "
        f"News Digest Score: {digest_score}, "
        f"Trending Score: {trending_score}, "
        f"Market Cap: {most_recent_market_cap}, "
        f"Volume (24h): {most_recent_volume_24h}, "
        f"Cumulative Surge Score: {cumulative_score} ({cumulative_score_percentage:.2f}%), "
        f"Consistent Monthly Growth: {'Yes' if consistent_monthly_growth_score else 'No'}, "
        f"Trend Conflict: {'Yes' if trend_conflict_score else 'No'} (Monthly growth without short-term support), "
        f"| Santiment Surge Score: {santiment_surge_score} ({santiment_surge_explanation})"
    )

    # Add top 3 news headlines
    if not coin_news.empty:
        news_headlines = coin_news['title'].dropna().astype(str).tolist()[:3]
        if news_headlines:
            explanation += f"; Top News: " + "; ".join(news_headlines)
        else:
            explanation += "; Top News: None"
    else:
        explanation += "; Top News: No recent news found."

    return {
        "coin_id": coin_id,
        "coin_name": coin_name,
        "market_cap": most_recent_market_cap,
        "volume_24h": most_recent_volume_24h,
        "price_change_score": int(price_change_score),
        "volume_change_score": int(volume_score),
        "tweets": len(twitter_df) if tweet_score else 0,
        "consistent_growth": "Yes" if consistent_growth_score else "No",
        "sustained_volume_growth": "Yes" if sustained_volume_growth_score else "No",
        "fear_and_greed_index": int(fear_and_greed_index) if fear_and_greed_index is not None else None,
        "events": recent_events_count,
        "sentiment_score": sentiment_score,
        "surging_keywords_score": surge_score,
        "news_digest_score": digest_score,
        "trending_score": float(trending_score),
        "liquidity_risk": liquidity_risk,
        "santiment_score": santiment_score,
        "santiment_surge_score": santiment_surge_score,
        "santiment_surge_explanation": santiment_surge_explanation,
        "cumulative_score": cumulative_score,
        "cumulative_score_percentage": round(cumulative_score_percentage, 2),
        "explanation": explanation,
        "coin_news": coin_news.to_dict('records') if not coin_news.empty else [],
        "trend_conflict": "Yes" if trend_conflict_score else "No",
    }

# ----------------------------
# MATCHING & THRESHOLDS
# ----------------------------

def match_coins_with_santiment(coin_name: str, santiment_slugs_df: pd.DataFrame, threshold: int = 90) -> Optional[str]:
    """
    Matches a given coin name with the Santiment slugs dataframe using exact then fuzzy match.
    """
    if 'name_normalized' not in santiment_slugs_df.columns:
        logger.warning("'name_normalized' column not found in the Santiment slugs dataframe.")
        return None

    coin_name_normalized = re.sub(r'\W+', '', str(coin_name).lower())

    # Exact match
    match = santiment_slugs_df[santiment_slugs_df['name_normalized'] == coin_name_normalized]
    if not match.empty:
        return str(match['slug'].values[0])

    # Fuzzy match fallback
    choices = santiment_slugs_df['name_normalized'].dropna().astype(str).tolist()
    if not choices:
        logger.info(f"No choices to fuzzy match for {coin_name}")
        return None

    best_match, score = process.extractOne(coin_name_normalized, choices)
    if score >= threshold:
        matched_row = santiment_slugs_df[santiment_slugs_df['name_normalized'] == best_match]
        if not matched_row.empty:
            return str(matched_row['slug'].values[0])
        else:
            logger.warning(f"Fuzzy matched name '{best_match}' not found in DataFrame.")

    logger.info(f"No suitable match found for {coin_name} (normalized: {coin_name_normalized})")
    return None


def get_price_change_thresholds(market_cap_class: str, volatility_class: str) -> Tuple[float, float, float]:
    """
    Returns (short, medium, long) price change thresholds for market cap & volatility classes.
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


def has_sustained_volume_growth(historical_df: pd.DataFrame) -> bool:
    """
    Returns True if >= 4 of last 7 days have positive volume change.
    """
    df = historical_df.copy()
    df['volume_change'] = df['volume_24h'].pct_change()
    last_week_df = df.tail(7)
    rising_volume_days = (last_week_df['volume_change'] > 0).sum()
    return rising_volume_days >= 4


def classify_liquidity_risk(volume_24h: float, market_cap_class: str) -> str:
    """
    Classify liquidity risk based on trading volume and market cap class.
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
    else:  # Small
        if volume_24h < LOW_VOLUME_THRESHOLD_SMALL:
            return "High"
        elif volume_24h < LOW_VOLUME_THRESHOLD_SMALL * 2:
            return "Medium"
        else:
            return "Low"


def has_consistent_weekly_growth(historical_df: pd.DataFrame) -> bool:
    """
    Returns True if >= 4 of last 7 days have positive price change.
    """
    df = historical_df.copy()
    df['price_change'] = df['price'].pct_change()
    last_week_df = df.tail(7)
    rising_days = (last_week_df['price_change'] > 0).sum()
    return rising_days >= 4


def get_volume_thresholds(market_cap_class: str, volatility_class: str) -> Tuple[float, float, float, float, float, float]:
    """
    Returns (short_thr, short_max, med_thr, med_max, long_thr, long_max) for volume ratios.
    """
    thresholds = {
        ("Large", "High"): (2, 4, 1.5, 3, 1.2, 2),
        ("Large", "Medium"): (1.5, 3, 1.2, 2, 1, 1.5),
        ("Large", "Low"): (1.2, 2, 1.1, 1.5, 1, 1.2),
        ("Mid", "High"): (3, 6, 2, 4, 1.5, 2.5),
        ("Mid", "Medium"): (2, 4, 1.5, 3, 1.2, 2),
        ("Mid", "Low"): (1.5, 3, 1.2, 2, 1, 1.5),
        ("Small", "High"): (5, 10, 3, 6, 2, 4),
        ("Small", "Medium"): (3, 6, 2, 4, 1.5, 2.5),
        ("Small", "Low"): (2, 4, 1.5, 3, 1.2, 2)
    }
    return thresholds.get((market_cap_class, volatility_class), (2, 4, 1.5, 3, 1.2, 2))


def score_surge_words(news_df: pd.DataFrame, surge_words: List[str]) -> Tuple[int, List[str]]:
    """
    Fuzzy-score surge words across news; returns (ceil(avg), explanation list).
    """
    total_surge_score = 0.0
    news_count = 0
    explanation: List[str] = []

    if not news_df.empty:
        for _, news_item in news_df.iterrows():
            description = news_item.get('description', '')

            if isinstance(description, str) and description.strip():
                surge_score = 0.0
                article_explanation = []

                for word in surge_words:
                    match_score = fuzz.partial_ratio(word.lower(), description.lower())
                    if match_score > 75:
                        surge_score += match_score / 100.0
                        article_explanation.append(f"Matched word '{word}' with score {match_score}%")

                if surge_score > 0:
                    explanation.append(f"Article: '{news_item.get('title', '')}' → {', '.join(article_explanation)}")

                total_surge_score += surge_score
                news_count += 1

    average_surge_score = (total_surge_score / news_count) if news_count else 0.0
    return int(math.ceil(average_surge_score)), explanation


def classify_volatility(volatility: float) -> str:
    """
    Classify a volatility value as "High", "Medium", or "Low".
    """
    if volatility > HIGH_VOLATILITY_THRESHOLD:
        return "High"
    elif volatility > MEDIUM_VOLATILITY_THRESHOLD:
        return "Medium"
    else:
        return "Low"


def classify_market_cap(market_cap: int) -> str:
    """
    Classify a market capitalization as "Large", "Mid", or "Small".
    """
    if market_cap > 10_000_000_000:
        return "Large"
    elif market_cap > 1_000_000_000:
        return "Mid"
    else:
        return "Small"


def compute_sentiment_for_coin(coin_name: str, news_data: List[Mapping[str, str]]) -> int:
    """
    Computes the sentiment score for a given coin based on its news data.

    Returns:
        1 if avg compound > 0.5 else 0.
    """
    sentiments: List[float] = []
    for news_item in news_data:
        description = news_item.get('description', '')
        if isinstance(description, str) and description.strip():
            sentiment_score = float(analyzer.polarity_scores(description)['compound'])
            sentiments.append(sentiment_score)

    average_sentiment = (sum(sentiments) / len(sentiments)) if sentiments else 0.0
    return 1 if average_sentiment > 0.5 else 0


def compute_santiment_score_with_thresholds(santiment_data: Mapping[str, float]) -> Tuple[int, str]:
    """
    Binary scoring using Santiment data with thresholds for each metric.

    Returns:
        (score, explanation)
    """
    thresholds = {
        'dev_activity': 10.0,             # %
        'daily_active_addresses': 5.0,    # %
    }

    dev_activity = float(santiment_data.get('dev_activity_increase', 0) or 0)
    daily_active_addresses = float(santiment_data.get('daily_active_addresses_increase', 0) or 0)

    explanations: List[str] = []

    if dev_activity > thresholds['dev_activity']:
        dev_activity_score = 1
        explanations.append(f"Development activity increase is significant: {dev_activity:.2f}% (>{thresholds['dev_activity']}%)")
    else:
        dev_activity_score = 0
        explanations.append(f"Development activity increase is low: {dev_activity:.2f}% (≤{thresholds['dev_activity']}%)")

    if daily_active_addresses > thresholds['daily_active_addresses']:
        daily_active_addresses_score = 1
        explanations.append(f"Daily active addresses show growth: {daily_active_addresses:.2f}% (>{thresholds['daily_active_addresses']}%)")
    else:
        daily_active_addresses_score = 0
        explanations.append(f"Daily active addresses growth is weak: {daily_active_addresses:.2f}% (≤{thresholds['daily_active_addresses']}%)")

    total_santiment_score = dev_activity_score + daily_active_addresses_score
    explanation = " | ".join(explanations)

    return total_santiment_score, explanation