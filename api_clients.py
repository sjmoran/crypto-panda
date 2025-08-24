#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crypto data utilities:
- CoinPaprika (twitter, events, historical OHLCV)
- Santiment via sanpy (metrics + project slugs)
- CryptoNews (trending mentions, weekly news, sundown digest)
- Alternative.me (Fear & Greed)

Design notes:
- UTC everywhere, ISO-8601 in logs.
- One shared requests.Session with retries (429, 5xx), respects Retry-After.
- Callable-level retry wrapper with jittered backoff for non-requests clients.
- Defensive JSON parsing; stable return types for pipelines.
"""

from __future__ import annotations

import os
import time
import math
import json
import logging
import random
import traceback
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Third-party SDKs
import san  # sanpy
from coinpaprika import client as Coinpaprika

# ----------------------------
# Configuration (env + config)
# ----------------------------

try:
    from config import (
        BACKOFF_FACTOR,
        MAX_RETRIES,
        COIN_PAPRIKA_API_KEY,
        SAN_API_KEY,
    )
except Exception:
    # Sane fallbacks if config object isn’t importable.
    BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "2.0"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
    COIN_PAPRIKA_API_KEY = os.getenv("COIN_PAPRIKA_API_KEY", "")
    SAN_API_KEY = os.getenv("SAN_API_KEY", "")

CRYPTO_NEWS_API_KEY = os.getenv("CRYPTO_NEWS_API_KEY", "")

# Set Santiment API key (sanpy)
if SAN_API_KEY:
    os.environ["SANAPIKEY"] = SAN_API_KEY
    san.ApiConfig.api_key = SAN_API_KEY

# CoinPaprika client
_coinpaprika_client = Coinpaprika.Client(api_key=COIN_PAPRIKA_API_KEY) if COIN_PAPRIKA_API_KEY else Coinpaprika.Client()

# ----------------------------
# Logging (console + single rolling file per script)
# ----------------------------

def setup_logging(name: str,
                  log_dir: Union[str, Path] = None,
                  level: str = None) -> logging.Logger:
    """
    Create a logger that writes to console and a single per-script logfile.
    File path: <log_dir>/<script_stem>.log (overwrites each run)
    """
    base_dir = Path(__file__).resolve().parent
    log_dir = Path(log_dir or (base_dir / "../logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{Path(__file__).stem}.log"

    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level_val = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level_val)
    logger.propagate = False

    # Prevent duplicate handlers on reload
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = "%(asctime)sZ [%(levelname)s] %(name)s | %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    ch = logging.StreamHandler()
    ch.setLevel(level_val)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8", delay=False)
    fh.setLevel(level_val)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Logging started → {log_path} (level={level_name})")
    return logger

logger = setup_logging(__name__)

# ----------------------------
# Helpers: time & session
# ----------------------------

def utcnow() -> datetime:
    """Current time in UTC (aware)."""
    return datetime.now(timezone.utc)

def to_date(dt: datetime) -> datetime:
    """Truncate aware datetime to its date (still aware, 00:00)."""
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()

def _make_session() -> requests.Session:
    """
    Shared session with a robust retry policy:
    - Retries 429 & 5xx
    - Respects Retry-After
    - Exponential backoff
    """
    session = requests.Session()

    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,  # urllib3 backoff = backoff_factor * (2 ** (retry_num - 1))
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=64)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # sane default timeout wrapper via a custom hook
    def _timeout_request(method, url, **kwargs):
        kwargs.setdefault("timeout", float(os.getenv("HTTP_TIMEOUT", "15")))
        return orig_request(method, url, **kwargs)

    orig_request = session.request
    session.request = _timeout_request  # type: ignore

    return session

SESSION = _make_session()

# ----------------------------
# Generic callable retry (for SDKs that are not requests-based)
# ----------------------------

def call_with_retries(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = MAX_RETRIES,
    backoff_factor: float = BACKOFF_FACTOR,
    jitter: Tuple[float, float] = (0.1, 0.5),
    **kwargs: Any,
) -> Any:
    """
    Retry a callable with jittered exponential backoff.
    Use for SDK calls like coinpaprika client or sanpy when they raise Exceptions.
    """
    attempts = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            attempts += 1
            if attempts > max_retries:
                logger.error(f"Max retries exceeded for {func.__name__}: {e}")
                raise
            sleep_s = (backoff_factor ** (attempts - 1)) + random.uniform(*jitter)
            logger.debug(f"{func.__name__} failed (attempt {attempts}/{max_retries}): {e}. Sleeping {sleep_s:.2f}s.")
            time.sleep(sleep_s)

# ----------------------------
# Data fetching functions
# ----------------------------

def fetch_twitter_data(coin_id: str) -> pd.DataFrame:
    """
    CoinPaprika twitter posts for the past 7 days (UTC).
    Returns empty DataFrame if nothing or columns missing.
    """
    try:
        tweets = call_with_retries(_coinpaprika_client.twitter, coin_id)
    except Exception as e:
        logger.debug(f"twitter API error for {coin_id}: {e}")
        return pd.DataFrame()

    if not tweets:
        return pd.DataFrame()

    df = pd.DataFrame(tweets)
    if not {"status", "date"} <= set(df.columns):
        logger.debug(f"Twitter data missing expected columns for {coin_id}: {df.columns.tolist()}")
        return pd.DataFrame()

    one_week_ago = utcnow() - timedelta(days=7)
    # CoinPaprika 'date' is ISO; coerce to UTC-aware
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[df["date"] >= one_week_ago]

    return df.reset_index(drop=True)


def fetch_santiment_metric(metric: str, coin_slug: str, start_date: str, end_date: str) -> Optional[float]:
    """
    Fetch a Santiment metric via sanpy.
    Returns latest 'value' or None if unavailable.
    """
    try:
        logger.debug(f"san.get metric={metric} slug={coin_slug} from={start_date} to={end_date}")
        result = san.get(metric, slug=coin_slug, from_date=start_date, to_date=end_date)
        if isinstance(result, pd.DataFrame) and not result.empty and "value" in result.columns:
            return float(result.iloc[-1]["value"])
        logger.debug(f"No data for {metric}:{coin_slug}")
        return None
    except Exception as e:
        logger.debug(f"Santiment metric error {metric}:{coin_slug}: {e}")
        return None


def fetch_santiment_data_for_coin(coin_slug: str) -> Dict[str, float]:
    """
    Returns a dict of Santiment metrics; defaults to 0.0 on missing/invalid.
    """
    now = utcnow()
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")

    def _safe(metric_name: str) -> float:
        val = fetch_santiment_metric(metric_name, coin_slug, start_date, end_date)
        try:
            return float(val) if val is not None and math.isfinite(float(val)) else 0.0
        except Exception:
            return 0.0

    try:
        payload = {
            "dev_activity_increase": _safe("30d_moving_avg_dev_activity_change_1d"),
            "daily_active_addresses_increase": _safe("active_addresses_24h_change_30d"),
            "exchange_inflow_usd": _safe("exchange_inflow_usd"),
            "exchange_outflow_usd": _safe("exchange_outflow_usd"),
            "whale_transaction_count_100k_usd_to_inf": _safe("whale_transaction_count_100k_usd_to_inf"),
            "transaction_volume_usd_change_1d": _safe("transaction_volume_usd_change_1d"),
            "sentiment_weighted_total": _safe("sentiment_weighted_total_1d"),
        }
        return payload
    except Exception as e:
        logger.error(f"Santiment bundle fetch failed for {coin_slug}: {e}")
        return {
            "dev_activity_increase": 0.0,
            "daily_active_addresses_increase": 0.0,
            "exchange_inflow_usd": 0.0,
            "exchange_outflow_usd": 0.0,
            "whale_transaction_count_100k_usd_to_inf": 0.0,
            "transaction_volume_usd_change_1d": 0.0,
            "sentiment_weighted_total": 0.0,
        }


def get_sundown_digest() -> List[Mapping[str, Any]]:
    """
    CryptoNews Sundown Digest page 1.
    """
    if not CRYPTO_NEWS_API_KEY:
        logger.debug("CRYPTO_NEWS_API_KEY not set; skipping sundown digest.")
        return []

    url = f"https://cryptonews-api.com/api/v1/sundown-digest?page=1&token={CRYPTO_NEWS_API_KEY}"
    try:
        resp = SESSION.get(url)
        if resp.status_code != 200:
            logger.debug(f"Sundown digest non-200: {resp.status_code} body={resp.text[:400]}")
            return []
        data = resp.json()
        return data.get("data", []) if isinstance(data, dict) else []
    except Exception as e:
        logger.debug(f"Sundown digest error: {e}")
        return []


def filter_active_and_ranked_coins(coins: Iterable[Mapping[str, Any]], max_coins: int, rank_threshold: int = 1000) -> List[Mapping[str, Any]]:
    """
    Keep active, non-new coins with rank <= threshold, then truncate to max_coins.
    """
    filtered = [
        c for c in coins
        if c.get("is_active") and not c.get("is_new") and (c.get("rank") is not None) and c.get("rank") <= rank_threshold
    ]
    return filtered[:max_coins]


@lru_cache(maxsize=1)
def fetch_santiment_slugs() -> pd.DataFrame:
    """
    Santiment 'projects/all' slugs. Cached (process-level) until interpreter exits.
    """
    try:
        projects = san.get(
            "projects/all",
            interval="1d",
            columns=["slug", "name", "ticker", "infrastructure", "mainContractAddress"],
        )
        df = pd.DataFrame(projects) if not isinstance(projects, pd.DataFrame) else projects.copy()
        if df.empty:
            return pd.DataFrame()
        df["name_normalized"] = df["name"].map(lambda x: re.sub(r"\W+", "", str(x).lower()))
        logger.info(f"Fetched {len(df)} Santiment slugs")
        return df
    except Exception as e:
        logger.error(f"Error fetching Santiment slugs: {e}")
        return pd.DataFrame()


def fetch_news_for_past_week(tickers: Mapping[str, str]) -> pd.DataFrame:
    """
    CryptoNews: for each {coin_name: ticker}, pull ranked news within last 7-day window.
    Returns DataFrame with columns: coin, date, title, description, url, source
    """
    if not CRYPTO_NEWS_API_KEY:
        logger.debug("CRYPTO_NEWS_API_KEY not set; skipping news.")
        return pd.DataFrame(columns=["coin", "date", "title", "description", "url", "source"])

    end_date = to_date(utcnow())  # 00:00Z today
    all_rows: List[Dict[str, Any]] = []

    # Iterate days backward for 7 days
    for _ in range(7):
        week_start = (end_date - timedelta(days=7)).strftime("%m%d%Y")
        week_end = end_date.strftime("%m%d%Y")
        date_str = f"{week_start}-{week_end}"

        for coin_name, ticker in tickers.items():
            url = f"https://cryptonews-api.com/api/v1?tickers={ticker}&items=1&date={date_str}&sortby=rank&token={CRYPTO_NEWS_API_KEY}"
            try:
                resp = SESSION.get(url)
                if resp.status_code != 200:
                    logger.debug(f"News non-200 for {coin_name} ({ticker}) {resp.status_code}: {resp.text[:300]}")
                    continue
                payload = resp.json()
                data_list = payload.get("data", []) if isinstance(payload, dict) else []
                for article in data_list:
                    all_rows.append(
                        {
                            "coin": coin_name,
                            "date": end_date.date().isoformat(),
                            "title": article.get("title", ""),
                            "description": article.get("text", "") or "",
                            "url": article.get("news_url", ""),
                            "source": article.get("source_name", ""),
                        }
                    )
            except Exception as e:
                logger.debug(f"News fetch error for {coin_name} ({ticker}): {e}")

            time.sleep(0.25)  # be nice

        end_date -= timedelta(days=1)

    return pd.DataFrame(all_rows)


def fetch_trending_coins_scores() -> Dict[str, float]:
    """
    CryptoNews top-mention over last 7 days -> normalized score in [0, 3].
    Returns {} on errors.
    """
    if not CRYPTO_NEWS_API_KEY:
        logger.debug("CRYPTO_NEWS_API_KEY not set; skipping trending coins.")
        return {}

    url = f"https://cryptonews-api.com/api/v1/top-mention?date=last7days&token={CRYPTO_NEWS_API_KEY}"
    try:
        resp = SESSION.get(url)
        status = resp.status_code
        body = resp.text
        if status != 200:
            logger.debug(f"top-mention non-200: {status} body={body[:400]}")
            return {}
        try:
            payload = resp.json()
        except Exception as je:
            logger.debug(f"JSON parse fail top-mention: {je} body={body[:400]}")
            return {}

        # Normalize schema variants
        data: List[Mapping[str, Any]] = []
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("all"), list):
                data = payload["data"]["all"]
            elif isinstance(payload.get("data"), list):
                data = payload["data"]
            elif isinstance(payload.get("all"), list):
                data = payload["all"]
        elif isinstance(payload, list):
            data = payload

        if not data:
            return {}

        raw: Dict[str, float] = {}
        for item in data:
            ticker = str(item.get("ticker", "")).strip().lower()
            if not ticker:
                continue
            try:
                s = float(item.get("sentiment_score", 0) or 0)
                m = float(item.get("total_mentions", 0) or 0)
            except (TypeError, ValueError):
                s, m = 0.0, 0.0
            raw[ticker] = raw.get(ticker, 0.0) + (s * m)

        if not raw:
            return {}

        lo, hi = min(raw.values()), max(raw.values())
        if hi == lo:
            return {t: 1.5 for t in raw}  # neutral mid if degenerate
        return {t: 3.0 * (v - lo) / (hi - lo) for t, v in raw.items()}

    except Exception as e:
        logger.debug(f"trending coins error: {e}")
        return {}


def fetch_fear_and_greed_index() -> Optional[int]:
    """
    Returns the current Fear & Greed Index value (0..100), or None.
    """
    try:
        resp = SESSION.get("https://api.alternative.me/fng/")
        if resp.status_code != 200:
            logger.debug(f"FNG non-200: {resp.status_code} body={resp.text[:200]}")
            return None
        payload = resp.json()
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        if isinstance(data, list) and data:
            return int(data[0].get("value"))
        return None
    except Exception as e:
        logger.debug(f"FNG fetch error: {e}")
        return None


def fetch_coin_events(coin_id: str) -> List[Mapping[str, Any]]:
    """
    CoinPaprika events within the past 7 days (UTC). Future-dated events excluded.
    """
    try:
        events = call_with_retries(_coinpaprika_client.events, coin_id=coin_id)
    except Exception as e:
        logger.debug(f"events API error for {coin_id}: {e}")
        return []

    if not events:
        return []

    now = utcnow()
    one_week_ago = now - timedelta(days=7)
    out: List[Mapping[str, Any]] = []

    for ev in events:
        try:
            # CoinPaprika event date: "YYYY-MM-DDTHH:MM:SSZ"
            ev_dt = datetime.strptime(ev["date"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            if one_week_ago <= ev_dt <= now:
                out.append(ev)
        except Exception:
            continue

    return out


def fetch_historical_ticker_data(coin_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    CoinPaprika historical candles (1d, USD).
    Returns DataFrame: date, price, coin_id, volume_24h, market_cap (sorted by date).
    """
    try:
        hist = call_with_retries(
            _coinpaprika_client.historical,
            coin_id=coin_id,
            start=start_date,
            end=end_date,
            interval="1d",
            quote="usd",
        )
    except Exception as e:
        logger.debug(f"historical error for {coin_id}: {e}")
        return pd.DataFrame(columns=["date", "price", "coin_id", "volume_24h", "market_cap"])

    if not isinstance(hist, list) or not hist:
        return pd.DataFrame(columns=["date", "price", "coin_id", "volume_24h", "market_cap"])

    df = pd.DataFrame(hist)
    expected = {"timestamp", "price", "volume_24h", "market_cap"}
    if not expected <= set(df.columns):
        logger.debug(f"Historical missing expected cols for {coin_id}: {df.columns.tolist()}")
        return pd.DataFrame(columns=["date", "price", "coin_id", "volume_24h", "market_cap"])

    df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.date
    df["coin_id"] = coin_id
    df = df[["date", "price", "coin_id", "volume_24h", "market_cap"]].sort_values("date").reset_index(drop=True)
    return df