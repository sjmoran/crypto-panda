#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import pandas as pd
import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import traceback
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns  # kept for correlation heatmap
from pathlib import Path
from typing import Union

from api_clients import (
    get_sundown_digest,
    fetch_trending_coins_scores,
    fetch_news_for_past_week,
    fetch_santiment_slugs,
    call_with_retries,
    filter_active_and_ranked_coins
)
from coin_analysis import analyze_coin
from data_management import (
    save_result_to_csv,
    retrieve_historical_data_from_aurora,
    save_cumulative_score_to_aurora,
    create_coin_data_table_if_not_exists,
    load_existing_results,
    load_tickers,
)
from plotting import plot_top_coins_over_time
from report_generation import (
    gpt4o_summarize_each_coin,
    save_report_to_excel,
    summarize_sundown_digest,
    generate_html_report_with_recommendations,
    send_email_with_report,
)
from config import (
    TEST_ONLY,
    CUMULATIVE_SCORE_REPORTING_THRESHOLD,
    NUMBER_OF_TOP_COINS_TO_MONITOR,
    CRYPTO_NEWS_TICKERS,
    LOG_DIR,
    COIN_PAPRIKA_API_KEY,
    FEAR_GREED_THRESHOLD,
)

from coinpaprika import client as Coinpaprika

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

    log_path = log_dir / f"{Path(__file__).stem}.log"

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
logger = setup_logging(__name__, log_dir=LOG_DIR)

# ----------------------------
# Setup
# ----------------------------

# Ensure log directory exists for any artifacts
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# CoinPaprika client (only used here for coins list)
client = Coinpaprika.Client(api_key=COIN_PAPRIKA_API_KEY)

coin_audit_log: list = []


def utc_today_iso() -> str:
    """Current date in UTC, YYYY-MM-DD."""
    return datetime.now(timezone.utc).date().isoformat()


def process_single_coin(
    coin: dict,
    existing_results: pd.DataFrame,
    tickers_dict: dict,
    digest_tickers: list,
    trending_coins_scores: dict,
    santiment_slugs_df: pd.DataFrame,
    end_date: str,
    score_usage: defaultdict,
    coin_audit_log: list,
):
    """
    Processes a single coin and records scoring usage + audit details.
    """
    try:
        coin_id = coin["id"]
        coin_name = str(coin["name"]).lower()

        if existing_results is not None and not existing_results.empty and coin_id in existing_results.get("coin_id", pd.Series([])).values:
            logger.debug(f"Skipping already processed coin: {coin_id}")
            return None

        logger.debug(f"Processing {coin_name} ({coin_id})")

        coins_dict = {coin_name: str(tickers_dict.get(coin_name, "")).upper()}
        news_df = fetch_news_for_past_week(coins_dict)

        result = analyze_coin(
            coin_id,
            coin_name,
            end_date,
            news_df,
            digest_tickers,
            trending_coins_scores,  # may be {}
            santiment_slugs_df,
        )

        # Score usage tracking (defensive conversions)
        score_usage["price_change_score"].append(int(result.get("price_change_score", 0) or 0))
        score_usage["volume_change_score"].append(int(result.get("volume_change_score", 0) or 0))
        score_usage["tweet_score"].append(1 if result.get("tweets") not in (None, "None", 0) else 0)
        score_usage["sentiment_score"].append(int(result.get("sentiment_score", 0) or 0))
        score_usage["surging_keywords_score"].append(int(result.get("surging_keywords_score", 0) or 0))
        score_usage["consistent_growth"].append(1 if result.get("consistent_growth", "No") == "Yes" else 0)
        score_usage["sustained_volume_growth"].append(1 if result.get("sustained_volume_growth", "No") == "Yes" else 0)

        try:
            fear_greed_value = int(result.get("fear_and_greed_index", 0) or 0)
            score_usage["fear_and_greed_index"].append(1 if fear_greed_value > FEAR_GREED_THRESHOLD else 0)
        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"Failed to process fear_and_greed_index: {e}")
            score_usage["fear_and_greed_index"].append(0)

        score_usage["event_score"].append(1 if (result.get("events", 0) or 0) > 0 else 0)
        score_usage["digest_score"].append(int(result.get("news_digest_score", 0) or 0))
        score_usage["trending_score"].append(float(result.get("trending_score", 0) or 0))
        score_usage["santiment_score"].append(int(result.get("santiment_score", 0) or 0))
        score_usage["santiment_surge_score"].append(int(result.get("santiment_surge_score", 0) or 0))
        score_usage["consistent_monthly_growth"].append(1 if result.get("consistent_monthly_growth", "No") == "Yes" else 0)
        score_usage["trend_conflict"].append(1 if result.get("trend_conflict", "No") == "Yes" else 0)
        score_usage["cumulative_score"].append(int(result.get("cumulative_score", 0) or 0))
        score_usage["cumulative_score_percentage"].append(float(result.get("cumulative_score_percentage", 0) or 0))

        # Persist
        save_result_to_csv(result)
        save_cumulative_score_to_aurora(
            result["coin_id"], result["coin_name"], result["cumulative_score_percentage"]
        )

        # Audit
        audit_entry = {
            "coin_id": coin_id,
            "coin_name": coin_name,
            "ticker": coins_dict.get(coin_name),
            "date": datetime.now(timezone.utc).isoformat(),
            "included_in_report": None,  # Set later
            "reason_for_exclusion": None,
            "scores": {
                "price_change_score": result.get("price_change_score"),
                "volume_change_score": result.get("volume_change_score"),
                "sentiment_score": result.get("sentiment_score"),
                "surging_keywords_score": result.get("surging_keywords_score"),
                "fear_and_greed_index": result.get("fear_and_greed_index"),
                "event_score": result.get("events"),
                "digest_score": result.get("news_digest_score"),
                "trending_score": result.get("trending_score"),
                "santiment_score": result.get("santiment_score"),
                "santiment_surge_score": result.get("santiment_surge_score"),
                "consistent_monthly_growth": result.get("consistent_monthly_growth"),
                "trend_conflict": result.get("trend_conflict"),
                "cumulative_score": result.get("cumulative_score"),
                "cumulative_score_percentage": result.get("cumulative_score_percentage"),
                "liquidity_risk": result.get("liquidity_risk", "Unknown"),
            },
        }

        coin_audit_log.append(audit_entry)
        return result

    except Exception as e:
        logger.debug(f"Error processing {coin.get('name')} ({coin.get('id')}): {e}")
        logger.debug(traceback.format_exc())
        return None


def summarize_scores(score_usage: defaultdict, output_dir: str = "../logs/"):
    """
    Generates a summary + histograms + correlation heatmap for score_usage.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "score_summary.txt")

    with open(summary_file, "w") as f:
        f.write("--- SCORING SUMMARY ---\n\n")
        for score_type, scores in score_usage.items():
            s = pd.Series(scores, dtype="float")
            summary = (
                f"{score_type}:\n"
                f"  Count: {len(s)}\n"
                f"  Mean: {s.mean():.2f}\n"
                f"  Std Dev: {s.std():.2f}\n"
                f"  Min: {s.min()}, Max: {s.max()}\n"
                f"  Non-zero count: {(s > 0).sum()} ({(s > 0).mean()*100:.2f}%)\n\n"
            )
            print(summary)
            f.write(summary)

            # Histogram
            plt.figure()
            s.hist(bins=10)
            plt.title(score_type)
            plt.xlabel("Score")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{score_type}_histogram.png"))
            plt.close()

    # Correlation heatmap (if enough data)
    df_scores = pd.DataFrame(score_usage)
    if df_scores.shape[0] >= 2 and df_scores.shape[1] >= 2:
        corr = df_scores.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation between scoring components")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "score_correlation_heatmap.png"))
        plt.close()
    else:
        logger.debug("Not enough data to compute correlation heatmap.")


def monitor_coins_and_send_report():
    """
    Main entry point for monitoring coins and generating a weekly report.
    """
    create_coin_data_table_if_not_exists()

    if TEST_ONLY:
        existing_results = pd.DataFrame([])
        coins_to_monitor = [
            {"id": "btc-bitcoin", "name": "Bitcoin"},
            {"id": "eth-ethereum", "name": "Ethereum"},
        ]
    else:
        existing_results = load_existing_results()
        try:
            # Use call_with_retries instead of the old api_call_with_retries
            coins_to_monitor = call_with_retries(client.coins)
        except Exception as e:
            logger.error(f"Failed to fetch coins list from CoinPaprika: {e}")
            coins_to_monitor = []

        logger.debug(f"Number of coins retrieved: {len(coins_to_monitor)}")
        coins_to_monitor = filter_active_and_ranked_coins(
            coins_to_monitor, NUMBER_OF_TOP_COINS_TO_MONITOR
        )

    logger.debug(f"Number of active and ranked coins selected: {len(coins_to_monitor)}")
    end_date = utc_today_iso()

    tickers_dict = load_tickers(CRYPTO_NEWS_TICKERS)

    # Sundown digest
    sundown_digest = get_sundown_digest()
    digest_summary = summarize_sundown_digest(sundown_digest)
    digest_tickers = digest_summary.get("tickers", [])
    logger.debug(f"Sundown digest tickers extracted: {len(digest_tickers)}")

    # Trending coins (defensive; may be {})
    trending_coins_scores = fetch_trending_coins_scores()
    if not trending_coins_scores:
        logger.debug("Trending coins scores unavailable or empty; continuing without trending influence.")

    # Santiment slugs
    santiment_slugs_df = fetch_santiment_slugs()

    score_usage: defaultdict = defaultdict(list)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = executor.map(
            lambda c: process_single_coin(
                c,
                existing_results,
                tickers_dict,
                digest_tickers,
                trending_coins_scores,
                santiment_slugs_df,
                end_date,
                score_usage,
                coin_audit_log,
            ),
            coins_to_monitor,
        )
        results_list = [
            r for r in tqdm(futures, total=len(coins_to_monitor), desc="Processing Coins") if r is not None
        ]

    if not results_list:
        logger.debug("No coin results produced; exiting after score summary.")
        summarize_scores(score_usage, output_dir=LOG_DIR)
        return

    df = pd.DataFrame(results_list)
    raw_df = df.copy()

    try:
        if not df.empty:
            # Filter for report
            df = df[
                (df["liquidity_risk"].isin(["Low", "Medium"]))
                & (df["cumulative_score_percentage"] > CUMULATIVE_SCORE_REPORTING_THRESHOLD)
            ]

            logger.debug("DataFrame is not empty, processing report entries.")
            coins_in_df = df["coin_name"].unique()

            # Plot top coins over time if we have history
            if len(coins_in_df) > 0:
                historical_data = retrieve_historical_data_from_aurora()
                if not historical_data.empty:
                    plot_top_coins_over_time(
                        historical_data[historical_data["coin_name"].isin(coins_in_df)],
                        top_n=10,
                    )

            report_entries = df.to_dict("records")
            report_entries = sorted(report_entries, key=lambda x: x.get("cumulative_score", 0), reverse=True)
            logger.debug(f"Report entries after sorting: {len(report_entries)} entries")

            # Ensure numeric fields are correctly typed
            numeric_fields = [
                "price_change_score", "volume_change_score", "sentiment_score",
                "surging_keywords_score", "news_digest_score", "trending_score",
                "santiment_score", "santiment_surge_score", "cumulative_score",
                "cumulative_score_percentage", "fear_and_greed_index", "market_cap",
                "volume_24h", "events",
            ]
            for field in numeric_fields:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors="coerce")

            logger.debug("DataFrame contents before GPT-4o recommendations:\n%s", df.to_string())

            gpt_recommendations = gpt4o_summarize_each_coin(df)
            logger.debug("GPT-4o recommendations generated.")

            html_report = generate_html_report_with_recommendations(report_entries, digest_summary, gpt_recommendations)
            logger.debug("HTML report generated successfully.")

            attachment_path = save_report_to_excel(report_entries)
            logger.debug(f"Report saved to Excel at: {attachment_path}")

            send_email_with_report(
                html_report,
                attachment_path,
                recommendations=gpt_recommendations.get("recommendations"),
            )
            logger.debug("Email sent successfully.")

            current_date = utc_today_iso()
            results_file = os.path.join(LOG_DIR, f"results_{current_date}.csv")

            if os.path.exists(results_file):
                try:
                    os.remove(results_file)
                    logger.debug(f"{results_file} has been deleted successfully.")
                except Exception as e:
                    logger.debug(f"Failed to delete {results_file}: {e}")

            summarize_scores(score_usage, output_dir=LOG_DIR)

            # Build audit flags
            raw_coin_ids = set(raw_df["coin_id"])
            final_coin_ids = set(df["coin_id"])
            already_processed_ids = set(existing_results["coin_id"]) if (existing_results is not None and not existing_results.empty and "coin_id" in existing_results.columns) else set()

            for entry in coin_audit_log:
                cid = entry["coin_id"]
                if cid in final_coin_ids:
                    entry["included_in_report"] = True
                    entry["reason_for_exclusion"] = None
                elif cid in raw_coin_ids:
                    entry["included_in_report"] = False
                    entry["reason_for_exclusion"] = "Filtered due to low score or liquidity"
                elif cid in already_processed_ids:
                    entry["included_in_report"] = False
                    entry["reason_for_exclusion"] = "Skipped (already processed)"
                else:
                    entry["included_in_report"] = False
                    entry["reason_for_exclusion"] = "Processing failed"

            audit_log_file = os.path.join(LOG_DIR, f"audit_log_{utc_today_iso()}.json")
            with open(audit_log_file, "w") as f:
                json.dump(coin_audit_log, f, indent=2)

            audit_df = pd.json_normalize(coin_audit_log)
            audit_df.to_csv(os.path.join(LOG_DIR, f"audit_log_{utc_today_iso()}.csv"), index=False)

        else:
            logger.debug("No valid entries to report. DataFrame is empty.")
            summarize_scores(score_usage, output_dir=LOG_DIR)

    except Exception as e:
        logger.error(f"An error occurred during the report generation process: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    monitor_coins_and_send_report()
