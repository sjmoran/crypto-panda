import os
import time
import pandas as pd
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import traceback
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from api_clients import (
    get_sundown_digest,
    fetch_trending_coins_scores,
    fetch_news_for_past_week,
    fetch_santiment_slugs,
    client,
    api_call_with_retries
)
from coin_analysis import analyze_coin
from data_management import (
    save_result_to_csv,
    retrieve_historical_data_from_aurora,
    save_cumulative_score_to_aurora,
    create_coin_data_table_if_not_exists,
    load_existing_results,
    load_tickers
)
from plotting import plot_top_coins_over_time
from report_generation import (
    gpt4o_summarize_each_coin,
    save_report_to_excel,
    summarize_sundown_digest,
    generate_html_report_with_recommendations,
    send_email_with_report
)
from config import (
    TEST_ONLY,
    CUMULATIVE_SCORE_REPORTING_THRESHOLD,
    NUMBER_OF_TOP_COINS_TO_MONITOR,
    CRYPTO_NEWS_TICKERS,
    LOG_DIR,
    COIN_PAPRIKA_API_KEY,FEAR_GREED_THRESHOLD
)
from helpers import (filter_active_and_ranked_coins)

from coinpaprika import client as Coinpaprika

client = Coinpaprika.Client(api_key=COIN_PAPRIKA_API_KEY)

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR + '/crypto-panda.log', mode='w')
    ]
)

logging.debug("Logging is set up and the application has started.")

def process_single_coin(coin, existing_results, tickers_dict, digest_tickers, trending_coins_scores, santiment_slugs_df, end_date, score_usage):
    """
    Processes a single coin, performing the following steps:

    1. Skips coins that have already been processed in the past week.
    2. Fetches news articles for the coin for the past week.
    3. Analyzes the coin using the analyze_coin function.
    4. Saves the result to the CSV file.
    5. Saves the cumulative score to Aurora.
    6. Returns the result.

    Parameters:
        coin (dict): The coin to process, with keys 'id' and 'name'.
        existing_results (pd.DataFrame): The existing results for the past week.
        tickers_dict (dict): A dictionary mapping coin names to their corresponding tickers.
        digest_tickers (list): A list of tickers extracted from the Sundown Digest.
        trending_coins_scores (dict): A dictionary with tickers as keys and their respective scores.
        santiment_slugs_df (pd.DataFrame): DataFrame containing Santiment slugs for various coins.
        end_date (str): The end date of the period for which to fetch the historical ticker data (in YYYY-MM-DD format).

    Returns:
        dict: The result of the analysis, with keys 'coin_id', 'coin_name', 'cumulative_score_percentage', and 'explanation'.
    """
    try:
        coin_id = coin['id']
        coin_name = coin['name'].lower()

        if existing_results is not None and not existing_results.empty and coin_id in existing_results['coin_id'].values:
            logging.debug(f"Skipping already processed coin: {coin_id}")
            return None

        logging.debug(f"Processing {coin_name} ({coin_id})")

        coins_dict = {coin_name: tickers_dict.get(coin_name, '').upper()}
        news_df = fetch_news_for_past_week(coins_dict)

        result = analyze_coin(
            coin_id,
            coin_name,
            end_date,
            news_df,
            digest_tickers,
            trending_coins_scores,
            santiment_slugs_df
        )
        
        score_usage["price_change_score"].append(int(result["price_change_score"]))
        score_usage["volume_change_score"].append(int(result["volume_change_score"]))
        score_usage["tweet_score"].append(1 if result["tweets"] != "None" else 0)
        score_usage["sentiment_score"].append(result["sentiment_score"])
        score_usage["surging_keywords_score"].append(result["surging_keywords_score"])
        score_usage["consistent_growth"].append(1 if result["consistent_growth"] == "Yes" else 0)
        score_usage["sustained_volume_growth"].append(1 if result["sustained_volume_growth"] == "Yes" else 0)

        try:
            fear_greed_value = int(result["fear_and_greed_index"])
            score_usage["fear_and_greed_index"].append(
                1 if fear_greed_value > FEAR_GREED_THRESHOLD else 0
            )
        except (ValueError, TypeError, KeyError) as e:
            logging.debug(f"Failed to process fear_and_greed_index: {e}")
            score_usage["fear_and_greed_index"].append(0)

        score_usage["event_score"].append(1 if result["events"] > 0 else 0)
        score_usage["digest_score"].append(result["news_digest_score"])
        score_usage["trending_score"].append(result["trending_score"])
        score_usage["santiment_score"].append(result["santiment_score"])
        score_usage["santiment_surge_score"].append(result["santiment_surge_score"])
        score_usage["consistent_monthly_growth"].append(1 if result.get("consistent_monthly_growth", "No") == "Yes" else 0)
        score_usage["trend_conflict"].append(1 if result.get("trend_conflict", "No") == "Yes" else 0)
        score_usage["cumulative_score"].append(result["cumulative_score"])
        score_usage["cumulative_score_percentage"].append(result["cumulative_score_percentage"])

        save_result_to_csv(result)
        save_cumulative_score_to_aurora(result['coin_id'], result['coin_name'], result['cumulative_score_percentage'])

        return result

    except Exception as e:
        logging.debug(f"Error processing {coin['name']} ({coin['id']}): {e}")
        logging.debug(traceback.format_exc())
        return None


def summarize_scores(score_usage, output_dir="../logs/"):
    """
    Generates a summary and histogram plot for each type of score in the given score usage dictionary.

    Parameters:
        score_usage (dict): A dictionary where keys represent different types of scores and values
                            are lists of score values.
        output_dir (str): Directory to save summary and plots.

    Saves:
        - A summary text file: score_summary.txt
        - Histogram plots for each score
        - A correlation heatmap of the scores
    """

    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "score_summary.txt")

    with open(summary_file, "w") as f:
        f.write("--- SCORING SUMMARY ---\n\n")
        for score_type, scores in score_usage.items():
            s = pd.Series(scores)
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

            # Save histogram
            plt.figure()
            s.hist(bins=10)
            plt.title(score_type)
            plt.xlabel("Score")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{score_type}_histogram.png"))
            plt.close()

    # Correlation heatmap
    df_scores = pd.DataFrame(score_usage)
    corr = df_scores.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation between scoring components")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_correlation_heatmap.png"))
    plt.close()

def monitor_coins_and_send_report():
    """
    Main entry point for monitoring coins and generating a weekly report.

    The following steps are taken in this function:

    1. Create the coin data table in Aurora if it does not exist.
    2. Load existing results from the previous week.
    3. Retrieve a list of active and ranked coins from CoinPaprika.
    4. Process each coin in parallel using the `process_single_coin` function.
    5. Filter the results to only include coins with a cumulative score greater than the threshold.
    6. Plot the top coins over time using the historical data.
    7. Generate an HTML report with the results and send it via email.

    :return: None
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
        coins_to_monitor = api_call_with_retries(client.coins)
        logging.debug(f"Number of coins retrieved: {len(coins_to_monitor)}")
        coins_to_monitor = filter_active_and_ranked_coins(coins_to_monitor, NUMBER_OF_TOP_COINS_TO_MONITOR)

    logging.debug(f"Number of active and ranked coins selected: {len(coins_to_monitor)}")
    end_date = datetime.now().strftime('%Y-%m-%d')

    tickers_dict = load_tickers(CRYPTO_NEWS_TICKERS)
    sundown_digest = get_sundown_digest()
    digest_summary = summarize_sundown_digest(sundown_digest)
    digest_tickers = digest_summary['tickers']
    trending_coins_scores = fetch_trending_coins_scores()
    santiment_slugs_df = fetch_santiment_slugs()

    score_usage = defaultdict(list)  # <-- Add here

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = executor.map(
            lambda coin: process_single_coin(
                coin,
                existing_results,
                tickers_dict,
                digest_tickers,
                trending_coins_scores,
                santiment_slugs_df,
                end_date,
                score_usage
            ),
            coins_to_monitor
        )
        report_entries = [r for r in tqdm(futures, total=len(coins_to_monitor), desc="Processing Coins") if r is not None]

    df = pd.DataFrame(report_entries)

    try:
        if not df.empty:

            df = df[(df['liquidity_risk'].isin(['Low', 'Medium'])) & (df['cumulative_score_percentage'] > CUMULATIVE_SCORE_REPORTING_THRESHOLD)]

            logging.debug("DataFrame is not empty, processing report entries.")
            coins_in_df = df['coin_name'].unique()

            if len(coins_in_df) > 0:
                historical_data = retrieve_historical_data_from_aurora()

                if not historical_data.empty:
                    plot_top_coins_over_time(historical_data[historical_data['coin_name'].isin(coins_in_df)], top_n=10)

            report_entries = df.to_dict('records')
            report_entries = sorted(report_entries, key=lambda x: x.get('cumulative_score', 0), reverse=True)
            logging.debug(f"Report entries after sorting: {report_entries}")
            
            # Ensure numeric fields are correctly typed
            numeric_fields = [
                "price_change_score", "volume_change_score", "sentiment_score",
                "surging_keywords_score", "news_digest_score", "trending_score",
                "santiment_score", "cumulative_score", "cumulative_score_percentage",
                "fear_and_greed_index", "market_cap", "volume_24h", "events"
            ]

            for field in numeric_fields:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce')

            logging.debug("DataFrame contents before GPT-4o recommendations:\n%s", df.to_string())

            gpt_recommendations = gpt4o_summarize_each_coin(df)
            logging.debug(f"GPT-4o recommendations: {gpt_recommendations}")

            html_report = generate_html_report_with_recommendations(report_entries, digest_summary, gpt_recommendations)
            logging.debug("HTML report generated successfully.")

            attachment_path = save_report_to_excel(report_entries)
            logging.debug(f"Report saved to Excel at: {attachment_path}")

            send_email_with_report(html_report, attachment_path, recommendations=gpt_recommendations['recommendations'])
            logging.debug("Email sent successfully.")

            current_date = datetime.now().strftime("%Y-%m-%d")
            results_file = LOG_DIR + f"/results_{current_date}.csv"

            if os.path.exists(results_file):
                try:
                    os.remove(results_file)
                    logging.debug(f"{results_file} has been deleted successfully.")
                except Exception as e:
                    logging.debug(f"Failed to delete {results_file}: {e}")
            
            summarize_scores(score_usage, output_dir=LOG_DIR)

        else:
            logging.debug("No valid entries to report. DataFrame is empty.")
    except Exception as e:
        logging.error(f"An error occurred during the report generation process: {e}")

if __name__ == "__main__":
    monitor_coins_and_send_report()