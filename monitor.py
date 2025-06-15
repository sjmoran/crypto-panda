import os
import time
import pandas as pd
import logging
from datetime import datetime
from api_clients import (
    get_sundown_digest,
    fetch_trending_coins_scores,
    fetch_news_for_past_week,
)
from coin_analysis import analyze_coin
from data_management import (
    save_result_to_csv,
    retrieve_historical_data_from_aurora,
    save_cumulative_score_to_aurora,
    create_coin_data_table_if_not_exists
)
from plotting import plot_top_coins_over_time
from report_generation import gpt4o_analyze_and_recommend,save_report_to_excel, summarize_sundown_digest, generate_html_report_with_recommendations, send_email_with_report
from config import (
    TEST_ONLY,
    CUMULATIVE_SCORE_REPORTING_THRESHOLD,
    NUMBER_OF_TOP_COINS_TO_MONITOR,
    CRYPTO_NEWS_TICKERS,
    TEST_ONLY,
    LOG_DIR
)

import re  # Add the missing import for regular expressions
from dotenv import load_dotenv
from coinpaprika import client as Coinpaprika
from data_management import load_existing_results
from api_clients import api_call_with_retries
from helpers import filter_active_and_ranked_coins
from data_management import load_tickers
from config import COIN_PAPRIKA_API_KEY
from api_clients import (
    fetch_santiment_slugs, client
)
import traceback
import logging 

# Load environment variables from .env
load_dotenv()

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all debug logs
    format='%(asctime)s - %(levelname)s - %(message)s',  # Customize log format
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(LOG_DIR+'/crypto-panda.log', mode='w')  # Log to a file (optional)
    ]
)

logging.debug("Logging is set up and the application has started.")

# Initialize the CoinPaprika client
client = Coinpaprika.Client(api_key=os.getenv('COIN_PAPRIKA_API_KEY'))

def monitor_coins_and_send_report():
    """
    Main entry point to monitor the specified coins, fetch news, analyze sentiment,
    and send a report with the results.

    If TEST_ONLY is set to True, only a few coins are processed and the results are
    saved to a file. Otherwise, all coins are processed and the results are sent
    via email.
    """
    # Ensure the table is created in Amazon Aurora if it doesn't exist
    create_coin_data_table_if_not_exists()

    if TEST_ONLY:
        existing_results = pd.DataFrame([])
        # Add a predefined mix of small, medium, and large-cap coins
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

    report_entries = []
    tickers_dict = load_tickers(CRYPTO_NEWS_TICKERS)

    # Fetch and summarize the Sundown Digest
    sundown_digest = get_sundown_digest()
    digest_summary = summarize_sundown_digest(sundown_digest)
    digest_tickers = digest_summary['tickers']

    # Fetch Trending Coins data once
    trending_coins_scores = fetch_trending_coins_scores()

    # Load Santiment slugs
    santiment_slugs_df = fetch_santiment_slugs()

    for coin in coins_to_monitor:
        try:
            print(f"Processing {coin['name']} ({coin['id']})")
            coin_id = coin['id']
            coin_name = coin['name'].lower()

            if existing_results is not None and not existing_results.empty and coin_id in existing_results['coin_id'].values:
                logging.debug(f"Skipping already processed coin: {coin_id}")
                continue

            # Fetch news directly for analysis
            coins_dict = {coin_name: tickers_dict.get(coin_name, '').upper()}
            news_df = fetch_news_for_past_week(coins_dict)

            # Analyze coin and save the result, passing the Santiment slug
            result = analyze_coin(coin_id, coin_name, end_date, news_df, digest_tickers, trending_coins_scores, santiment_slugs_df)
            logging.debug(f"Result for {coin_name}: {result}")

            save_result_to_csv(result)
            report_entries.append(result)

            # Save the cumulative score to Amazon Aurora
            save_cumulative_score_to_aurora(result['coin_id'], result['coin_name'], result['cumulative_score_percentage'])

            time.sleep(20)

        except Exception as e:
            logging.debug(f"An error occurred while processing {coin_name} ({coin_id}): {e}")
            logging.debug(traceback.format_exc())
            continue

    df = pd.DataFrame(report_entries)

    try:
        if not df.empty:
            # Filter the report_entries based on liquidity risk and cumulative score percentage
            df = df[(df['liquidity_risk'].isin(['Low', 'Medium'])) & (df['cumulative_score_percentage'] > CUMULATIVE_SCORE_REPORTING_THRESHOLD)]

            logging.debug("DataFrame is not empty, processing report entries.")

            # Extract the coin names from the filtered DataFrame
            coins_in_df = df['coin_name'].unique()  # Extract unique coin names from the filtered DataFrame

            if len(coins_in_df) > 0:
                # Retrieve historical data from Amazon Aurora
                historical_data = retrieve_historical_data_from_aurora()

                if not historical_data.empty:
                    # Filter the historical data for only the coins present in the filtered df
                    plot_top_coins_over_time(historical_data[historical_data['coin_name'].isin(coins_in_df)], top_n=10)

            # Proceed with sorting and generating the report
            report_entries = df.to_dict('records')
            report_entries = sorted(report_entries, key=lambda x: x.get('cumulative_score', 0), reverse=True)
            logging.debug(f"Report entries after sorting: {report_entries}")

            logging.debug(f"DataFrame contents before GPT-4o recommendations:\n{df.to_string()}")

            # Get GPT-4o recommendations
            gpt_recommendations = gpt4o_analyze_and_recommend(df)
            logging.debug(f"GPT-4o recommendations: {gpt_recommendations}")

            # Generate HTML report with recommendations
            html_report = generate_html_report_with_recommendations(report_entries, digest_summary, gpt_recommendations)
            logging.debug("HTML report generated successfully.")

            # Save the report to Excel
            attachment_path = save_report_to_excel(report_entries)
            logging.debug(f"Report saved to Excel at: {attachment_path}")

            # Send email with the report and the plot attached
            send_email_with_report(html_report, attachment_path, recommendations=gpt_recommendations['recommendations'])
            logging.debug("Email sent successfully.")

            # Delete the results CSV file after sending the email
            current_date = datetime.now().strftime("%Y-%m-%d")
            results_file = LOG_DIR+f"/results_{current_date}.csv"

            if os.path.exists(results_file):
                try:
                    os.remove(results_file)
                    logging.debug(f"{results_file} has been deleted successfully.")
                except Exception as e:
                    logging.debug(f"Failed to delete {results_file}: {e}")
        else:
            logging.debug("No valid entries to report. DataFrame is empty.")
    except Exception as e:
        logging.error(f"An error occurred during the report generation process: {e}")


# Main function call
if __name__ == "__main__":
    monitor_coins_and_send_report()
