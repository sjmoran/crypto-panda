import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import ( LOG_DIR )

def plot_top_coins_over_time(historical_data, top_n=5, file_name=LOG_DIR+'/top_coins_plot.png', window=5):
    """
    Plots the cumulative scores of the top coins over time with optional smoothing and saves the plot to a file.

    Args:
        historical_data (pd.DataFrame): DataFrame containing the historical data with 'coin_name', 'cumulative_score', and 'timestamp' columns.
        top_n (int): The number of top coins to plot.
        file_name (str): The name of the file to save the plot to.
        window (int): The window size for rolling average smoothing (default: 5).
    """
    # Convert 'timestamp' to datetime format
    historical_data.loc[:, 'timestamp'] = pd.to_datetime(historical_data['timestamp'])

    # Calculate the average cumulative score for each coin and select the top N coins
    top_coins = historical_data.groupby('coin_name')['cumulative_score'].mean().nlargest(top_n).index

    # Filter data for only the top coins
    top_data = historical_data[historical_data['coin_name'].isin(top_coins)]

    # Plot each top coin's cumulative score over time with smoothing
    plt.figure(figsize=(10, 6))
    for coin in top_coins:
        coin_data = top_data[top_data['coin_name'] == coin].sort_values('timestamp')
        
        # Apply rolling average for smoothing
        coin_data['smoothed_score'] = coin_data['cumulative_score'].rolling(window=window, min_periods=1).mean()
        
        # Plot the smoothed data
        plt.plot(coin_data['timestamp'], coin_data['smoothed_score'], label=coin, marker='o')

    # Format x-axis with date formatting based on the range of dates in the data
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjust the date ticks
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the ticks as 'Year-Month-Day'

    # Plot settings
    plt.title(f'Top {top_n} Coins by Cumulative Score Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Score')
    plt.legend()

    # Save plot to file
    plt.tight_layout()
    plt.savefig(file_name)
    #plt.show()

