def normalize_score(raw_score, min_score, max_score, range=3):
    """
    Normalize a score to be between 0 and 3 and return the nearest integer.

    Args:
        raw_score (float): The original score.
        min_score (float): The minimum score in the data.
        max_score (float): The maximum score in the data.

    Returns:
        int: The normalized score between 0 and 3, rounded to the nearest integer.
    """
    # Handle the edge case where all scores might be the same
    if max_score == min_score:
        return 2  # Midpoint value since all scores are the same

    # Normalize to a 0-1 range
    normalized = (raw_score - min_score) / (max_score - min_score)
    # Scale to a 0-3 range and round to the nearest integer
    return round(normalized * range)



def calculate_price_change(price_data, period="short", span=7):
    """
    Calculate the percentage change in price over a given period with optional smoothing using Exponential Moving Average (EMA).

    Parameters:
        price_data (pd.DataFrame): Price data for the cryptocurrency with a single column.
        period (str, optional): The time period to calculate the price change for. Options are "short" (7 days), "medium" (30 days), and "long" (90 days). Defaults to "short".
        span (int, optional): The span for the EMA smoothing. Defaults to 7.

    Returns:
        float: The percentage change in price over the specified period, smoothed by EMA.
    """
    # Apply EMA
    smoothed_data = price_data.ewm(span=span, adjust=False).mean()

    if period == "short":
        period_data = smoothed_data.tail(7)  # Last 7 days
    elif period == "medium":
        period_data = smoothed_data.tail(30)  # Last 30 days
    else:  # long term
        period_data = smoothed_data.tail(90)  # Last 90 days

    start_price = period_data.iloc[0]  # Extract the first value from the single column
    end_price = period_data.iloc[-1]  # Extract the last value from the single column

    if start_price == 0:
        return None  # or some other appropriate value or handling mechanism
    return (end_price - start_price) / start_price

def calculate_volume_change(volume_data, period="short", span=7):
    """
    Calculate the percentage change in volume over a given period with optional smoothing using Exponential Moving Average (EMA).

    Parameters:
        volume_data (pd.DataFrame): Volume data for the cryptocurrency with a single column.
        period (str, optional): The time period to calculate the volume change for. Options are "short" (7 days), "medium" (30 days), and "long" (90 days). Defaults to "short".
        span (int, optional): The span for the EMA smoothing. Defaults to 7.

    Returns:
        float: The percentage change in volume over the specified period, smoothed by EMA.
    """
    # Apply EMA
    smoothed_data = volume_data.ewm(span=span, adjust=False).mean()

    if period == "short":
        period_data = smoothed_data.tail(7)  # Last 7 days
    elif period == "medium":
        period_data = smoothed_data.tail(30)  # Last 30 days
    else:  # long term
        period_data = smoothed_data.tail(90)  # Last 90 days

    start_volume = period_data.iloc[0]  # Extract the first value from the single column
    end_volume = period_data.iloc[-1]  # Extract the last value from the single column

    if start_volume == 0:
        return None  # or some other appropriate value or handling mechanism
    return (end_volume - start_volume) / start_volume


def filter_active_and_ranked_coins(coins, max_coins=250, rank_threshold=1000):
    """
    Filters the list of coins to focus on the bottom-ranked coins, selecting up to max_coins.

    Parameters:
        coins (list): List of coins with rank, activity status, and new status information.
        max_coins (int): The maximum number of coins to return.
        rank_threshold (int): The maximum rank a coin must have to be included (e.g., rank <= 1000).

    Returns:
        list: Filtered list of coins that are active, not new, and ranked within the bottom of the rank_threshold.
    """
    # Filter out coins that are not active, are new, or have a rank above the rank_threshold
    active_ranked_coins = [
        coin for coin in coins
        if coin.get('is_active', False)
        and not coin.get('is_new', True)
        and 1 <= coin.get('rank', None) <= max_coins
    ]

    # Limit the list to max_coins
    return active_ranked_coins[:max_coins]

