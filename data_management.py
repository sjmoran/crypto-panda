import os
import pandas as pd
import psycopg2
from datetime import datetime
from sqlalchemy import create_engine
from config import ( LOG_DIR
)
from psycopg2 import OperationalError
import logging 
import glob
from sqlalchemy.exc import SQLAlchemyError

def load_tickers(file_path):
    """
    Loads a CSV file containing coin names and tickers, and returns a dictionary mapping
    the coin names to their tickers.

    Parameters:
        file_path (str): The path to the CSV file to load.

    Returns:
        dict: A dictionary mapping coin names to their tickers.
    """
    tickers_df = pd.read_csv(file_path)
    # Create a dictionary mapping the coin names to their tickers
    tickers_dict = pd.Series(tickers_df['Ticker'].values, index=tickers_df['Name']).to_dict()
    return tickers_dict

def save_result_to_csv(result):
    """
    Saves a single result as a row in a CSV file for the current date.

    The result will be appended to the existing file if it exists, or written to a new file if not.

    Parameters:
        result (dict): A dictionary containing at least the keys 'coin', 'market_cap', 'volume_24h', 
        'price_change_7d', and 'fear_greed_index'.
    """
    # Get current date as a string (e.g., '2024-10-03')
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create a filename with the current date
    results_file = os.path.join(LOG_DIR, f"results_{current_date}.csv")

    # Check if today's results file exists
    if not os.path.exists(results_file):
        # If the file doesn't exist, create it with headers
        pd.DataFrame([result]).to_csv(results_file, mode='w', header=True, index=False)
    else:
        # If the file exists, append to it without writing headers again
        pd.DataFrame([result]).to_csv(results_file, mode='a', header=False, index=False)


def retrieve_historical_data_from_aurora():
    """
    Retrieves historical cumulative scores from Amazon Aurora for all coins
    within the past 2 months.

    This function assumes that the environment variables AURORA_USER, AURORA_PASSWORD,
    AURORA_HOST, AURORA_PORT, and AURORA_DB are set to point to the
    Amazon Aurora database.

    Returns:
        pd.DataFrame: A DataFrame containing the timestamp, coin name, and cumulative score.
    """
    engine = None
    try:
        # Build the database connection string
        db_connection_str = (
            f"postgresql://{os.getenv('AURORA_USER')}:{os.getenv('AURORA_PASSWORD')}"
            f"@{os.getenv('AURORA_HOST')}:{os.getenv('AURORA_PORT', 5432)}/{os.getenv('AURORA_DB')}"
        )

        # Create an SQLAlchemy engine
        engine = create_engine(db_connection_str)

        # Define the SQL query to retrieve data from the past 2 months
        query = """
            SELECT coin_name, cumulative_score, timestamp 
            FROM coin_data
            WHERE timestamp >= NOW() - INTERVAL '2 months'
            ORDER BY timestamp;
        """
        
        # Use pandas to execute the query and return the result as a DataFrame
        # the past 2 months.
        df = pd.read_sql(query, engine)
        print("Historical data (last 2 months) retrieved successfully.")
        return df

    except SQLAlchemyError as e:
        print(f"Error retrieving historical data: {e}")

        return pd.DataFrame()  # Return empty DataFrame on failure

        # The read_sql function takes the query and the engine, and returns a DataFrame
    finally:

        # Print a message if the query is successful
        if engine:
            engine.dispose()  # Close the connection
            print("PostgreSQL connection is closed.")

def load_tickers(file_path):
    """
    Loads a CSV file containing coin names and tickers, and returns a dictionary mapping
    the coin names to their tickers.

    Parameters:
        file_path (str): The path to the CSV file to load.

    Returns:
        dict: A dictionary mapping coin names to their tickers.
    """
    tickers_df = pd.read_csv(file_path)
    # Create a dictionary mapping the coin names to their tickers
    tickers_dict = pd.Series(tickers_df['Ticker'].values, index=tickers_df['Name']).to_dict()
    return tickers_dict


def save_cumulative_score_to_aurora(coin_id, coin_name, cumulative_score):
    """
    Save a cumulative score for a specific coin in Amazon Aurora (PostgreSQL) with a date-based timestamp.

    Parameters:
        coin_id (str): The unique identifier for the coin.
        coin_name (str): The name of the coin.
        cumulative_score (float): The cumulative score of the coin.
    """
    connection = None  # Initialize connection variable
    cursor = None  # Initialize cursor variable
    
    try:
        # Establish connection to PostgreSQL Aurora instance
        connection = psycopg2.connect(
            host=os.getenv('AURORA_HOST'),
            database=os.getenv('AURORA_DB'),
            user=os.getenv('AURORA_USER'),
            password=os.getenv('AURORA_PASSWORD'),
            port=os.getenv('AURORA_PORT', 5432)  # Default port for PostgreSQL is 5432
        )
        
        cursor = connection.cursor()

        # Insert the cumulative score with the current date (no time part)
        insert_query = """
            INSERT INTO coin_data (coin_id, coin_name, cumulative_score, timestamp)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (coin_id, timestamp) 
            DO UPDATE SET cumulative_score = EXCLUDED.cumulative_score;
        """
        
        # Truncate timestamp to just the day (remove time component)
        current_date = datetime.now().date()  # Get only the date part
        
        cursor.execute(insert_query, (coin_id, coin_name, cumulative_score, current_date))
        
        connection.commit()
        print(f"Cumulative score for {coin_name} saved/updated successfully for {current_date}.")
    
    except psycopg2.OperationalError as e:
        print(f"Error connecting to Amazon Aurora DB: {e}")
    
    finally:
        # Check if cursor was created and close it
        if cursor is not None:
            try:
                cursor.close()
                print("Cursor is closed.")
            except Exception as e:
                print(f"Error closing cursor: {e}")

        # Check if connection was created and close it
        if connection is not None:
            try:
                connection.close()
                print("PostgreSQL connection is closed.")
            except Exception as e:
                print(f"Error closing connection: {e}")



def load_existing_results():
    """
    Loads existing results from the CSV file for the current date.
    
    If the file for the current date does not exist, all other 'results_' CSV files are deleted, and an empty DataFrame is returned.

    Parameters:
        None

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the existing results, or an empty DataFrame if no file exists for the current date.
    """
    def adjust_row_length(row, expected_columns=20):
        # Adjust rows with missing data by filling in default values (e.g., None)
        if len(row) < expected_columns:
            row += [None] * (expected_columns - len(row))  # Fill missing fields with None
        return row

    # Get the current date as a string (e.g., '2024-10-03')
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Construct the expected file name
    results_file = os.path.join(LOG_DIR, f"results_{current_date}.csv")

    # Check if the file exists for the current date
    if not os.path.exists(results_file):
        logging.debug(f"File {results_file} does not exist. Removing all old results files.")

        # Remove all other CSV files that start with 'results_'
        for file in glob.glob('results_*.csv'):
            try:
                os.remove(file)
                logging.info(f"Deleted old results file: {file}")
            except Exception as e:
                logging.error(f"Failed to delete file {file}: {e}")

        # Return an empty DataFrame since no file exists for today
        return pd.DataFrame()

    try:
        # Read the CSV and treat the first row as the header (column names)
        df = pd.read_csv(results_file, header=0, delimiter=',', engine='python', on_bad_lines='skip')

        # Get the number of expected columns from the DataFrame's columns
        expected_columns = len(df.columns)

        # Convert DataFrame rows to lists for manual adjustment
        adjusted_rows = df.apply(lambda row: adjust_row_length(list(row), expected_columns), axis=1)

        # Convert back to DataFrame after adjustment, using the original column names
        adjusted_df = pd.DataFrame(adjusted_rows.tolist(), columns=df.columns)

        return adjusted_df

    except FileNotFoundError:
        logging.error(f"File {results_file} not found.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is not found

    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if parsing fails

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame for any other error

def create_coin_data_table_if_not_exists():
    """
    Creates the 'coin_data' table in Amazon Aurora (PostgreSQL) if it doesn't already exist,
    storing time series data for cumulative scores.
    """
    connection = None  # Initialize the connection variable to None
    try:
        # Connect to PostgreSQL Aurora instance
        connection = psycopg2.connect(
            host=os.getenv('AURORA_HOST'),
            database=os.getenv('AURORA_DB'),
            user=os.getenv('AURORA_USER'),
            password=os.getenv('AURORA_PASSWORD'),
            port=os.getenv('AURORA_PORT', 5432)  # Default port for PostgreSQL is 5432
        )
        
        cursor = connection.cursor()

        # SQL to create the table if it doesn't exist, allowing time series data
        create_table_query = """
        CREATE TABLE IF NOT EXISTS coin_data (
            id SERIAL PRIMARY KEY,
            coin_id VARCHAR(255) NOT NULL,
            coin_name VARCHAR(255) NOT NULL,
            cumulative_score FLOAT NOT NULL,
            timestamp DATE DEFAULT CURRENT_DATE,
            UNIQUE (coin_id, timestamp)  -- Unique constraint to ensure one entry per coin per day
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        print("Table created or already exists.")

    except OperationalError as e:
        print(f"Error while connecting to Amazon Aurora: {e}")
    
    finally:
        # Close the connection if it was successfully created
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed.")
