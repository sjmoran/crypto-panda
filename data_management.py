#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd
import psycopg2
from psycopg2 import OperationalError
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from config import LOG_DIR

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

    # Clear existing handlers to avoid dupes on re-imports
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
# Helpers
# ----------------------------

def _utc_today_str() -> str:
    """YYYY-MM-DD (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _ensure_dir(p: Union[str, Path]) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

# ----------------------------
# Tickers I/O
# ----------------------------

def load_tickers(file_path: str) -> Dict[str, str]:
    """
    Loads a CSV file containing coin names and tickers, and returns a dictionary mapping
    the coin names to their tickers.
    """
    df = pd.read_csv(file_path)
    tickers_dict = pd.Series(df['Ticker'].values, index=df['Name']).to_dict()
    logger.info(f"Loaded {len(tickers_dict)} tickers from {file_path}")
    return tickers_dict

# ----------------------------
# Results CSV (daily, UTC)
# ----------------------------

def save_result_to_csv(result: Mapping[str, object]) -> None:
    """
    Appends a single result as a row in a CSV file for the current UTC date.
    Creates the file with headers if it does not exist.
    """
    _ensure_dir(LOG_DIR)
    current_date = _utc_today_str()
    results_file = os.path.join(LOG_DIR, f"results_{current_date}.csv")

    try:
        exists = os.path.exists(results_file)
        pd.DataFrame([result]).to_csv(
            results_file,
            mode='a' if exists else 'w',
            header=not exists,
            index=False
        )
        logger.info(f"Wrote result row to {results_file}")
    except Exception as e:
        logger.error(f"Failed to write result to {results_file}: {e}")

def load_existing_results() -> pd.DataFrame:
    """
    Loads existing results from today's CSV (UTC).
    If today's file does not exist, deletes other 'results_*.csv' files in LOG_DIR and returns empty DataFrame.
    """
    def adjust_row_length(row, expected_columns=20):
        if len(row) < expected_columns:
            row += [None] * (expected_columns - len(row))
        return row

    current_date = _utc_today_str()
    results_file = os.path.join(LOG_DIR, f"results_{current_date}.csv")

    if not os.path.exists(results_file):
        logger.debug(f"File {results_file} does not exist. Removing all old results files in {LOG_DIR}.")
        try:
            for file in glob.glob(os.path.join(LOG_DIR, 'results_*.csv')):
                try:
                    os.remove(file)
                    logger.info(f"Deleted old results file: {file}")
                except Exception as e:
                    logger.error(f"Failed to delete file {file}: {e}")
        except Exception as e:
            logger.error(f"Error during cleanup in {LOG_DIR}: {e}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(results_file, header=0, delimiter=',', engine='python', on_bad_lines='skip')
        expected_columns = len(df.columns)
        adjusted_rows = df.apply(lambda row: adjust_row_length(list(row), expected_columns), axis=1)
        adjusted_df = pd.DataFrame(adjusted_rows.tolist(), columns=df.columns)
        logger.info(f"Loaded {len(adjusted_df)} rows from {results_file}")
        return adjusted_df

    except FileNotFoundError:
        logger.error(f"File {results_file} not found after existence check.")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing {results_file}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error reading {results_file}: {e}")
        return pd.DataFrame()

# ----------------------------
# Aurora (PostgreSQL) I/O
# ----------------------------

def retrieve_historical_data_from_aurora() -> pd.DataFrame:
    """
    Retrieves historical cumulative scores from Amazon Aurora for the past 2 months.
    """
    engine = None
    try:
        db_connection_str = (
            f"postgresql://{os.getenv('AURORA_USER')}:{os.getenv('AURORA_PASSWORD')}"
            f"@{os.getenv('AURORA_HOST')}:{os.getenv('AURORA_PORT', 5432)}/{os.getenv('AURORA_DB')}"
        )
        engine = create_engine(db_connection_str)
        query = """
            SELECT coin_name, cumulative_score, timestamp
            FROM coin_data
            WHERE timestamp >= NOW() - INTERVAL '2 months'
            ORDER BY timestamp;
        """
        df = pd.read_sql(query, engine)
        logger.info(f"Historical data (last 2 months) retrieved successfully: {len(df)} rows")
        return df

    except SQLAlchemyError as e:
        logger.error(f"Error retrieving historical data: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error retrieving historical data: {e}")
        return pd.DataFrame()
    finally:
        if engine:
            try:
                engine.dispose()
                logger.info("SQLAlchemy engine disposed.")
            except Exception as e:
                logger.error(f"Error disposing engine: {e}")

def save_cumulative_score_to_aurora(coin_id: str, coin_name: str, cumulative_score: float) -> None:
    """
    Save a cumulative score for a specific coin with a UTC date-based timestamp.
    """
    connection = None
    cursor = None
    try:
        connection = psycopg2.connect(
            host=os.getenv('AURORA_HOST'),
            database=os.getenv('AURORA_DB'),
            user=os.getenv('AURORA_USER'),
            password=os.getenv('AURORA_PASSWORD'),
            port=os.getenv('AURORA_PORT', 5432)
        )
        cursor = connection.cursor()
        insert_query = """
            INSERT INTO coin_data (coin_id, coin_name, cumulative_score, timestamp)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (coin_id, timestamp)
            DO UPDATE SET cumulative_score = EXCLUDED.cumulative_score;
        """
        current_date = datetime.now(timezone.utc).date()
        cursor.execute(insert_query, (coin_id, coin_name, cumulative_score, current_date))
        connection.commit()
        logger.info(f"Cumulative score saved/updated for {coin_name} on {current_date} (score={cumulative_score}).")

    except OperationalError as e:
        logger.error(f"Error connecting to Amazon Aurora DB: {e}")
    except Exception as e:
        logger.error(f"Error saving cumulative score for {coin_name}: {e}")
    finally:
        if cursor is not None:
            try:
                cursor.close()
                logger.debug("Cursor closed.")
            except Exception as e:
                logger.error(f"Error closing cursor: {e}")
        if connection is not None:
            try:
                connection.close()
                logger.debug("PostgreSQL connection closed.")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

def create_coin_data_table_if_not_exists() -> None:
    """
    Creates the 'coin_data' table in Amazon Aurora (PostgreSQL) if it doesn't already exist,
    storing time series data for cumulative scores.
    """
    connection = None
    cursor = None
    try:
        connection = psycopg2.connect(
            host=os.getenv('AURORA_HOST'),
            database=os.getenv('AURORA_DB'),
            user=os.getenv('AURORA_USER'),
            password=os.getenv('AURORA_PASSWORD'),
            port=os.getenv('AURORA_PORT', 5432)
        )
        cursor = connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS coin_data (
            id SERIAL PRIMARY KEY,
            coin_id VARCHAR(255) NOT NULL,
            coin_name VARCHAR(255) NOT NULL,
            cumulative_score FLOAT NOT NULL,
            timestamp DATE DEFAULT CURRENT_DATE,
            UNIQUE (coin_id, timestamp)
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        logger.info("Table 'coin_data' created or already exists.")

    except OperationalError as e:
        logger.error(f"Error while connecting to Amazon Aurora: {e}")
    except Exception as e:
        logger.error(f"Error creating table 'coin_data': {e}")
    finally:
        if cursor is not None:
            try:
                cursor.close()
                logger.debug("Cursor closed.")
            except Exception as e:
                logger.error(f"Error closing cursor: {e}")
        if connection is not None:
            try:
                connection.close()
                logger.debug("PostgreSQL connection closed.")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")