#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from pathlib import Path
from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
# Plotting
# ----------------------------

def plot_top_coins_over_time(
    historical_data: pd.DataFrame,
    top_n: int = 5,
    file_name: str = os.path.join(LOG_DIR, "top_coins_plot.png"),
    window: int = 5,
) -> None:
    """
    Plots the cumulative scores of the top coins over time with optional smoothing and saves the plot to a file.
    """
    try:
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

        # Ensure timestamp is datetime
        historical_data = historical_data.copy()
        historical_data.loc[:, 'timestamp'] = pd.to_datetime(historical_data['timestamp'])

        # Calculate average cumulative score and select top N
        top_coins = historical_data.groupby('coin_name')['cumulative_score'].mean().nlargest(top_n).index
        logger.info(f"Plotting top {len(top_coins)} coins: {list(top_coins)}")

        # Filter data
        top_data = historical_data[historical_data['coin_name'].isin(top_coins)]

        plt.figure(figsize=(10, 6))
        for coin in top_coins:
            coin_data = top_data[top_data['coin_name'] == coin].sort_values('timestamp')
            coin_data['smoothed_score'] = coin_data['cumulative_score'].rolling(window=window, min_periods=1).mean()
            plt.plot(coin_data['timestamp'], coin_data['smoothed_score'], label=coin, marker='o')

        # Format x-axis
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Labels and legend
        plt.title(f"Top {top_n} Coins by Cumulative Score Over Time")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Score")
        plt.legend()

        # Save
        plt.tight_layout()
        plt.savefig(file_name)
        logger.info(f"Plot saved → {file_name}")

    except Exception as e:
        logger.error(f"Error plotting top coins over time: {e}")