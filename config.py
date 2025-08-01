import os
from dotenv import load_dotenv  # Allows loading environment variables from a .env file
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Sentiment analysis tool

# Path to your .env file where secrets and config are stored
DOT_ENV_PATH = "/home/ec2-user/crypto-panda/crypto-panda/.env"

# Load environment variables from the specified .env file
load_dotenv(dotenv_path=DOT_ENV_PATH)

# Sanpy API key (used for fetching crypto metrics)
SAN_API_KEY = os.getenv('SAN_API_KEY')

# Directory to store logs
LOG_DIR = '../logs/'

# Directory to store data files (e.g., price data, tickers)
DATA_DIR = '../data/'

# Keywords used to detect bullish or surge-related language in crypto news
surge_words = [
    # Existing
    "surge", "spike", "soar", "rocket", "skyrocket", "rally", "boom", "bullish",
    "explosion", "rise", "uptrend", "bull run", "moon", "parabolic", "spurt",
    "climb", "jump", "upswing", "gain", "increase", "growth", "rebound",
    "breakout", "pump", "fly", "explode", "shoot up", "hike",
    "expand", "appreciate", "bull market", "peak", "momentum", "outperform",
    "spike up", "ascend", "elevation", "expansion", "revive", "uprising",
    "push up", "escalate", "rise sharply", "escalation", "recover",
    "inflation", "strengthen", "gain strength", "intensify",
    
    # ✅ Crypto slang / community terms
    "send it", "going vertical", "breaking out", "melting faces", "green candle",
    "altseason", "bull flag", "supercycle", "squeeze", "ripping", "face melting",
    
    # ✅ Financial/technical analysis language
    "overbought", "golden cross", "ascending triangle", "channel breakout",
    "breakout pattern", "positive divergence", "momentum shift", "buy pressure",
    
    # ✅ Social/hype indicators
    "FOMO", "buzzing", "hype", "trending", "hot", "popular", "buzzword", "popping",
    
    # ✅ Strong verbs and upward metaphors
    "accelerate", "take off", "launch", "ignite", "trigger", "break resistance",
    "snap resistance", "catalyst", "momentum build", "price discovery"
]

# Daily trading volume thresholds to assess liquidity risk by market cap
LOW_VOLUME_THRESHOLD_LARGE = 1_000_000   # Large-cap coins considered illiquid if volume < $1M
LOW_VOLUME_THRESHOLD_MID = 500_000       # Mid-cap coins considered illiquid if volume < $500K
LOW_VOLUME_THRESHOLD_SMALL = 100_000     # Small-cap coins considered illiquid if volume < $100K

# Email configuration variables, pulled from environment
EMAIL_FROM = os.getenv('EMAIL_FROM')           # Sender email address
EMAIL_TO = os.getenv('EMAIL_TO')               # Recipient email address
SMTP_SERVER = os.getenv('SMTP_SERVER')         # SMTP server address (e.g., smtp.gmail.com)
SMTP_USERNAME = os.getenv('SMTP_USERNAME')     # SMTP login username
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')     # SMTP login password
SMTP_PORT = 587                                # SMTP port (587 is standard for TLS)

# File paths used in the analysis pipeline
RESULTS_FILE = LOG_DIR + "/surging_coins.csv"          # Where analysis results are saved
CRYPTO_NEWS_TICKERS = DATA_DIR + "/tickers.csv"        # List of tickers for filtering news

# Thresholds used to score coins
FEAR_GREED_THRESHOLD = 60               # Only consider coins if fear/greed index > 60
HIGH_VOLATILITY_THRESHOLD = 0.05        # >5% daily price movement is high volatility
MEDIUM_VOLATILITY_THRESHOLD = 0.02      # 2–5% is medium volatility
NUMBER_OF_TOP_COINS_TO_MONITOR = 2000     # Limit number of top coins considered in reports

# Control flags for running/testing
TEST_ONLY = False                       # Set to True to test without triggering actions
MAX_RETRIES = 2                         # Number of retries for failed API requests
BACKOFF_FACTOR = 2                      # Delay multiplier between retries (e.g., 2, 4, 8 seconds)

# Only include coins in reports if they exceed this cumulative score threshold
CUMULATIVE_SCORE_REPORTING_THRESHOLD = 40

# Aurora PostgreSQL configuration (used to log or retrieve coin scores)
AURORA_HOST = os.getenv('AURORA_HOST')              # Database host endpoint
AURORA_PORT = os.getenv('AURORA_PORT', 5432)        # Default port for PostgreSQL
AURORA_DB = os.getenv('AURORA_DB')                  # Name of the database
AURORA_USER = os.getenv('AURORA_USER')              # Database user
AURORA_PASSWORD = os.getenv('AURORA_PASSWORD')      # Database password

# CoinPaprika API key for fetching coin metadata and price data
COIN_PAPRIKA_API_KEY = os.getenv('COIN_PAPRIKA_API_KEY')

# Initialize the VADER sentiment analyzer (used to evaluate sentiment in news headlines/text)
analyzer = SentimentIntensityAnalyzer()