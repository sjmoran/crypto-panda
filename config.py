import os
from dotenv import load_dotenv  # Load dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load environment variables from .env file
load_dotenv()

# Initialize Sanpy API key
SAN_API_KEY = os.getenv('SAN_API_KEY')

LOG_DIR='../logs/'
DATA_DIR='../data/'

# Surge-related words
surge_words = [
    "surge", "spike", "soar", "rocket", "skyrocket", "rally", "boom", "bullish", 
    "explosion", "rise", "uptrend", "bull run", "moon", "parabolic", "spurt", 
    "climb", "jump", "upswing", "gain", "increase", "growth", "rebound", 
    "breakout", "spurt", "pump", "fly", "explode", "shoot up", "hike", 
    "expand", "appreciate", "bull market", "peak", "momentum", "outperform", 
    "spike up", "ascend", "elevation", "expansion", "revive", "uprising", 
    "push up", "escalate", "rise sharply", "escalation", "recover", 
    "inflation", "strengthen", "gain strength", "intensify"
]

# Volume thresholds for liquidity risk
LOW_VOLUME_THRESHOLD_LARGE = 1_000_000  # Large-cap coins with daily volume under $1M
LOW_VOLUME_THRESHOLD_MID = 500_000  # Mid-cap coins with daily volume under $500k
LOW_VOLUME_THRESHOLD_SMALL = 100_000  # Small-cap coins with daily volume under $100k

# Email configuration
EMAIL_FROM = os.getenv('CRYPTOPANDA_EMAIL_FROM')
EMAIL_TO = os.getenv('EMAIL_TO')
SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
SMTP_PORT = 587

DOT_ENV_PATH="/home/ec2-user/crypto-panda/crypto-panda/.env"

# Files and Tickers
RESULTS_FILE = LOG_DIR + "/surging_coins.csv"
CRYPTO_NEWS_TICKERS = DATA_DIR + "/tickers.csv"

# Score thresholds
FEAR_GREED_THRESHOLD = 60  # Fear and Greed index threshold
HIGH_VOLATILITY_THRESHOLD = 0.05  # 5% volatility is considered high
MEDIUM_VOLATILITY_THRESHOLD = 0.02  # 2% volatility is considered medium
NUMBER_OF_TOP_COINS_TO_MONITOR = 500

# Testing and retries
TEST_ONLY = False  # Set to False to monitor all coins
MAX_RETRIES = 2  # Maximum number of retries for API calls
BACKOFF_FACTOR = 2  # Factor by which the wait time increases after each failure

# Reporting
CUMULATIVE_SCORE_REPORTING_THRESHOLD = 40  # Only report results with cumulative score above this % value

AURORA_HOST = os.getenv('AURORA_HOST')  # Make sure this points to the correct server
AURORA_PORT = os.getenv('AURORA_PORT', 5432)  # Ensure the port is correct (default is 5432)
AURORA_DB = os.getenv('AURORA_DB')
AURORA_USER = os.getenv('AURORA_USER')
AURORA_PASSWORD = os.getenv('AURORA_PASSWORD')

COIN_PAPRIKA_API_KEY=os.getenv('COIN_PAPRIKA_API_KEY')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()