
# 🐼 Crypto-Panda: Cryptocurrency Analysis & Reporting Tool

[![GitHub Repo](https://img.shields.io/badge/GitHub-sjmoran%2Fcrypto--panda-blue?logo=github)](https://github.com/sjmoran/crypto-panda)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC--BY--NC%204.0-orange)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Stars](https://img.shields.io/github/stars/sjmoran/crypto-panda?style=social)](https://github.com/sjmoran/crypto-panda/stargazers)
[![Issues](https://img.shields.io/github/issues/sjmoran/crypto-panda)](https://github.com/sjmoran/crypto-panda/issues)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

<img src="./images/crypto_panda_trading.png" alt="Crypto Panda Trading" width="50%"/>

---

## 🔍 What is Crypto-Panda?

**Crypto-Panda** is a smart, automated reporting tool that tracks the cryptocurrency market, analyzes patterns using both quantitative signals and AI, and emails you a weekly report on the coins worth watching.

Powered by Python, OpenAI's GPT-4o, Santiment, and CoinPaprika APIs — it's designed to help you cut through the noise and make sense of the chaos.

---

## 🧠 What It Can Do

- 📈 **Market Trend Analysis**  
  Pulls historical price/volume data via CoinPaprika and analyzes short- and long-term trends.

- 🧪 **Santiment Intelligence**  
  Tracks developer activity, daily active addresses, and other on-chain signals.

- 📰 **News & Social Sentiment**  
  Uses VADER and GPT-4o to extract sentiment from headlines and social chatter.

- 🚀 **Surge Detection**  
  Uses a composite scoring mechanism to flag coins with breakout potential.

- 🤖 **GPT-4o Investment Suggestions**  
  Generates natural-language investment suggestions from raw data.

- 📬 **Weekly HTML Report by Email**  
  Fully automated and ready for inboxes.

- 🔁 **Reliable API Access**  
  Includes built-in retry handling for flaky requests.

---

## 📊 Supported Metrics in the Crypto Analysis Pipeline

| **Metric Name**               | **Score Range** | **Purpose / Description** |
|------------------------------|-----------------|----------------------------|
| `price_change_score`         | 0–3             | Assesses short-, medium-, and long-term price momentum relative to market cap and volatility thresholds. |
| `volume_change_score`        | 0–3             | Detects surges in trading volume over multiple periods, adjusted for market cap and volatility. |
| `tweet_score`                | 0–1             | Indicates whether there was any Twitter data retrieved for the coin (1 = present). |
| `sentiment_score`            | 0–1             | Evaluates average sentiment from news using VADER; returns 1 if sentiment is highly positive. |
| `surging_keywords_score`     | 0–1             | Scores presence of bullish phrases in news articles using fuzzy matching with a surge keyword list. |
| `consistent_growth_score`    | 0–1             | True if price rose on at least 4 of the last 7 days, signaling short-term bullish behavior. |
| `sustained_volume_growth`    | 0–1             | True if volume increased on at least 4 of the last 7 days, showing consistent demand. |
| `fear_and_greed_score`       | 0–1             | Based on the Alternative.me Fear & Greed Index; returns 1 if index exceeds a defined threshold. |
| `event_score`                | 0–1             | 1 if any significant recent events (within past 7 days) are associated with the coin. |
| `digest_score`               | 0–1             | Indicates whether the coin appears in the curated Sundown Digest list. |
| `trending_score`             | 0–2             | Uses fuzzy string matching to detect if the coin is trending in external APIs (e.g., CoinPaprika). |
| `santiment_score`            | 0–2             | Based on binary thresholds from Santiment metrics such as dev activity and daily active addresses. |
| `santiment_surge_score`      | 0–6             | Aggregates surge signals from Santiment metrics like net exchange flow, whale tx count, sentiment. |
| `consistent_monthly_growth`  | 0–1             | Checks for 18+ positive price change days in the past 30, signaling strong longer-term accumulation. |
| `trend_conflict_score`       | 0–1             | True if there's strong monthly growth without recent short-term confirmation — possible breakout signal. |
| `liquidity_risk`             | "Low"/"Medium"/"High" | Categorizes trading risk based on volume relative to market cap class. |
| `cumulative_score`           | 0–22            | Total score based on all signals above. Used to rank coins. |
| `cumulative_score_percentage`| 0–100%          | Normalized version of `cumulative_score` as a percentage of `MAX_POSSIBLE_SCORE`. |

---

## 📬 Example Report

Each weekly email includes top-ranked coins and GPT-generated insights:

<img src="./images/example_report.png" alt="AI Generated Crypto Coin Report" width="50%"/>

---

## ⚙️ Requirements

- Python 3.8+
- Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file with the following:

```env
COIN_PAPRIKA_API_KEY=
OPENAI_API_KEY=
CRYPTO_NEWS_API_KEY=
SAN_API_KEY=
EMAIL_FROM=
EMAIL_TO=
SMTP_SERVER=
SMTP_USERNAME=
SMTP_PASSWORD=
```

---

## 🚀 Quickstart

```bash
git clone https://github.com/sjmoran/crypto-panda.git
cd crypto-panda
pip install -r requirements.txt
python monitor.py
```

> 💡 Run weekly via cron, Lambda, or EventBridge.

---

## ⚙️ Config Options

| Variable                                | Description                                    |
|----------------------------------------|------------------------------------------------|
| `TEST_ONLY`                             | Run on a small subset for testing              |
| `RESULTS_FILE`                          | Output filename for saving results             |
| `HIGH_VOLATILITY_THRESHOLD`            | Flag coins with high volatility                |
| `FEAR_GREED_THRESHOLD`                 | Fear & Greed Index threshold                   |
| `CUMULATIVE_SCORE_REPORTING_THRESHOLD` | Min score required to include coin in report   |

---

## 📊 Metrics Tracked (via Santiment)

- **Development Activity** – GitHub commit activity  
- **Daily Active Addresses** – Network usage metrics  
- **Sentiment Signals** – From media and social platforms  
- **Price & Volume** – Historical performance data

---

## 🤖 GPT-4o Intelligence

GPT-4o combines market, sentiment, and social signals to generate:
- Natural-language investment briefs
- Summarized outlooks
- Coin-specific recommendations

---

## ☁️ Deployment Notes

Deploy cheaply on AWS using:
- EC2 `t2.micro` instance (shutdown after 96h)
- Lambda + EventBridge for scheduling
- CloudFormation for VPC and IAM setup

> Runtime (1000 coins): ~20 hours  
> API Costs (monthly): ~$100 with paid tiers

---

## 🛠️ Contributing

PRs welcome!  
Fork → Improve → Submit a pull request 💪

---

## 📬 Contact

Open an [issue](https://github.com/sjmoran/crypto-panda/issues) with questions or feedback.

---

## ⚠️ Disclaimer

> **Not financial advice.**  
> Use this project at your own risk. Always do your own research and consider consulting a licensed advisor before making trading decisions.

---

## 📄 License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).  
See the [LICENSE](LICENSE) file for more.

---

## 🙏 Acknowledgments

- [CoinPaprika API](https://api.coinpaprika.com/)
- [Santiment API](https://santiment.net/)
- [OpenAI GPT-4o](https://openai.com/)
- [Fear and Greed Index](https://alternative.me/crypto/fear-and-greed-index/)
