# Solana AI Trading Bot with Raydium DEX Integration

---

## Overview

This repository contains a fully functional Solana AI Trading Bot that integrates with the Raydium DEX to perform automated trading based on advanced chain-of-thought reasoning using OpenAI's GPT-3/4 models. The bot is capable of:

- **Dynamic Token Selection**: Automatically selects top-performing tokens to trade.
- **Advanced Reasoning**: Uses AI to analyze technical indicators, news, and social media sentiment to make trading decisions.
- **Automated Trading**: Executes trades on the Raydium DEX.
- **Risk Management**: Implements stop-loss and take-profit mechanisms.
- **Logging and Alerts**: Provides detailed logs and sends alerts via email and Discord.
- **Automatic Pool and Token Mint Retrieval**: Automatically fetches all necessary pool and token mint addresses from Raydium's API.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Important Notes](#important-notes)
8. [Disclaimer](#disclaimer)

---

## Project Structure


---

## Features

- **Chain-of-Thought Reasoning**: Uses OpenAI's GPT models to analyze market data and generate trading decisions with reasoning.
- **Raydium DEX Integration**: Executes trades on the Raydium DEX using Solana smart contracts.
- **Automatic Pool and Token Mint Retrieval**: Automatically fetches all necessary pool information and token mint addresses from Raydium's API.
- **Technical Analysis**: Incorporates multiple technical indicators.
- **Sentiment Analysis**: Analyzes social media sentiment using Twitter data.
- **News Analysis**: Fetches and considers recent news articles related to tokens.
- **Risk Management**: Implements stop-loss and take-profit strategies.
- **Alerts**: Sends notifications via email and Discord.
- **Logging**: Logs all activities for monitoring and debugging.

---

## Prerequisites

- **Python 3.8 or higher**
- **Solana CLI** and a **Solana Wallet** (e.g., [Solflare](https://solflare.com/))
- **API Keys**:
  - **OpenAI API Key**
  - **News API Key** (e.g., [NewsAPI.org](https://newsapi.org/))
  - **Twitter API Keys**
  - **Discord Bot Token**
- **Basic Understanding** of Solana, Python, and cryptocurrency trading.

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/solana_ai_trading_bot.git
   cd solana_ai_trading_bot

## Install dependencies

```pip install -r requirements.txt```

## Create ENV file

```# .env

# Solana Wallet (Base64 encoded private key)
SOLANA_PRIVATE_KEY="your_base64_encoded_private_key"

# Email Alerts
ALERT_EMAIL="your_email@example.com"
ALERT_EMAIL_PASSWORD="your_email_password"
RECIPIENT_EMAIL="recipient_email@example.com"

# Discord Alerts
DISCORD_TOKEN="your_discord_bot_token"
DISCORD_CHANNEL_ID=123456789012345678  # Replace with your Discord channel ID

# OpenAI API Key
OPENAI_API_KEY="your_openai_api_key"

# News API Key (e.g., NewsAPI.org)
NEWS_API_KEY="your_news_api_key"

# Twitter API Keys for Sentiment Analysis
TWITTER_API_KEY="your_twitter_api_key"
TWITTER_API_SECRET_KEY="your_twitter_api_secret_key"
TWITTER_ACCESS_TOKEN="your_twitter_access_token"
TWITTER_ACCESS_TOKEN_SECRET="your_twitter_access_token_secret"

# Gradio UI Configuration
GRADIO_HOST="0.0.0.0"
GRADIO_PORT=7860

```

## Usage

```python trading_bot.py```

### Access the Gradio UI

    Open your web browser and navigate to http://localhost:7860 (or the host and port specified in your .env file).
    Use the UI to start, stop, and monitor the bot.

### Monitor Logs

    The bot logs its activities in trading_bot.log.
    Monitor this file to track the bot's performance and debug if necessary.

### Receive Alerts

    Ensure your email and Discord configurations are correct to receive alerts.
    The bot will send notifications for significant events.

## Important Notes

    Automatic Retrieval of Pool and Token Mint Addresses: The bot uses raydium_sdk.py to automatically fetch all necessary pool information and token mint addresses from Raydium's API. Users do not need to input any pool addresses or mint addresses manually.

    Configuration: Ensure that your .env file is properly configured with all necessary API keys and sensitive information.

    Testing: Thoroughly test the bot on Solana's Devnet before deploying to Mainnet.

    Security: Keep your private keys and API keys secure. Never share them or commit them to version control.

    Pool Information: The bot retrieves pool information from Raydium's public API (https://api.raydium.io/v2/sdk/ammPools). Ensure this endpoint is accessible and adjust if necessary.
