# data_functions.py

import requests
from dotenv import load_dotenv
import os
import tweepy
import logging
import re
from textblob import TextBlob
import aiohttp
from typing import List

load_dotenv()

NEWS_API_KEY = os.getenv('NEWS_API_KEY')

TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET_KEY = os.getenv('TWITTER_API_SECRET_KEY')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

def fetch_news_for_token(token_symbol):
    """
    Fetch recent news articles related to the token.
    """
    url = f'https://newsapi.org/v2/everything?q={token_symbol}&language=en&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    articles = response.json().get('articles', [])
    news_list = [article['title'] for article in articles[:5]]  # Get top 5 headlines
    return news_list

def analyze_sentiment(token_symbol):
    """
    Analyze social media sentiment for the token using Twitter data.
    """
    auth = tweepy.OAuth1UserHandler(
        TWITTER_API_KEY, TWITTER_API_SECRET_KEY,
        TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET
    )
    api = tweepy.API(auth)

    try:
        tweets = api.search_tweets(q=token_symbol, lang='en', count=100)
        tweet_texts = [re.sub(r'http\S+', '', tweet.text) for tweet in tweets]
        sentiment_scores = [TextBlob(text).sentiment.polarity for text in tweet_texts]
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        return average_sentiment
    except Exception as e:
        logging.error(f"Error fetching tweets: {e}")
        return 0

async def fetch_all_news(self, token_symbols: List[str]) -> dict:
    news_data = {}
    async with aiohttp.ClientSession() as session:
        tasks = [self.fetch_news_for_token(session, symbol) for symbol in token_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for symbol, result in zip(token_symbols, results):
            if isinstance(result, Exception):
                news_data[symbol] = "Error fetching news."
            else:
                news_data[symbol] = result
    return news_data

async def fetch_news_for_token(self, session: aiohttp.ClientSession, token_symbol: str) -> str:
    url = f'https://newsapi.org/v2/everything?q={token_symbol}&language=en&apiKey={NEWS_API_KEY}'
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            articles = data.get('articles', [])
            news_list = [article['title'] for article in articles[:5]]
            return "\n".join(news_list)
        else:
            raise Exception(f"Failed to fetch news for {token_symbol}, Status Code: {response.status}")
