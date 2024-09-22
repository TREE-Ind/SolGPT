# utils.py

import pandas as pd
import ta

def add_indicators(df):
    # Moving Averages
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma200'] = df['close'].rolling(window=200).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Technical Indicators
    df['rsi14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    bb_indicator = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bb_indicator.bollinger_hband()
    df['bb_low'] = bb_indicator.bollinger_lband()

    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    stochastic = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['stoch_k'] = stochastic.stoch()
    df['stoch_d'] = stochastic.stoch_signal()

    return df
