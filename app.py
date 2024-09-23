# app.py

import gradio as gr
from trading_bot import bot
import asyncio
import threading
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def start_bot():
    asyncio.run(bot.start())
    return "Bot started."

def stop_bot():
    asyncio.run(bot.stop())
    return "Bot stopped."

def get_bot_status():
    status = "Running" if bot.running else "Stopped"
    return status

def get_positions():
    positions = bot.current_positions
    if positions:
        df = pd.DataFrame.from_dict(positions, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Token'}, inplace=True)
        return df
    else:
        return pd.DataFrame(columns=['Token', 'amount', 'purchase_price'])

def get_recent_trades():
    trades = bot.recent_trades
    if trades:
        df = pd.DataFrame(trades)
        return df
    else:
        return pd.DataFrame(columns=['timestamp', 'token', 'action', 'amount', 'price'])

def get_performance_metrics():
    metrics = bot.performance_metrics
    if metrics:
        total_profit = metrics.get('total_profit', 0)
        trades = metrics.get('trades', 0)
        wins = metrics.get('wins', 0)
        losses = metrics.get('losses', 0)
        win_rate = (wins / trades) * 100 if trades > 0 else 0
        return f"Total Profit: {total_profit:.2f} USDT\nTrades: {trades}\nWins: {wins}\nLosses: {losses}\nWin Rate: {win_rate:.2f}%"
    else:
        return "No performance metrics available."

def get_logs():
    try:
        with open('trading_bot.log', 'r') as f:
            logs = f.read()
        return logs
    except FileNotFoundError:
        return "Log file not found."

def get_price_chart(token_symbol):
    df = asyncio.run(bot.fetch_data(f"{token_symbol}/USDT"))
    if df.empty:
        return "No data available."
    fig = px.line(df, x='datetime', y='close', title=f'{token_symbol}/USDT Price')
    return fig

def get_technical_chart(token_symbol):
    df = asyncio.run(bot.fetch_data(f"{token_symbol}/USDT"))
    if df.empty:
        return "No data available."
    df = bot.add_indicators(df)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['datetime'], open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name='Price'))
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['ma50'], line=dict(color='blue', width=1), name='MA50'))
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['ma200'], line=dict(color='red', width=1), name='MA200'))
    fig.update_layout(title=f'{token_symbol}/USDT Technical Indicators')
    return fig

def get_news_sentiment(token_symbol):
    news_events = bot.fetch_news_for_token(token_symbol)
    social_sentiment = bot.analyze_sentiment(token_symbol)
    return f"News Headlines:\n{news_events}\n\nSocial Sentiment Score: {social_sentiment}"

def get_balance():
    sol_balance = asyncio.run(get_balance(solana_client, bot.wallet.pubkey()))
    return f"SOL Balance: {sol_balance:.5f}"

with gr.Blocks() as demo:
    gr.Markdown("# Solana AI Trading Bot Dashboard")

    with gr.Tab("Control Panel"):
        gr.Markdown("## Bot Control")
        with gr.Row():
            start_button = gr.Button("Start Bot")
            stop_button = gr.Button("Stop Bot")
            status_text = gr.Textbox(label="Bot Status", value=get_bot_status(), interactive=False)
        start_button.click(start_bot, outputs=status_text)
        stop_button.click(stop_bot, outputs=status_text)
        balance_text = gr.Textbox(label="Wallet Balance", value=get_balance(), interactive=False)
        refresh_balance_button = gr.Button("Refresh Balance")
        refresh_balance_button.click(fn=get_balance, inputs=None, outputs=balance_text)

    with gr.Tab("Current Positions"):
        gr.Markdown("## Current Positions")
        positions_df = gr.DataFrame(value=get_positions(), headers="keys")

    with gr.Tab("Recent Trades"):
        gr.Markdown("## Recent Trades")
        trades_df = gr.DataFrame(value=get_recent_trades(), headers="keys")

    with gr.Tab("Performance Metrics"):
        gr.Markdown("## Performance Metrics")
        performance_text = gr.Textbox(value=get_performance_metrics(), interactive=False)

    with gr.Tab("Price Charts"):
        gr.Markdown("## Token Price Chart")
        token_input = gr.Textbox(label="Token Symbol", placeholder="Enter token symbol (e.g., SOL)")
        price_chart = gr.Plot()
        token_input.submit(get_price_chart, inputs=token_input, outputs=price_chart)

    with gr.Tab("Technical Analysis"):
        gr.Markdown("## Technical Indicators")
        tech_token_input = gr.Textbox(label="Token Symbol", placeholder="Enter token symbol (e.g., SOL)")
        technical_chart = gr.Plot()
        tech_token_input.submit(get_technical_chart, inputs=tech_token_input, outputs=technical_chart)

    with gr.Tab("News and Sentiment"):
        gr.Markdown("## News and Sentiment Analysis")
        news_token_input = gr.Textbox(label="Token Symbol", placeholder="Enter token symbol (e.g., SOL)")
        news_sentiment_text = gr.Textbox()
        news_token_input.submit(get_news_sentiment, inputs=news_token_input, outputs=news_sentiment_text)

    with gr.Tab("Logs"):
        gr.Markdown("## Bot Logs")
        logs_text = gr.Textbox(value=get_logs(), lines=15, interactive=False)
        refresh_logs_button = gr.Button("Refresh Logs")
        refresh_logs_button.click(fn=get_logs, inputs=None, outputs=logs_text)

    # Auto-refresh status and positions
    def refresh_status():
        status = get_bot_status()
        positions = get_positions()
        balance = get_balance()
        return status, positions, balance

    demo.load(refresh_status, inputs=None, outputs=[status_text, positions_df, balance_text], every=10)

# Run the Gradio app
if __name__ == "__main__":
    threading.Thread(target=demo.launch, kwargs={
        "server_name": os.getenv("GRADIO_HOST", "0.0.0.0"),
        "server_port": int(os.getenv("GRADIO_PORT", "7860")),
        "share": False
    }).start()
    # Keep the main thread alive to allow the bot to run
    asyncio.get_event_loop().run_forever()
