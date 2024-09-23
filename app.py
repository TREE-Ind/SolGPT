# app.py

import gradio as gr
from trading_bot import bot
import asyncio
import threading
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Function to start the trading bot
async def initialize_bot():
    await bot.start()

# Start the bot in a separate thread to avoid blocking Gradio
def start_bot():
    asyncio.run(initialize_bot())

# Function to stop the bot gracefully
async def stop_bot_async():
    await bot.stop()

def stop_bot():
    asyncio.run(stop_bot_async())
    return "Bot stopped."

def get_bot_status():
    status = "🟢 Running" if bot.running else "🔴 Stopped"
    return status

async def get_balance():
    try:
        sol_balance = await get_balance_func()
        return f"**SOL Balance:** {sol_balance:.5f} SOL"
    except Exception as e:
        return f"Error fetching balance: {e}"

async def get_balance_func():
    from solana_connection import get_balance
    return await get_balance(bot.solana_client, bot.wallet.pubkey())

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
        return f"**Total Profit:** {total_profit:.2f} USDT\n**Trades:** {trades}\n**Wins:** {wins}\n**Losses:** {losses}\n**Win Rate:** {win_rate:.2f}%"
    else:
        return "No performance metrics available."

def get_logs():
    try:
        with open('trading_bot.log', 'r') as f:
            logs = f.read()
        return logs
    except FileNotFoundError:
        return "Log file not found."

async def get_price_chart(token_symbol):
    try:
        df = await bot.fetch_data(f"{token_symbol}/USDT")
        if df.empty:
            return "No data available."
        fig = px.line(df, x='datetime', y='close', title=f'{token_symbol}/USDT Price')
        fig.update_layout(xaxis_title='Time', yaxis_title='Price (USDT)')
        return fig
    except Exception as e:
        return f"Error generating price chart: {e}"

async def get_technical_chart(token_symbol):
    try:
        df = await bot.fetch_data(f"{token_symbol}/USDT")
        if df.empty:
            return "No data available."
        df = bot.add_indicators(df)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['datetime'], open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name='Price'))
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['ma50'], line=dict(color='blue', width=1), name='MA50'))
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['ma200'], line=dict(color='red', width=1), name='MA200'))
        fig.update_layout(title=f'{token_symbol}/USDT Technical Indicators',
                          xaxis_title='Time',
                          yaxis_title='Price (USDT)')
        return fig
    except Exception as e:
        return f"Error generating technical chart: {e}"

async def get_news_sentiment(token_symbol):
    try:
        news_events = bot.fetch_news_for_token(token_symbol)
        social_sentiment = bot.analyze_sentiment(token_symbol)
        return f"**News Headlines:**\n{news_events}\n\n**Social Sentiment Score:** {social_sentiment}"
    except Exception as e:
        return f"Error fetching news and sentiment: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# 🪙 Solana AI Trading Bot Dashboard 🪙")
    
    with gr.Tab("Control Panel"):
        gr.Markdown("## ⚙️ Bot Control")
        with gr.Row():
            start_button = gr.Button("▶️ Start Bot")
            stop_button = gr.Button("⏹️ Stop Bot")
            status_text = gr.Textbox(label="🔍 Bot Status", value=get_bot_status(), interactive=False)
        with gr.Row():
            balance_text = gr.Markdown(await get_balance())
            refresh_balance_button = gr.Button("🔄 Refresh Balance")
        start_button.click(start_bot, outputs=status_text)
        stop_button.click(stop_bot, outputs=status_text)
        refresh_balance_button.click(fn=get_balance, inputs=None, outputs=balance_text)
    
    with gr.Tab("📈 Current Positions"):
        gr.Markdown("## 📊 Current Positions")
        positions_df = gr.DataFrame(value=get_positions(), interactive=False)
    
    with gr.Tab("💼 Recent Trades"):
        gr.Markdown("## 🔄 Recent Trades")
        trades_df = gr.DataFrame(value=get_recent_trades(), interactive=False)
    
    with gr.Tab("📊 Performance Metrics"):
        gr.Markdown("## 📈 Performance Metrics")
        performance_text = gr.Markdown(get_performance_metrics())
    
    with gr.Tab("📉 Price Charts"):
        gr.Markdown("## 📈 Token Price Chart")
        with gr.Row():
            token_input = gr.Textbox(label="🔠 Token Symbol", placeholder="Enter token symbol (e.g., SOL)", value="SOL")
            generate_price_chart = gr.Button("Generate Chart")
        price_chart = gr.Plot()
        generate_price_chart.click(get_price_chart, inputs=token_input, outputs=price_chart)
    
    with gr.Tab("📊 Technical Analysis"):
        gr.Markdown("## 📉 Technical Indicators")
        with gr.Row():
            tech_token_input = gr.Textbox(label="🔠 Token Symbol", placeholder="Enter token symbol (e.g., SOL)", value="SOL")
            generate_tech_chart = gr.Button("Generate Chart")
        technical_chart = gr.Plot()
        generate_tech_chart.click(get_technical_chart, inputs=tech_token_input, outputs=technical_chart)
    
    with gr.Tab("📰 News and Sentiment"):
        gr.Markdown("## 📰 News and Sentiment Analysis")
        with gr.Row():
            news_token_input = gr.Textbox(label="🔠 Token Symbol", placeholder="Enter token symbol (e.g., SOL)", value="SOL")
            generate_news_sentiment = gr.Button("Get News & Sentiment")
        news_sentiment_text = gr.Markdown()
        generate_news_sentiment.click(get_news_sentiment, inputs=news_token_input, outputs=news_sentiment_text)
    
    with gr.Tab("📝 Logs"):
        gr.Markdown("## 📜 Bot Logs")
        logs_text = gr.Textbox(value=get_logs(), lines=15, interactive=False)
        refresh_logs_button = gr.Button("🔄 Refresh Logs")
        refresh_logs_button.click(fn=get_logs, inputs=None, outputs=logs_text)
    
    # Auto-refresh status, positions, and balance every 10 seconds
    async def refresh_status_positions_balance():
        status = get_bot_status()
        positions = get_positions()
        balance = await get_balance()
        return status, positions, balance
    
    demo.load(refresh_status_positions_balance, inputs=None, outputs=[status_text, positions_df, balance_text], every=10)

# Run the Gradio app
if __name__ == "__main__":
    # Start the trading bot in a separate thread
    threading.Thread(target=start_bot, daemon=True).start()
    
    # Launch the Gradio interface
    demo.launch(server_name=os.getenv("GRADIO_HOST", "0.0.0.0"),
                server_port=int(os.getenv("GRADIO_PORT", "7860")),
                share=False)
    
    # Remove the conflicting asyncio loop
    # Keep the main thread alive by letting Gradio handle the event loop
    # No need for asyncio.get_event_loop().run_forever()

