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

async def update_initial_balance():
    return await get_balance()

async def get_ai_reasoning():
    reasoning_history = bot.get_reasoning_history()
    if reasoning_history:
        df = pd.DataFrame(reasoning_history)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        return df
    else:
        return pd.DataFrame(columns=['timestamp', 'token', 'reasoning', 'decision'])

async def get_discovered_tokens():
    return pd.DataFrame(list(bot.discovered_tokens), columns=['Token'])

async def get_top_tokens():
    top_tokens = await bot.select_top_tokens()
    return pd.DataFrame(top_tokens, columns=['Token'])

async def refresh_status_positions_balance_reasoning():
    status = get_bot_status()
    positions = get_positions()
    balance = await get_balance()
    reasoning = await get_ai_reasoning()
    return status, positions, balance, reasoning

async def refresh_discovered_tokens_async():
    return await get_discovered_tokens()

async def refresh_top_tokens_async():
    return await get_top_tokens()

async def refresh_reasoning_async():
    return await get_ai_reasoning()

def clear_logs():
    """
    Clears the contents of the trading_bot.log file.
    """
    try:
        open('trading_bot.log', 'w').close()
        log_message("Log file cleared by user.", level='INFO')
        return "Log file has been cleared."
    except Exception as e:
        log_message(f"Error clearing log file: {e}", level='ERROR')
        return f"Error clearing log file: {e}"

def get_latest_logs(n=100):
    """
    Retrieves the latest n lines from the trading_bot.log file.
    """
    try:
        with open('trading_bot.log', 'r') as f:
            lines = f.readlines()
            latest_logs = ''.join(lines[-n:])  # Get the last n lines
            return latest_logs
    except FileNotFoundError:
        return "Log file not found."
    except Exception as e:
        return f"Error reading log file: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# 🪙 Solana AI Trading Bot Dashboard 🪙")
    
    with gr.Tab("Control Panel"):
        gr.Markdown("## ⚙️ Bot Control")
        with gr.Row():
            start_button = gr.Button("▶️ Start Bot")
            stop_button = gr.Button("⏹️ Stop Bot")
            status_text = gr.Textbox(label="🔍 Bot Status", value=get_bot_status(), interactive=False)
        with gr.Row():
            balance_text = gr.Markdown("Loading balance...")
            refresh_balance_button = gr.Button("🔄 Refresh Balance")
        start_button.click(start_bot, outputs=status_text)
        stop_button.click(stop_bot, outputs=status_text)
        refresh_balance_button.click(get_balance, inputs=None, outputs=balance_text)
    
    with gr.Tab("Bot Steps"):
        gr.Markdown("## 🔍 Bot Step-by-Step Process")
        
        with gr.Accordion("Step 1: Token Discovery"):
            gr.Markdown("### 🔎 Newly Discovered Tokens")
            discovered_tokens_df = gr.DataFrame(interactive=False)
            refresh_discovered_tokens = gr.Button("🔄 Refresh Discovered Tokens")
        
        with gr.Accordion("Step 2: Top Token Selection"):
            gr.Markdown("### 🏆 Top Selected Tokens")
            top_tokens_df = gr.DataFrame(interactive=False)
            refresh_top_tokens = gr.Button("🔄 Refresh Top Tokens")
        
        with gr.Accordion("Step 3: AI Analysis and Decision"):
            gr.Markdown("### 🧠 AI Reasoning and Decisions")
            reasoning_df = gr.DataFrame(interactive=False)
            refresh_reasoning_button = gr.Button("🔄 Refresh AI Reasoning")
        
        with gr.Accordion("Step 4: Trade Execution"):
            gr.Markdown("### 💼 Recent Trades")
            trades_df = gr.DataFrame(value=get_recent_trades(), interactive=False)
            refresh_trades_button = gr.Button("🔄 Refresh Recent Trades")
    
    with gr.Tab("Portfolio"):
        gr.Markdown("## 📊 Current Portfolio")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Current Positions")
                positions_df = gr.DataFrame(value=get_positions(), interactive=False)
            with gr.Column():
                gr.Markdown("### 📊 Performance Metrics")
                performance_text = gr.Markdown(get_performance_metrics())
        refresh_portfolio_button = gr.Button("🔄 Refresh Portfolio")
    
    with gr.Tab("Market Analysis"):
        gr.Markdown("## 📊 Market Analysis Tools")
        
        with gr.Accordion("Price Charts"):
            gr.Markdown("### 📈 Token Price Chart")
            with gr.Row():
                token_input = gr.Textbox(label="🔠 Token Symbol", placeholder="Enter token symbol (e.g., SOL)", value="SOL")
                generate_price_chart = gr.Button("Generate Chart")
            price_chart = gr.Plot()
        
        with gr.Accordion("Technical Analysis"):
            gr.Markdown("### 📉 Technical Indicators")
            with gr.Row():
                tech_token_input = gr.Textbox(label="🔠 Token Symbol", placeholder="Enter token symbol (e.g., SOL)", value="SOL")
                generate_tech_chart = gr.Button("Generate Chart")
            technical_chart = gr.Plot()
        
        with gr.Accordion("News and Sentiment"):
            gr.Markdown("### 📰 News and Sentiment Analysis")
            with gr.Row():
                news_token_input = gr.Textbox(label="🔠 Token Symbol", placeholder="Enter token symbol (e.g., SOL)", value="SOL")
                generate_news_sentiment = gr.Button("Get News & Sentiment")
            news_sentiment_text = gr.Markdown()
    
    with gr.Tab("Logs"):
        gr.Markdown("## 📝 Bot Logs")
        logs_text = gr.Textbox(value=get_latest_logs(), lines=15, interactive=False)
        with gr.Row():
            refresh_logs_button = gr.Button("🔄 Refresh Logs")
            clear_logs_button = gr.Button("🗑️ Clear Logs")
    
    # Set up refresh functions
    refresh_discovered_tokens.click(
        lambda: asyncio.run(refresh_discovered_tokens_async()),
        outputs=discovered_tokens_df
    )
    refresh_top_tokens.click(
        lambda: asyncio.run(refresh_top_tokens_async()),
        outputs=top_tokens_df
    )
    refresh_reasoning_button.click(
        lambda: asyncio.run(refresh_reasoning_async()),
        outputs=reasoning_df
    )
    refresh_trades_button.click(get_recent_trades, outputs=trades_df)
    refresh_portfolio_button.click(
        lambda: (get_positions(), get_performance_metrics()),
        outputs=[positions_df, performance_text]
    )
    refresh_logs_button.click(get_latest_logs, outputs=logs_text)
    clear_logs_button.click(clear_logs, outputs=logs_text)
    
    generate_price_chart.click(get_price_chart, inputs=token_input, outputs=price_chart)
    generate_tech_chart.click(get_technical_chart, inputs=tech_token_input, outputs=technical_chart)
    generate_news_sentiment.click(get_news_sentiment, inputs=news_token_input, outputs=news_sentiment_text)

    # Auto-refresh every 30 seconds
    demo.load(
        refresh_status_positions_balance_reasoning,
        outputs=[status_text, positions_df, balance_text, reasoning_df],
        every=30
    )

    # Initial balance update
    demo.load(update_initial_balance, outputs=balance_text)

    # Auto-refresh logs every 10 seconds
    demo.load(
        get_latest_logs,
        outputs=logs_text,
        every=10
    )

if __name__ == "__main__":
    threading.Thread(target=start_bot, daemon=True).start()
    demo.launch(
        server_name=os.getenv("GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=False
   )

