# app.py

import gradio as gr
from trading_bot import bot
import asyncio
import threading

def start_bot():
    loop = asyncio.get_event_loop()
    loop.create_task(bot.start())
    return "Bot started."

def stop_bot():
    loop = asyncio.get_event_loop()
    loop.create_task(bot.stop())
    return "Bot stopped."

def get_bot_status():
    status = "Running" if bot.running else "Stopped"
    return status

def get_positions():
    positions = bot.current_positions
    if positions:
        return str(positions)
    else:
        return "No current positions."

def get_logs():
    try:
        with open('trading_bot.log', 'r') as f:
            logs = f.read()
        return logs
    except FileNotFoundError:
        return "Log file not found."

with gr.Blocks() as demo:
    gr.Markdown("# Solana AI Trading Bot")
    with gr.Row():
        start_button = gr.Button("Start Bot")
        stop_button = gr.Button("Stop Bot")
    status_text = gr.Textbox(label="Bot Status", value=get_bot_status(), interactive=False)
    positions_text = gr.Textbox(label="Current Positions", value=get_positions(), interactive=False)
    logs_text = gr.Textbox(label="Logs", value=get_logs(), lines=10, interactive=False)

    start_button.click(start_bot, outputs=status_text)
    stop_button.click(stop_bot, outputs=status_text)

    refresh_button = gr.Button("Refresh Status")
    refresh_button.click(fn=get_bot_status, inputs=None, outputs=status_text)
    refresh_button.click(fn=get_positions, inputs=None, outputs=positions_text)
    refresh_button.click(fn=get_logs, inputs=None, outputs=logs_text)

# Run the Gradio app
if __name__ == "__main__":
    threading.Thread(target=demo.launch, kwargs={"server_name": os.getenv("GRADIO_HOST", "0.0.0.0"),
                                                 "server_port": int(os.getenv("GRADIO_PORT", "7860")),
                                                 "share": False}).start()
    # Keep the main thread alive to allow the bot to run
    asyncio.get_event_loop().run_forever()
