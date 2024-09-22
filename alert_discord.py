# alert_discord.py

import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
import logging

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))

class DiscordAlert:
    def __init__(self):
        intents = discord.Intents.default()
        self.client = commands.Bot(command_prefix='!', intents=intents)
        self.channel_id = DISCORD_CHANNEL_ID

        @self.client.event
        async def on_ready():
            logging.info(f'Discord bot connected as {self.client.user}')

    async def send_message(self, message):
        channel = self.client.get_channel(self.channel_id)
        if channel:
            await channel.send(message)
            logging.info(f"Discord message sent: {message}")
        else:
            logging.error("Discord channel not found.")

    def run_bot(self):
        self.client.loop.create_task(self.client.start(DISCORD_TOKEN))
