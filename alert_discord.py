# alert_discord.py

import discord
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

class DiscordAlert:
    def __init__(self):
        intents = discord.Intents.default()
        self.client = discord.Client(intents=intents)
        self.channel_id = int(os.getenv("DISCORD_CHANNEL_ID"))
        self.ready_event = asyncio.Event()

        @self.client.event
        async def on_ready():
            print(f'Logged in as {self.client.user}')
            self.ready_event.set()

    async def start(self):
        DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
        if not DISCORD_TOKEN:
            raise ValueError("DISCORD_TOKEN not set in .env")
        await self.client.start(DISCORD_TOKEN)

    async def send_message(self, message):
        await self.ready_event.wait()  # Ensure the client is ready
        channel = self.client.get_channel(self.channel_id)
        if channel:
            await channel.send(message)
        else:
            print("Channel not found.")

    async def run_bot(self):
        try:
            await self.start()
        except asyncio.CancelledError:
            await self.client.close()
            print("Discord client closed.")
