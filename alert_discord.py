# alert_discord.py

import discord
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")

class DiscordAlert(discord.Client):
    def __init__(self, channel_id, enabled=True, **kwargs):
        intents = discord.Intents.default()
        intents.messages = True
        super().__init__(intents=intents, **kwargs)
        self.channel_id = channel_id
        self.message_queue = asyncio.Queue()
        self.bg_task = None
        self.enabled = enabled

    async def setup_hook(self):
        if self.enabled:
            # Start the background task to process messages
            self.bg_task = self.loop.create_task(self.process_messages())

    async def on_ready(self):
        if self.enabled:
            print(f'Logged in as {self.user} (ID: {self.user.id})')
            channel = self.get_channel(self.channel_id)
            if channel is None:
                print(f"Channel ID {self.channel_id} not found.")
                self.enabled = False  # Disable if channel not found
            else:
                print(f"Connected to channel: {channel.name}")

    async def process_messages(self):
        await self.wait_until_ready()
        channel = self.get_channel(self.channel_id)
        if channel is None:
            print(f"Channel ID {self.channel_id} not found.")
            return
        while not self.is_closed():
            message = await self.message_queue.get()
            try:
                await channel.send(message)
                print(f"Sent message to Discord: {message}")
            except Exception as e:
                print(f"Failed to send message to Discord: {e}")
            self.message_queue.task_done()

    async def send_message(self, message):
        if self.enabled:
            await self.message_queue.put(message)

    async def close_client(self):
        if self.enabled:
            await self.close()
