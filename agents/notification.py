import asyncio
from utils import log_message

class NotificationAgent:
    def __init__(self, bot):
        self.bot = bot

    async def notify_decision(self, token_symbol, decision, reasoning):
        message = f"Decision: **{decision.upper()}** {token_symbol}\nReasoning: {reasoning}"
        log_message(f"NotificationAgent - Sending notification for {token_symbol}: {decision.upper()}", level='INFO')
        if self.bot.discord_alert:
            await self.bot.discord_alert.send_message(message)
            log_message(f"NotificationAgent - Sent Discord message for {token_symbol}.", level='DEBUG')
        send_email(
            subject=f"Trading Bot Alert: {decision.upper()} {token_symbol}",
            body=message
        )
        log_message(f"NotificationAgent - Sent email for {token_symbol}.", level='DEBUG')