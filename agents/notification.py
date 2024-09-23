import asyncio

class NotificationAgent:
    def __init__(self, bot):
        self.bot = bot

    async def notify_decision(self, token_symbol, decision, reasoning):
        message = f"Decision: **{decision.upper()}** {token_symbol}\nReasoning: {reasoning}"
        if self.bot.discord_alert:
            await self.bot.discord_alert.send_message(message)
        send_email(
            subject=f"Trading Bot Alert: {decision.upper()} {token_symbol}",
            body=message
        )