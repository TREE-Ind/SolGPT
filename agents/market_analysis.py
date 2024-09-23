import asyncio
from utils import log_message

class MarketAnalysisAgent:
    def __init__(self, bot):
        self.bot = bot

    async def perform_analysis(self):
        log_message("MarketAnalysisAgent - Starting market analysis.", level='INFO')
        top_tokens = await self.bot.select_top_tokens()
        for symbol in top_tokens:
            token_symbol = symbol.split('/')[0]
            reasoning, decision = await self.bot.advanced_reasoning_decision(token_symbol)
            log_message(f"MarketAnalysisAgent - Token: {token_symbol}, Decision: {decision.upper()}", level='DEBUG')
            self.bot.store_reasoning(token_symbol, reasoning, decision)
            await self.bot.notification_agent.notify_decision(token_symbol, decision, reasoning)
            if decision in ['buy', 'sell']:
                self.bot.trade_signals[symbol] = decision
        log_message("MarketAnalysisAgent - Market analysis completed.", level='INFO')