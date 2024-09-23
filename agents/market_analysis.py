import asyncio
import logging

class MarketAnalysisAgent:
    def __init__(self, bot):
        self.bot = bot

    async def perform_analysis(self):
        top_tokens = await self.bot.select_top_tokens()
        for symbol in top_tokens:
            token_symbol = symbol.split('/')[0]
            reasoning, decision = await self.bot.advanced_reasoning_decision(token_symbol)
            logging.info(f"MarketAnalysisAgent - Token: {token_symbol}, Decision: {decision.upper()}, Reasoning: {reasoning}")
            self.bot.store_reasoning(token_symbol, reasoning, decision)
            await self.bot.notification_agent.notify_decision(token_symbol, decision, reasoning)
            if decision in ['buy', 'sell']:
                self.bot.trade_signals[symbol] = decision
        logging.info("MarketAnalysisAgent - Analysis complete.")