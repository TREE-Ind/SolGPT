import asyncio
from utils import log_message

class MarketAnalysisAgent:
    def __init__(self, bot):
        self.bot = bot

    async def perform_analysis(self):
        log_message("MarketAnalysisAgent - Starting market analysis.", level='INFO')
        top_tokens = await self.bot.select_top_tokens()
        log_message(f"MarketAnalysisAgent - Selected top tokens: {top_tokens}", level='INFO')
        
        for symbol in top_tokens:
            token_symbol = symbol.split('/')[0]
            log_message(f"MarketAnalysisAgent - Analyzing {token_symbol}", level='DEBUG')
            reasoning, decision = await self.bot.advanced_reasoning_decision(token_symbol)
            log_message(f"MarketAnalysisAgent - Token: {token_symbol}, Decision: {decision.upper()}", level='INFO')
            self.bot.store_reasoning(token_symbol, reasoning, decision)
            await self.bot.notification_agent.notify_decision(token_symbol, decision, reasoning)
            if decision in ['buy', 'sell']:
                self.bot.trade_signals[symbol] = decision
                log_message(f"MarketAnalysisAgent - Generated trade signal: {decision.upper()} for {symbol}", level='INFO')
        
        log_message(f"MarketAnalysisAgent - Analysis complete. Generated {len(self.bot.trade_signals)} trade signals.", level='INFO')