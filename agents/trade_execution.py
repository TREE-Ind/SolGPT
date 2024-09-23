import asyncio
import logging

class TradeExecutionAgent:
    def __init__(self, bot):
        self.bot = bot

    async def execute_trades(self):
        for symbol, decision in self.bot.trade_signals.items():
            if decision == 'buy' and symbol not in self.bot.current_positions:
                await self.bot.make_trade(symbol, 'buy')
            elif decision == 'sell' and symbol in self.bot.current_positions:
                await self.bot.make_trade(symbol, 'sell')
            # Remove the signal after processing
            del self.bot.trade_signals[symbol]
        logging.info("TradeExecutionAgent - Trades executed.")