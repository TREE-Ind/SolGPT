import asyncio
from utils import log_message

class TradeExecutionAgent:
    def __init__(self, bot):
        self.bot = bot

    async def execute_trades(self):
        log_message("TradeExecutionAgent - Executing trade signals.", level='INFO')
        for symbol, decision in list(self.bot.trade_signals.items()):
            if decision == 'buy' and symbol not in self.bot.current_positions:
                log_message(f"TradeExecutionAgent - Initiating BUY for {symbol}.", level='DEBUG')
                await self.bot.make_trade(symbol, 'buy')
            elif decision == 'sell' and symbol in self.bot.current_positions:
                log_message(f"TradeExecutionAgent - Initiating SELL for {symbol}.", level='DEBUG')
                await self.bot.make_trade(symbol, 'sell')
            # Remove the signal after processing
            del self.bot.trade_signals[symbol]
        log_message("TradeExecutionAgent - Trade execution completed.", level='INFO')