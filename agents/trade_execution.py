import asyncio
from utils import log_message

class TradeExecutionAgent:
    def __init__(self, bot):
        self.bot = bot

    async def execute_trades(self):
        log_message("TradeExecutionAgent - Starting trade execution.", level='INFO')
        if not self.bot.trade_signals:
            log_message("TradeExecutionAgent - No trade signals to execute.", level='INFO')
            return

        for symbol, decision in list(self.bot.trade_signals.items()):
            log_message(f"TradeExecutionAgent - Processing signal: {decision.upper()} for {symbol}", level='DEBUG')
            if decision == 'buy' and symbol not in self.bot.current_positions:
                log_message(f"TradeExecutionAgent - Initiating BUY for {symbol}.", level='INFO')
                await self.bot.make_trade(symbol, 'buy')
            elif decision == 'sell' and symbol in self.bot.current_positions:
                log_message(f"TradeExecutionAgent - Initiating SELL for {symbol}.", level='INFO')
                await self.bot.make_trade(symbol, 'sell')
            else:
                log_message(f"TradeExecutionAgent - Skipping {decision.upper()} for {symbol}. Condition not met.", level='INFO')
            # Remove the signal after processing
            del self.bot.trade_signals[symbol]
        
        log_message("TradeExecutionAgent - Trade execution completed.", level='INFO')