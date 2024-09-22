# trading_bot.py

import asyncio
import ccxt
import pandas as pd
import logging
import os
import openai
from dotenv import load_dotenv
from utils import add_indicators
from alert import send_email
##from alert_discord import DiscordAlert
from solana_connection import load_wallet, get_balance
from data_functions import fetch_news_for_token, analyze_sentiment
from raydium_sdk import get_raydium_pools, find_pool_by_tokens, get_token_mints
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solana.transaction import Transaction
from solders.instruction import Instruction, AccountMeta
from solana.rpc.types import TxOpts
from solders.system_program import ID as SYS_PROGRAM_ID
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.async_client import AsyncToken
from spl.token.instructions import get_associated_token_address
import struct

# Configure logging
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Solana client
solana_client = AsyncClient("https://api.mainnet-beta.solana.com")

# Initialize CCXT for data fetching
exchange = ccxt.binance()

class TradingBot:
    def __init__(self):
        # Initialize variables
        self.running = False
        self.wallet = load_wallet()
        self.timeframe = '1h'
        self.limit = 500
        self.current_positions = {}
        self.stop_loss_pct = 0.95
        self.take_profit_pct = 1.05
        self.top_n_tokens = 5
        self.token_selection_interval = 3600
        #self.discord_alert = DiscordAlert()
        #self.discord_alert.run_bot()
        self.token_mints = get_token_mints()
        self.pools = get_raydium_pools()
        self.loop = asyncio.get_event_loop()
        self.task = None

    async def start(self):
        if not self.running:
            self.running = True
            self.task = self.loop.create_task(self.run())
            logging.info("Trading bot started.")

    async def stop(self):
        if self.running:
            self.running = False
            if self.task:
                self.task.cancel()
            logging.info("Trading bot stopped.")

    async def run(self):
        while self.running:
            try:
                top_tokens = await self.select_top_tokens()
                for symbol in top_tokens:
                    token_symbol = symbol.split('/')[0]
                    reasoning, decision = await self.advanced_reasoning_decision(token_symbol)
                    logging.info(f"Token: {token_symbol}, Decision: {decision.upper()}, Reasoning: {reasoning}")

                    if decision == 'buy' and token_symbol not in self.current_positions:
                        await self.make_trade(symbol, 'buy')
                    elif decision == 'sell' and token_symbol in self.current_positions:
                        await self.make_trade(symbol, 'sell')
                    else:
                        logging.info(f"No trade action for {token_symbol}")

                # Wait before next cycle
                await asyncio.sleep(self.token_selection_interval)
            except asyncio.CancelledError:
                logging.info("Trading bot task cancelled.")
                break
            except Exception as e:
                logging.error(f"Error in bot run loop: {e}")
                send_email(
                    subject="Trading Bot Alert: Error",
                    body=f"An error occurred: {e}"
                )
                await asyncio.sleep(60)

    async def fetch_data(self, symbol):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    async def select_top_tokens(self):
        markets = exchange.load_markets()
        tokens = [symbol for symbol in markets if '/USDT' in symbol and symbol != 'SOL/USDT']

        token_scores = {}
        for token in tokens:
            try:
                df = await self.fetch_data(token)
                if df.empty:
                    continue
                df = add_indicators(df)
                df.dropna(inplace=True)
                latest_close = df['close'].iloc[-1]
                latest_volume = df['volume'].iloc[-1]
                rsi = df['rsi14'].iloc[-1]
                volume = latest_volume
                score = (volume / df['volume'].mean()) * (1 if 30 < rsi < 70 else 0.5)
                token_scores[token] = score
            except Exception as e:
                logging.error(f"Error fetching data for {token}: {e}")

        sorted_tokens = sorted(token_scores, key=token_scores.get, reverse=True)
        top_tokens = sorted_tokens[:self.top_n_tokens]
        logging.info(f"Selected top tokens: {top_tokens}")
        return top_tokens

    async def advanced_reasoning_decision(self, token_symbol):
        technical_data = await self.get_technical_data(f"{token_symbol}/USDT")
        news_events = fetch_news_for_token(token_symbol)
        social_sentiment = analyze_sentiment(token_symbol)

        # Craft prompt
        prompt = f"""
You are an expert financial analyst. Based on the following data:

- Technical Indicators: {technical_data}
- Recent News Headlines: {news_events}
- Social Media Sentiment Score: {social_sentiment}

Provide a detailed analysis of the potential price movement of {token_symbol} in the next hour. Include your reasoning and conclude with a recommendation to BUY, SELL, or HOLD.
"""

        # Get response from LLM
        try:
            response = openai.Completion.create(
                engine='gpt-4o-mini',
                prompt=prompt,
                max_tokens=2048,
                temperature=0.7,
                n=1,
                stop=None,
            )

            reasoning = response.choices[0].text.strip()

            # Extract decision
            decision = None
            if 'BUY' in reasoning.upper():
                decision = 'buy'
            elif 'SELL' in reasoning.upper():
                decision = 'sell'
            elif 'HOLD' in reasoning.upper():
                decision = 'hold'
            else:
                decision = 'hold'  # Default to hold if no clear decision

            return reasoning, decision
        except Exception as e:
            logging.error(f"Error with OpenAI API: {e}")
            return "", "hold"

    async def get_technical_data(self, symbol):
        df = await self.fetch_data(symbol)
        if df.empty:
            return {}
        df = add_indicators(df)
        latest_data = df.iloc[-1]
        technical_summary = {
            'Price': latest_data['close'],
            'RSI': latest_data['rsi14'],
            'MACD': latest_data['macd'],
            'Volume': latest_data['volume'],
            'MA50': latest_data['ma50'],
            'MA200': latest_data['ma200']
        }
        return technical_summary

    async def perform_swap(self, from_token_symbol, to_token_symbol, amount_in):
        """
        Performs a token swap on Raydium.
        """
        # Raydium program IDs and accounts
        RAYDIUM_SWAP_PROGRAM_ID = Pubkey.from_string('rvkYEt3Qp6eC3Ndy6EMrxJD9F6xZjQ9S8dHEs7hY3vv')  # Raydium Swap Program ID

        # Token mints
        from_token_mint_address = self.token_mints.get(from_token_symbol)
        to_token_mint_address = self.token_mints.get(to_token_symbol)

        if not from_token_mint_address or not to_token_mint_address:
            logging.error(f"Token mint addresses not found for {from_token_symbol} or {to_token_symbol}")
            return

        from_token_mint = Pubkey.from_string(from_token_mint_address)
        to_token_mint = Pubkey.from_string(to_token_mint_address)

        # Get associated token accounts
        from_token_account = await get_associated_token_address(self.wallet.pubkey(), from_token_mint)
        to_token_account = await get_associated_token_address(self.wallet.pubkey(), to_token_mint)

        # Get pool information for the token pair
        pool_info = find_pool_by_tokens(self.pools, from_token_mint_address, to_token_mint_address)
        if not pool_info:
            logging.error(f"No pool found for {from_token_symbol}/{to_token_symbol}")
            return

        # Build transaction
        transaction = Transaction()
        # Construct swap instruction
        swap_instruction = self.create_swap_instruction(
            user_source_token_account=from_token_account,
            user_destination_token_account=to_token_account,
            user_authority=self.wallet.pubkey(),
            pool_info=pool_info,
            amount_in=amount_in
        )
        transaction.add(swap_instruction)

        # Sign and send transaction
        try:
            response = await solana_client.send_transaction(
                transaction, self.wallet, opts=TxOpts(preflight_commitment="confirmed")
            )
            logging.info(f"Swap transaction sent: {response}")
            send_email(
                subject="Trading Bot Alert: Swap Executed",
                body=f"Swapped {amount_in} units of {from_token_symbol} to {to_token_symbol}. Transaction ID: {response}"
            )
            await self.discord_alert.send_message(f"Swapped {amount_in} units of {from_token_symbol} to {to_token_symbol}.")
        except Exception as e:
            logging.error(f"Error performing swap: {e}")
            send_email(
                subject="Trading Bot Alert: Swap Failed",
                body=f"Failed to swap {amount_in} units of {from_token_symbol} to {to_token_symbol}. Error: {e}"
            )

    def create_swap_instruction(self, user_source_token_account, user_destination_token_account,
                                user_authority, pool_info, amount_in):
        """
        Constructs the swap instruction for Raydium.
        """
        # Raydium's Swap instruction data layout:
        # u8 Instruction (9 for Swap)
        # u64 Amount In
        # u64 Minimum Amount Out

        instruction_data = struct.pack('<BQQ', 9, amount_in, 0)  # Swap instruction ID, amount_in, min_amount_out (0)

        keys = [
            # User accounts
            AccountMeta(pubkey=user_source_token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=user_destination_token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=user_authority, is_signer=True, is_writable=False),

            # Pool accounts
            AccountMeta(pubkey=Pubkey.from_string(pool_info['ammId']), is_signer=False, is_writable=True),
            AccountMeta(pubkey=Pubkey.from_string(pool_info['ammAuthority']), is_signer=False, is_writable=False),
            AccountMeta(pubkey=Pubkey.from_string(pool_info['ammOpenOrders']), is_signer=False, is_writable=True),
            AccountMeta(pubkey=Pubkey.from_string(pool_info['ammTargetOrders']), is_signer=False, is_writable=True),
            AccountMeta(pubkey=Pubkey.from_string(pool_info['poolCoinTokenAccount']), is_signer=False, is_writable=True),
            AccountMeta(pubkey=Pubkey.from_string(pool_info['poolPcTokenAccount']), is_signer=False, is_writable=True),

            # Serum market accounts
            AccountMeta(pubkey=Pubkey.from_string(pool_info['serumProgramId']), is_signer=False, is_writable=False),
            AccountMeta(pubkey=Pubkey.from_string(pool_info['serumMarket']), is_signer=False, is_writable=True),
            AccountMeta(pubkey=Pubkey.from_string(pool_info['serumBids']), is_signer=False, is_writable=True),
            AccountMeta(pubkey=Pubkey.from_string(pool_info['serumAsks']), is_signer=False, is_writable=True),
            AccountMeta(pubkey=Pubkey.from_string(pool_info['serumEventQueue']), is_signer=False, is_writable=True),
            AccountMeta(pubkey=Pubkey.from_string(pool_info['serumCoinVaultAccount']), is_signer=False, is_writable=True),
            AccountMeta(pubkey=Pubkey.from_string(pool_info['serumPcVaultAccount']), is_signer=False, is_writable=True),
            AccountMeta(pubkey=Pubkey.from_string(pool_info['serumVaultSigner']), is_signer=False, is_writable=False),

            # Programs
            AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYS_PROGRAM_ID, is_signer=False, is_writable=False),
        ]

        swap_instruction = Instruction(
            program_id=Pubkey.from_string(pool_info['programId']),
            accounts=keys,
            data=instruction_data
        )

        return swap_instruction

    async def make_trade(self, symbol, decision):
        token_symbol = symbol.split('/')[0]
        from_token = 'USDT' if decision == 'buy' else token_symbol
        to_token = token_symbol if decision == 'buy' else 'USDT'
        amount = 1  # Adjust as per your strategy

        # Convert amount to smallest units
        decimals = 6  # Default to 6 decimals
        amount_in = int(amount * (10 ** decimals))

        logging.info(f"Initiating trade: {decision.upper()} {from_token} to {to_token}, Amount: {amount_in}")
        await self.perform_swap(from_token, to_token, amount_in)

        if decision == 'buy':
            data = await self.fetch_data(symbol)
            if data.empty:
                logging.error(f"Failed to fetch data for {symbol}")
                return
            latest_close = data['close'].iloc[-1]
            self.current_positions[token_symbol] = {
                'amount': amount,
                'purchase_price': latest_close
            }
            logging.info(f"Purchased {token_symbol} at {latest_close} USDT")
            send_email(
                subject=f"Trading Bot Alert: BUY {token_symbol}",
                body=f"Purchased {token_symbol} at {latest_close} USDT"
            )
            await self.discord_alert.send_message(f"Purchased {token_symbol} at {latest_close} USDT")
        elif decision == 'sell':
            if token_symbol in self.current_positions:
                logging.info(f"Sold {token_symbol}")
                send_email(
                    subject=f"Trading Bot Alert: SELL {token_symbol}",
                    body=f"Sold {token_symbol}"
                )
                await self.discord_alert.send_message(f"Sold {token_symbol}")
                del self.current_positions[token_symbol]

# Instantiate the bot
bot = TradingBot()
