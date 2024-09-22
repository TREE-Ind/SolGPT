# solana_connection.py

from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from dotenv import load_dotenv
import os
import base64

load_dotenv()

def load_wallet():
    private_key_str = os.getenv("SOLANA_PRIVATE_KEY")
    if not private_key_str:
        raise ValueError("SOLANA_PRIVATE_KEY not set in .env")

    private_key_bytes = base64.b64decode(private_key_str)
    keypair = Keypair.from_bytes(private_key_bytes)
    return keypair

async def get_balance(client, public_key):
    balance_response = await client.get_balance(public_key)
    balance = balance_response['result']['value'] / 1e9  # Convert lamports to SOL
    return balance
