# solana_connection.py

import json
import base64
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from dotenv import load_dotenv
import os

load_dotenv()

def load_wallet():
    private_key_str = os.getenv("SOLANA_PRIVATE_KEY")
    if not private_key_str:
        raise ValueError("SOLANA_PRIVATE_KEY not set in .env")

    try:
        # Try to parse the private key as a JSON array
        private_key_list = json.loads(private_key_str)
        if isinstance(private_key_list, list):
            private_key_bytes = bytes(private_key_list)
            keypair = Keypair.from_bytes(private_key_bytes)
            return keypair
        else:
            raise ValueError("Private key is not a valid list.")
    except json.JSONDecodeError:
        # If it's not JSON, assume it's base64-encoded
        try:
            private_key_bytes = base64.b64decode(private_key_str)
            keypair = Keypair.from_bytes(private_key_bytes)
            return keypair
        except Exception as e:
            raise ValueError(f"Failed to decode private key: {e}")

async def get_balance(client: AsyncClient, pubkey: str) -> float:
    """
    Fetches the SOL balance for a given public key.

    Args:
        client (AsyncClient): The Solana RPC client.
        pubkey (str): The public key of the wallet.

    Returns:
        float: The SOL balance.
    """
    try:
        response = await client.get_balance(pubkey)
        # The balance is now directly accessible from the response
        return response.value / 10**9  # Convert lamports to SOL
    except Exception as e:
        logging.error(f"Error fetching balance: {e}")
        return None
