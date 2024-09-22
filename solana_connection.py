import json
import base64
from solders.keypair import Keypair

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

async def get_balance(client, public_key):
    balance_response = await client.get_balance(public_key)
    balance = balance_response['result']['value'] / 1e9  # Convert lamports to SOL
    return balance
