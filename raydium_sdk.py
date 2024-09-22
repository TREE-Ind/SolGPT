# raydium_sdk.py

import requests
from solana.publickey import PublicKey
import logging

def get_raydium_pools():
    """
    Fetch the list of pools from Raydium's API.
    """
    try:
        response = requests.get("https://api.raydium.io/v2/sdk/ammPools")
        pools = response.json()
        return pools
    except Exception as e:
        logging.error(f"Error fetching Raydium pools: {e}")
        return []

def find_pool_by_tokens(pools, from_token_mint, to_token_mint):
    """
    Find the pool information for a given token pair.
    """
    for pool in pools:
        if (pool['baseMint'] == from_token_mint and pool['quoteMint'] == to_token_mint) or \
           (pool['baseMint'] == to_token_mint and pool['quoteMint'] == from_token_mint):
            return pool
    return None

def get_token_mints():
    """
    Fetch the token mints from a public Solana token list.
    """
    try:
        response = requests.get("https://raw.githubusercontent.com/solana-labs/token-list/main/src/tokens/solana.tokenlist.json")
        tokens = response.json()['tokens']
        token_mints = {token['symbol']: token['address'] for token in tokens}
        return token_mints
    except Exception as e:
        logging.error(f"Error fetching token mints: {e}")
        return {}
