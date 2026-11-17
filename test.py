# Scratch file kept for manual experiments with the exchange client.
# Never hardcode credentials here. Read them from the environment instead:
#
#   BINANCE_API_KEY=...
#   BINANCE_API_SECRET=...
#
# import os
# import ccxt
#
# exchange = ccxt.binance({
#     "apiKey": os.getenv("BINANCE_API_KEY"),
#     "secret": os.getenv("BINANCE_API_SECRET"),
# })
# exchange.set_sandbox_mode(True)
# balance = exchange.fetch_balance()
# print("USDT free:", balance["USDT"]["free"])
