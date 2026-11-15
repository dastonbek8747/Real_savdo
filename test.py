# Namuna: haqiqiy qiymatlarni .env faylida saqlang, kodga yozmang.
# .env.example dan nusxa olib, .env yarating va o'z kalitlaringizni kiriting.

# import ccxt
#
# exchange = ccxt.binance(
#     {
#         "apiKey": os.getenv("BINANCE_API_KEY"),
#         "secret": os.getenv("BINANCE_API_SECRET"),
#     }
# )
# exchange.set_sandbox_mode(True)
# balance = exchange.fetch_balance()
#
# usdt_balance = balance["USDT"]["free"]   # faqat ishlatish mumkin bo'lgan
# usdt_total   = balance["USDT"]["total"]  # free + locked
# usdt_used    = balance["USDT"]["used"]   # orderlarda band
#
# print("USDT free:", usdt_balance)
# print("USDT total:", usdt_total)
# print("USDT used:", usdt_used)

# === .env namunasi (.env.example ga qarang) ===
# BINANCE_API_KEY=your_binance_api_key
# BINANCE_API_SECRET=your_binance_api_secret
# SYMBOL=BTC/USDT
# TIMEFRAME=1m
# TRADE_AMOUNT_USDT=100.0
# RISK_PER_TRADE_PCT=0.7
#
# # Indicators
# EMA_FAST=20
# EMA_SLOW=50
# RSI_PERIOD=14
# ATR_PERIOD=14
#
# # Risk Management
# ATR_SL_MULTIPLIER=1.5
# RISK_REWARD_RATIO=2.0
# MAX_DAILY_LOSS_PCT=3.0
# MAX_TOTAL_LOSS_PCT=6.0
# MAX_CONSECUTIVE_LOSSES=2
#
# # Mode
# USE_TESTNET=True
#
# # Telegram
# TELEGRAM_BOT_TOKEN=your_telegram_bot_token
# TELEGRAM_CHAT_ID=your_telegram_chat_id
