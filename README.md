# Real_savdo — Avtomatik savdo boti

Binance birjasida texnik indikatorlar asosida avtomatik savdo qiluvchi bot.
Streamlit orqali boshqariladi, natijalar Telegram'ga yuboriladi.

## Imkoniyatlar

- Binance (testnet va asosiy tarmoq) bilan ishlash — `ccxt`
- Indikatorlar: EMA (fast/slow), RSI, ATR
- Risk menejmenti: stop-loss (ATR asosida), risk/reward nisbati, kunlik va umumiy zarar limiti
- Trailing stop qo'llab-quvvatlanadi
- Savdo sessiyalari (London / New York)
- Telegram orqali xabarnoma
- Streamlit veb-interfeys (`main.py`)

## O'rnatish

```bash
git clone https://github.com/dastonbek8747/Real_savdo.git
cd Real_savdo
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Sozlash

1. `.env.example` dan nusxa oling:

   ```bash
   cp .env.example .env
   ```

2. `.env` faylini o'z qiymatlaringiz bilan to'ldiring (Binance API kalitlari, Telegram token va h.k.).

   > `.env` faylini hech qachon GitHub'ga yuklamang — u `.gitignore`da.

## Ishga tushirish

Streamlit interfeysi:

```bash
streamlit run main.py
```

Avtomatik savdo (fon rejimi):

```bash
python avto_savdo.py
```

## Ogohlantirish

Avtomatik savdo moliyaviy xavf bilan bog'liq. Avval `USE_TESTNET=True` bilan sinab ko'ring.
Muallif yuzaga kelishi mumkin bo'lgan zarar uchun javobgar emas.

## Litsenziya

MIT
