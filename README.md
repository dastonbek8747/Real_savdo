# M1 Scalping Trading Bot

> An M1 scalping crypto trading bot with a Streamlit dashboard, ccxt and TA-Lib indicators. Demo and testnet only.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)


## Features

- One-minute scalping strategy with technical indicators.
- Streamlit dashboard with live charts.
- ccxt exchange layer.
- Demo/testnet mode by default.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Exchange | ccxt |
| Indicators | TA-Lib |
| Data | pandas |

## Quick Start

```bash
git clone https://github.com/dastonbek8747/m1-scalping-trading-bot.git
cd m1-scalping-trading-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add API keys
streamlit run main.py
```

## Disclaimer

Educational project. Not financial advice. Run it on a testnet first. The API keys that were once committed here are no longer valid; always rotate any key you ever pushed.

## License

MIT © 2026 Doston Sultonov.
