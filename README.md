# 💰 Real Savdo — M1 Scalping Bot

**M1 (1 daqiqa)** skalping strategiyasi asosida ishlaydigan savdo boti. Streamlit panel orqali boshqariladi, holatni saqlaydi va jarayon sifatida ishga tushadi.

---

## ✨ Imkoniyatlar

- ⚡ **M1 skalping:** qisqa muddatli savdo signallari
- 🧠 **Holat boshqaruvi:** `bot_state.json` orqali pozitsiyalar saqlanadi
- 📊 **Streamlit panel:** balans, savdolar, grafiklar
- 🔔 **Telegram xabarnomalar**
- 🛑 **Xavfsiz to'xtatish:** signal (SIGINT/SIGTERM) bilan toza yopilish
- 📝 **Log fayl:** `trading_bot.log`

---

## 🛠 Texnologiyalar

| Texnologiya | Maqsad |
|---|---|
| **Python 3.10+** | Asosiy til |
| **ccxt** | Birja bilan aloqa |
| **TA-Lib** | Texnik indikatorlar |
| **Streamlit** | Veb-interfeys |
| **Plotly** | Grafiklar |
| **psutil** | Jarayon boshqaruvi |

> ⚠️ **TA-Lib** tizimda alohida o'rnatilishi kerak (C kutubxonasi).

---

## ⚙️ O'rnatish

```bash
git clone https://github.com/dastonbek8747/Real_savdo.git
cd Real_savdo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# TA-Lib (Ubuntu/Debian)
sudo apt install build-essential
# ... TA-Lib C kutubxonasini o'rnating, so'ng: pip install TA-Lib
```

`.env` fayl yarating:

```env
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
```

> 🔒 `.env` git'ga qo'shilmaydi.

---

## ▶️ Ishga tushirish

```bash
streamlit run main.py
```

---

## 📁 Loyiha tuzilishi

```
Real_savdo/
├── main.py          # Streamlit panel
├── avto_savdo.py    # Savdo boti mantiqi
├── test.py          # Balans tekshirish namunasi
├── requirements.txt
└── .gitignore
```

---

## ⚠️ Ogohlantirish

Haqiqiy bozorda savdo qilish **moliyaviy zarar** keltirishi mumkin. Avval testnet'da sinang.

---

## 📄 Litsenziya

Litsenziya ko'rsatilmagan.
