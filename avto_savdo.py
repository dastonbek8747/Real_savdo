import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import talib
import requests
from dotenv import load_dotenv
import logging
import signal
import sys

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)


class BotStateManager:
    """Bot holatini boshqarish"""
    STATE_FILE = 'bot_state.json'

    @staticmethod
    def save_state(running, pid=None):
        """Bot holatini saqlash"""
        state = {
            'running': running,
            'pid': pid,
            'last_update': datetime.now().isoformat()
        }
        try:
            with open(BotStateManager.STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logging.error(f"State saqlash xatosi: {e}")

    @staticmethod
    def clear_state():
        """Bot to'xtaganida holatni tozalash"""
        BotStateManager.save_state(False, None)


class TradingBot:
    """Professional M1 Scalping Bot - Real Trading Ready"""

    def __init__(self):
        # API credentials
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')

        if not self.api_key or not self.api_secret:
            raise ValueError("❌ API keys .env faylida bo'lishi kerak!")

        # Trading parameters
        self.symbol = os.getenv('SYMBOL', 'BTC/USDT')
        self.timeframe = os.getenv('TIMEFRAME', '1m')
        self.trade_amount = float(os.getenv('TRADE_AMOUNT_USDT', '50'))

        # Technical indicators
        self.ema_fast = int(os.getenv('EMA_FAST', '20'))
        self.ema_slow = int(os.getenv('EMA_SLOW', '50'))
        self.rsi_period = int(os.getenv('RSI_PERIOD', '14'))
        self.atr_period = int(os.getenv('ATR_PERIOD', '14'))

        # Risk management
        self.atr_sl_multiplier = float(os.getenv('ATR_SL_MULTIPLIER', '1.5'))
        self.risk_reward_ratio = float(os.getenv('RISK_REWARD_RATIO', '2.0'))
        self.risk_per_trade_pct = float(os.getenv('RISK_PER_TRADE_PCT', '0.5'))
        self.max_daily_loss_pct = float(os.getenv('MAX_DAILY_LOSS_PCT', '3.0'))
        self.max_total_loss_pct = float(os.getenv('MAX_TOTAL_LOSS_PCT', '6.0'))
        self.max_consecutive_losses = int(os.getenv('MAX_CONSECUTIVE_LOSSES', '2'))

        # Market filters
        self.min_atr_value = float(os.getenv('MIN_ATR_VALUE', '50'))
        self.max_spread_pct = float(os.getenv('MAX_SPREAD_PCT', '0.15'))
        self.min_signal_confidence = float(os.getenv('MIN_SIGNAL_CONFIDENCE', '0.8'))

        # Trading sessions (UTC)
        self.london_start = int(os.getenv('LONDON_SESSION_START', '8'))
        self.london_end = int(os.getenv('LONDON_SESSION_END', '16'))
        self.ny_start = int(os.getenv('NY_SESSION_START', '13'))
        self.ny_end = int(os.getenv('NY_SESSION_END', '21'))

        # Trailing stop (optional)
        self.use_trailing_stop = os.getenv('USE_TRAILING_STOP', 'False').lower() == 'true'
        self.trailing_stop_pct = float(os.getenv('TRAILING_STOP_PCT', '0.5'))

        # Telegram
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # Demo/Real mode
        self.use_testnet = os.getenv('USE_TESTNET', 'True').lower() == 'true'

        # Initialize exchange
        self._init_exchange()

        # State tracking
        self.initial_balance = self.get_account_balance()
        self.daily_start_balance = self.initial_balance
        self.daily_loss = 0
        self.total_loss = 0
        self.consecutive_losses = 0
        self.current_position = None
        self.trades_history = self.load_trades_history()
        self.is_running = False
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
        self.last_trade_time = None
        self.min_trade_interval = 60  # Minimum 1 daqiqa trade orasida

        # Signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        logging.info(f"🤖 Bot tayyor: {self.symbol} {self.timeframe} Scalping")
        logging.info(f"💰 Boshlang'ich balans: ${self.initial_balance:.2f}")

    def _init_exchange(self):
        """Exchange ulanishni sozlash"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
                'timeout': 30000,
            })

            if self.use_testnet:
                self.exchange.set_sandbox_mode(True)
                logging.info("🔶 DEMO REJIM - Testnet")
            else:
                logging.warning("🟢 REAL REJIM - Live Trading!")

            # Test connection
            balance = self.exchange.fetch_balance()
            usdt = balance.get('USDT', {}).get('total', 0)
            logging.info(f"✅ Binance ulandi! USDT: ${usdt:.2f}")

        except Exception as e:
            logging.error(f"❌ Binance ulanish xatosi: {e}")
            raise Exception(f"Exchange xatosi: {str(e)}")

    def signal_handler(self, signum, frame):
        """Signal handler"""
        logging.info(f"🛑 Signal qabul qilindi: {signum}")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Cleanup jarayoni"""
        logging.info("🧹 Cleanup boshlandi...")

        # Ochiq pozitsiyani yopish
        if self.current_position:
            try:
                current_price = self.get_current_price()
                if current_price > 0:
                    self.close_position(current_price, "BOT_STOPPED")
            except Exception as e:
                logging.error(f"Pozitsiya yopish xatosi: {e}")

        # Holatni tozalash
        BotStateManager.clear_state()

        # Final report
        final_balance = self.get_account_balance()
        total_pnl = final_balance - self.initial_balance

        self.send_telegram(
            f"🛑 <b>Bot To'xtatildi</b>\n\n"
            f"💼 Boshlang'ich: ${self.initial_balance:.2f}\n"
            f"💰 Yakuniy: ${final_balance:.2f}\n"
            f"📊 Umumiy P/L: ${total_pnl:.2f} ({(total_pnl / self.initial_balance) * 100:.2f}%)"
        )

        logging.info("👋 Cleanup tugadi")

    def load_trades_history(self):
        """Savdolar tarixini yuklash"""
        try:
            with open('trades.json', 'r') as f:
                return json.load(f)
        except:
            return []

    def save_trades_history(self):
        """Savdolar tarixini saqlash"""
        try:
            with open('trades.json', 'w') as f:
                json.dump(self.trades_history, f, indent=2)
        except Exception as e:
            logging.error(f"❌ Saqlash xatosi: {e}")

    def get_account_balance(self):
        """Hisob balansini olish"""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get('USDT', {}).get('free', 0))
        except Exception as e:
            logging.error(f"Balans olish xatosi: {e}")
            return 0

    def get_current_price(self):
        """Joriy narxni olish"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return float(ticker['last'])
        except Exception as e:
            logging.error(f"Narx olish xatosi: {e}")
            return 0

    def reset_daily_stats(self):
        """Kunlik statistikani yangilash"""
        now = datetime.now()
        if now.date() > self.daily_reset_time.date():
            self.daily_loss = 0
            self.daily_start_balance = self.get_account_balance()
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0)
            logging.info("🔄 Kunlik statistika yangilandi")

    def is_trading_session(self):
        """Trading sessiyasini tekshirish"""
        now = datetime.utcnow()
        hour = now.hour

        in_london = self.london_start <= hour < self.london_end
        in_ny = self.ny_start <= hour < self.ny_end

        return in_london or in_ny

    def get_historical_data(self, limit=200):
        """Tarixiy ma'lumotlarni olish"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            return df
        except Exception as e:
            logging.error(f"❌ Ma'lumot olish xatosi: {e}")
            return None

    def calculate_indicators(self, df):
        """Texnik indikatorlarni hisoblash"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values

            # EMA
            df['ema_fast'] = talib.EMA(close, timeperiod=self.ema_fast)
            df['ema_slow'] = talib.EMA(close, timeperiod=self.ema_slow)

            # RSI
            df['rsi'] = talib.RSI(close, timeperiod=self.rsi_period)

            # ATR
            df['atr'] = talib.ATR(high, low, close, timeperiod=self.atr_period)

            # MACD (qo'shimcha filtr)
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close, 12, 26, 9)

            # Volume MA
            df['volume_ma'] = talib.SMA(df['volume'].values, timeperiod=20)

            return df
        except Exception as e:
            logging.error(f"❌ Indikator xatosi: {e}")
            return df

    def check_market_conditions(self, df):
        """Bozor sharoitlarini tekshirish"""
        if len(df) < self.ema_slow:
            return False, "Ma'lumot yetarli emas"

        latest = df.iloc[-1]

        # ATR tekshiruvi (volatilnost)
        if latest['atr'] < self.min_atr_value:
            return False, f"ATR juda past: {latest['atr']:.2f}"

        # Spread tekshiruvi
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            bid = float(ticker['bid'])
            ask = float(ticker['ask'])
            spread_pct = ((ask - bid) / bid) * 100

            if spread_pct > self.max_spread_pct:
                return False, f"Spread juda katta: {spread_pct:.3f}%"
        except:
            pass

        # Volume tekshiruvi
        if latest['volume'] < latest['volume_ma'] * 0.5:
            return False, "Volume juda past"

        return True, "OK"

    def check_buy_signal(self, df):
        """BUY signalni tekshirish"""
        if len(df) < self.ema_slow:
            return False, 0

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Asosiy shartlar
        cond1 = latest['close'] > latest['ema_slow']  # Narx EMA50 dan yuqori
        cond2 = latest['ema_fast'] > latest['ema_slow']  # EMA20 > EMA50
        cond3 = latest['rsi'] > 50 and latest['rsi'] < 70  # RSI 50-70 oralig'ida
        cond4 = prev['close'] > prev['open']  # Oldingi sham bullish
        cond5 = self.is_trading_session()  # Trading sessiya

        # Qo'shimcha filtrlar
        cond6 = latest['macd'] > latest['macd_signal']  # MACD ijobiy
        cond7 = latest['volume'] > latest['volume_ma']  # Volume yuqori
        cond8 = latest['close'] > prev['close']  # Narx o'sishda

        main_conditions = [cond1, cond2, cond3, cond4, cond5]
        extra_conditions = [cond6, cond7, cond8]

        # Confidence hisoblash
        main_score = sum(main_conditions) / len(main_conditions)
        extra_score = sum(extra_conditions) / len(extra_conditions)
        confidence = (main_score * 0.7) + (extra_score * 0.3)

        signal = all(main_conditions) and confidence >= self.min_signal_confidence

        if signal:
            logging.info(
                f"✅ BUY SIGNAL | Confidence: {confidence:.2%} | "
                f"Narx: {latest['close']:.2f} | RSI: {latest['rsi']:.1f} | "
                f"ATR: {latest['atr']:.2f}"
            )

        return signal, confidence

    def check_sell_signal(self, df):
        """SELL signalni tekshirish"""
        if len(df) < self.ema_slow:
            return False, 0

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Asosiy shartlar
        cond1 = latest['close'] < latest['ema_slow']
        cond2 = latest['ema_fast'] < latest['ema_slow']
        cond3 = latest['rsi'] < 50 and latest['rsi'] > 30
        cond4 = prev['close'] < prev['open']
        cond5 = self.is_trading_session()

        # Qo'shimcha filtrlar
        cond6 = latest['macd'] < latest['macd_signal']
        cond7 = latest['volume'] > latest['volume_ma']
        cond8 = latest['close'] < prev['close']

        main_conditions = [cond1, cond2, cond3, cond4, cond5]
        extra_conditions = [cond6, cond7, cond8]

        main_score = sum(main_conditions) / len(main_conditions)
        extra_score = sum(extra_conditions) / len(extra_conditions)
        confidence = (main_score * 0.7) + (extra_score * 0.3)

        signal = all(main_conditions) and confidence >= self.min_signal_confidence

        if signal:
            logging.info(
                f"✅ SELL SIGNAL | Confidence: {confidence:.2%} | "
                f"Narx: {latest['close']:.2f} | RSI: {latest['rsi']:.1f} | "
                f"ATR: {latest['atr']:.2f}"
            )

        return signal, confidence

    def calculate_position_size(self, entry_price, sl_distance):
        """Pozitsiya hajmini hisoblash"""
        # Risk asosida
        risk_amount = self.initial_balance * (self.risk_per_trade_pct / 100)
        quantity = risk_amount / sl_distance

        # Market precision
        market = self.exchange.market(self.symbol)
        min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.001)

        quantity = max(quantity, min_amount)
        quantity = self.exchange.amount_to_precision(self.symbol, quantity)

        return float(quantity)

    def calculate_sl_tp(self, entry_price, side, atr_value):
        """Stop Loss va Take Profit hisoblash"""
        sl_distance = atr_value * self.atr_sl_multiplier
        tp_distance = sl_distance * self.risk_reward_ratio

        if side == 'BUY':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        return stop_loss, take_profit, sl_distance

    def check_risk_limits(self):
        """Risk limitlarini tekshirish"""
        self.reset_daily_stats()

        # Ketma-ket zarar
        if self.consecutive_losses >= self.max_consecutive_losses:
            logging.warning(f"⚠️ {self.consecutive_losses} ta ketma-ket zarar!")
            return False, "CONSECUTIVE_LOSS_LIMIT"

        # Kunlik DD
        current_balance = self.get_account_balance()
        daily_dd = self.daily_start_balance - current_balance
        daily_dd_pct = (daily_dd / self.daily_start_balance) * 100

        if daily_dd_pct >= self.max_daily_loss_pct:
            logging.warning(f"⚠️ Kunlik DD: {daily_dd_pct:.2f}%")
            return False, "DAILY_DD_LIMIT"

        # Umumiy DD
        total_dd_pct = ((self.initial_balance - current_balance) / self.initial_balance) * 100

        if total_dd_pct >= self.max_total_loss_pct:
            logging.warning(f"⚠️ Umumiy DD: {total_dd_pct:.2f}%")
            return False, "TOTAL_DD_LIMIT"

        # Ochiq pozitsiya
        if self.current_position is not None:
            return False, "POSITION_OPEN"

        # Trade interval
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < self.min_trade_interval:
                return False, "MIN_INTERVAL"

        return True, "OK"

    def send_telegram(self, message):
        """Telegram xabar yuborish"""
        if not self.telegram_token or not self.telegram_chat_id:
            return

        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logging.error(f"❌ Telegram xato: {e}")

    def execute_trade(self, signal, current_price, atr_value, confidence):
        """Savdoni bajarish"""
        risk_ok, reason = self.check_risk_limits()
        if not risk_ok:
            logging.info(f"⏸️ Savdo rad etildi: {reason}")
            return

        try:
            stop_loss, take_profit, sl_distance = self.calculate_sl_tp(
                current_price, signal, atr_value
            )

            quantity = self.calculate_position_size(current_price, sl_distance)

            # Order berish
            try:
                if signal == 'BUY':
                    order = self.exchange.create_market_buy_order(self.symbol, quantity)
                else:
                    order = self.exchange.create_market_sell_order(self.symbol, quantity)

                logging.info(f"✅ Order executed: {order['id']}")
                actual_price = float(order.get('price', current_price))

            except Exception as order_error:
                logging.warning(f"⚠️ Order xatosi: {order_error}")
                # Simulation mode
                order = {
                    'id': f"SIM_{int(time.time())}",
                    'symbol': self.symbol,
                    'side': signal.lower(),
                    'price': current_price,
                    'amount': quantity,
                    'status': 'filled'
                }
                actual_price = current_price

            # Pozitsiyani saqlash
            self.current_position = {
                'entry_time': datetime.now().isoformat(),
                'entry_price': actual_price,
                'quantity': quantity,
                'side': signal,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr_value,
                'confidence': confidence,
                'order_id': order['id'],
                'highest_price': actual_price if signal == 'BUY' else None,
                'lowest_price': actual_price if signal == 'SELL' else None
            }

            # Trade log
            trade_log = {
                'timestamp': datetime.now().isoformat(),
                'action': 'OPEN',
                'signal': signal,
                'entry_price': actual_price,
                'quantity': quantity,
                'sl': stop_loss,
                'tp': take_profit,
                'atr': atr_value,
                'risk_reward': f"1:{self.risk_reward_ratio}",
                'confidence': confidence
            }
            self.trades_history.append(trade_log)
            self.save_trades_history()

            # Update last trade time
            self.last_trade_time = datetime.now()

            # Telegram notification
            mode = "🔶 DEMO" if self.use_testnet else "🟢 REAL"
            risk_amount = quantity * sl_distance
            potential_profit = quantity * (sl_distance * self.risk_reward_ratio)

            message = f"""
{mode} <b>🚀 POZITSIYA OCHILDI</b>

📊 Strategiya: M1 Scalping
💰 Instrument: {self.symbol}
📈 Signal: <b>{signal}</b>
🎲 Confidence: {confidence * 100:.1f}%

💵 Entry: ${actual_price:.2f}
📦 Hajm: {quantity} ({quantity * actual_price:.2f} USDT)

🛑 Stop Loss: ${stop_loss:.2f}
🎯 Take Profit: ${take_profit:.2f}

📊 Risk: ${risk_amount:.2f}
💰 Kutilgan foyda: ${potential_profit:.2f}
📈 R:R = 1:{self.risk_reward_ratio}
📊 ATR: {atr_value:.2f}

⏰ {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
"""
            self.send_telegram(message)
            logging.info(
                f"✅ {signal} @ ${actual_price:.2f} | "
                f"Qty: {quantity} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}"
            )

        except Exception as e:
            logging.error(f"❌ Execute xatosi: {e}")

    def check_exit_conditions(self, current_price):
        """Chiqish shartlarini tekshirish"""
        if self.current_position is None:
            return

        position = self.current_position
        side = position['side']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']

        # Trailing stop
        if self.use_trailing_stop:
            if side == 'BUY':
                if position['highest_price'] is None or current_price > position['highest_price']:
                    position['highest_price'] = current_price
                    # Update SL
                    new_sl = current_price * (1 - self.trailing_stop_pct / 100)
                    if new_sl > stop_loss:
                        stop_loss = new_sl
                        position['stop_loss'] = new_sl
                        logging.info(f"📈 Trailing SL yangilandi: ${new_sl:.2f}")
            else:
                if position['lowest_price'] is None or current_price < position['lowest_price']:
                    position['lowest_price'] = current_price
                    new_sl = current_price * (1 + self.trailing_stop_pct / 100)
                    if new_sl < stop_loss:
                        stop_loss = new_sl
                        position['stop_loss'] = new_sl
                        logging.info(f"📉 Trailing SL yangilandi: ${new_sl:.2f}")

        should_exit = False
        exit_reason = ""

        # Stop Loss
        if side == 'BUY' and current_price <= stop_loss:
            should_exit = True
            exit_reason = "STOP_LOSS"
        elif side == 'SELL' and current_price >= stop_loss:
            should_exit = True
            exit_reason = "STOP_LOSS"

        # Take Profit
        if side == 'BUY' and current_price >= take_profit:
            should_exit = True
            exit_reason = "TAKE_PROFIT"
        elif side == 'SELL' and current_price <= take_profit:
            should_exit = True
            exit_reason = "TAKE_PROFIT"

        if should_exit:
            self.close_position(current_price, exit_reason)

    def close_position(self, exit_price, reason):
        """Pozitsiyani yopish"""
        if self.current_position is None:
            return

        position = self.current_position
        entry_price = position['entry_price']
        quantity = position['quantity']
        side = position['side']

        try:
            # Order berish
            if side == 'BUY':
                close_order = self.exchange.create_market_sell_order(self.symbol, quantity)
            else:
                close_order = self.exchange.create_market_buy_order(self.symbol, quantity)

            logging.info(f"✅ Closed: {close_order['id']}")
            actual_exit = float(close_order.get('price', exit_price))

        except Exception as e:
            logging.warning(f"⚠️ Close xatosi: {e}")
            actual_exit = exit_price

        # P&L hisoblash
        if side == 'BUY':
            pnl = (actual_exit - entry_price) * quantity
        else:
            pnl = (entry_price - actual_exit) * quantity

        pnl_pct = (pnl / (entry_price * quantity)) * 100

        # Stats yangilash
        self.daily_loss += pnl if pnl < 0 else 0
        self.total_loss += pnl if pnl < 0 else 0

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Trade log
        duration = datetime.now() - datetime.fromisoformat(position['entry_time'])
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'action': 'CLOSE',
            'reason': reason,
            'entry_price': entry_price,
            'exit_price': actual_exit,
            'quantity': quantity,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'duration_seconds': duration.total_seconds()
        }
        self.trades_history.append(trade_log)
        self.save_trades_history()

        # Telegram notification
        mode = "🔶 DEMO" if self.use_testnet else "🟢 REAL"
        emoji = "🟢" if pnl > 0 else "🔴"
        current_balance = self.get_account_balance()

        message = f"""
{mode} {emoji} <b>POZITSIYA YOPILDI</b>

💰 Instrument: {self.symbol}
📋 Sabab: <b>{reason}</b>

📥 Entry: ${entry_price:.2f}
📤 Exit: ${actual_exit:.2f}
⏱️ Davomiyligi: {int(duration.total_seconds() / 60)} daqiqa

💵 P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)
📊 Ketma-ket zarar: {self.consecutive_losses}
💼 Joriy balans: ${current_balance:.2f}

⏰ {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
"""
        self.send_telegram(message)

        result_emoji = "✅" if pnl > 0 else "❌"
        logging.info(
            f"{result_emoji} Closed: {reason} | "
            f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | "
            f"Duration: {int(duration.total_seconds() / 60)}m"
        )

        self.current_position = None

    def run(self):
        """Bot asosiy loop"""
        self.is_running = True

        # PID saqlash
        BotStateManager.save_state(True, os.getpid())

        logging.info("=" * 60)
        logging.info("🚀 PROFESSIONAL M1 SCALPING BOT")
        logging.info("=" * 60)
        logging.info(f"📍 PID: {os.getpid()}")
        logging.info(f"💰 Symbol: {self.symbol}")
        logging.info(f"⏰ Timeframe: {self.timeframe}")
        logging.info(f"💵 Initial Balance: ${self.initial_balance:.2f}")
        logging.info(f"🎯 Risk per trade: {self.risk_per_trade_pct}%")
        logging.info(f"📊 R:R Ratio: 1:{self.risk_reward_ratio}")
        logging.info(f"🛡️ Max Daily DD: {self.max_daily_loss_pct}%")
        logging.info(f"🛡️ Max Total DD: {self.max_total_loss_pct}%")
        logging.info("=" * 60)

        mode = "DEMO (Testnet)" if self.use_testnet else "⚠️ REAL (Live Trading)"
        self.send_telegram(
            f"✅ <b>Bot Ishga Tushdi</b>\n\n"
            f"📊 Strategiya: M1 Scalping\n"
            f"🎮 Rejim: {mode}\n"
            f"💰 Instrument: {self.symbol}\n"
            f"⏰ Timeframe: {self.timeframe}\n"
            f"💵 Balans: ${self.initial_balance:.2f}\n"
            f"📈 Risk:Reward: 1:{self.risk_reward_ratio}\n"
            f"🎯 Risk: {self.risk_per_trade_pct}%\n"
            f"🛡️ Max DD: {self.max_daily_loss_pct}% kunlik\n"
            f"🔢 PID: {os.getpid()}"
        )

        iteration = 0
        last_balance_check = time.time()

        while self.is_running:
            try:
                iteration += 1

                # Ma'lumotlarni olish
                df = self.get_historical_data(limit=200)
                if df is None or len(df) < self.ema_slow:
                    logging.warning("⚠️ Ma'lumot yetarli emas")
                    time.sleep(60)
                    continue

                # Indikatorlarni hisoblash
                df = self.calculate_indicators(df)

                current_price = df.iloc[-1]['close']
                atr_value = df.iloc[-1]['atr']

                # Ochiq pozitsiyani tekshirish
                if self.current_position:
                    self.check_exit_conditions(current_price)

                    # Unrealized P/L
                    if iteration % 5 == 0:  # Har 5 iteratsiyada
                        position = self.current_position
                        if position['side'] == 'BUY':
                            unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                        else:
                            unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']

                        logging.info(
                            f"📊 Position update | "
                            f"Entry: ${position['entry_price']:.2f} | "
                            f"Current: ${current_price:.2f} | "
                            f"Unrealized: ${unrealized_pnl:.2f}"
                        )

                # Yangi savdo imkoniyatlarini qidirish
                if self.current_position is None:
                    # Trading sessiyani tekshirish
                    if not self.is_trading_session():
                        if iteration % 10 == 0:
                            logging.info("⏸️ Trading session emas")
                        time.sleep(60)
                        continue

                    # Bozor sharoitlarini tekshirish
                    market_ok, market_reason = self.check_market_conditions(df)
                    if not market_ok:
                        if iteration % 10 == 0:
                            logging.info(f"⏸️ Bozor: {market_reason}")
                        time.sleep(60)
                        continue

                    # BUY signal
                    buy_signal, buy_conf = self.check_buy_signal(df)
                    if buy_signal:
                        self.execute_trade('BUY', current_price, atr_value, buy_conf)

                    # SELL signal
                    if not buy_signal:  # Faqat BUY signal bo'lmasa
                        sell_signal, sell_conf = self.check_sell_signal(df)
                        if sell_signal:
                            self.execute_trade('SELL', current_price, atr_value, sell_conf)

                # Balansni davriy tekshirish (har 5 daqiqada)
                if time.time() - last_balance_check > 300:
                    current_balance = self.get_account_balance()
                    total_pnl = current_balance - self.initial_balance
                    logging.info(
                        f"💼 Balance: ${current_balance:.2f} | "
                        f"Total P/L: ${total_pnl:.2f} ({(total_pnl / self.initial_balance) * 100:.2f}%)"
                    )
                    last_balance_check = time.time()

                # Status log
                if iteration % 10 == 0:
                    status = "OPEN" if self.current_position else "WAITING"
                    logging.info(
                        f"💤 Iteration {iteration} | Status: {status} | "
                        f"Price: ${current_price:.2f} | ATR: {atr_value:.2f}"
                    )

                # 60 sekund kutish (1m timeframe)
                time.sleep(60)

            except KeyboardInterrupt:
                logging.info("🛑 Foydalanuvchi to'xtatdi")
                break
            except Exception as e:
                logging.error(f"❌ Loop xatosi: {e}")
                time.sleep(60)

        # Cleanup
        self.cleanup()

    def stop(self):
        """Botni to'xtatish"""
        self.is_running = False
        logging.info("🛑 Bot to'xtatilmoqda...")


if __name__ == "__main__":
    bot = None
    try:
        bot = TradingBot()
        bot.run()
    except KeyboardInterrupt:
        logging.info("🛑 Keyboard interrupt")
        if bot:
            bot.cleanup()
    except Exception as e:
        logging.error(f"❌ Fatal xato: {e}")
        if bot:
            bot.cleanup()
    finally:
        BotStateManager.clear_state()
        logging.info("✅ Dastur to'liq to'xtadi")