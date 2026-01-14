import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import ccxt
from dotenv import load_dotenv
import time
import subprocess
import sys
import psutil

load_dotenv()

st.set_page_config(
    page_title="M1 Scalping Bot - Professional Trading",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (minimal va professional)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        * { 
            font-family: 'Inter', sans-serif;
        }

        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 15px 0;
            margin-bottom: 5px;
        }

        .sub-header {
            text-align: center;
            color: #94a3b8;
            font-size: 1rem;
            margin-bottom: 25px;
        }

        .status-running {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }

        .status-stopped {
            background: linear-gradient(135deg, #64748b, #475569);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
        }

        .position-card {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
            border: 2px solid #10b981;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
        }

        .position-card.sell {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
            border-color: #ef4444;
        }

        .metric-positive { color: #10b981; font-weight: 600; }
        .metric-negative { color: #ef4444; font-weight: 600; }

        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #1e293b;
            margin: 20px 0 10px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #e2e8f0;
        }

        .info-badge {
            background: #f1f5f9;
            padding: 6px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
            color: #475569;
            display: inline-block;
            margin: 3px;
        }
    </style>
    """, unsafe_allow_html=True)


class BotStateManager:
    """Bot holatini boshqarish"""
    STATE_FILE = 'bot_state.json'

    @staticmethod
    def save_state(running, pid=None):
        state = {
            'running': running,
            'pid': pid,
            'last_update': datetime.now().isoformat()
        }
        try:
            with open(BotStateManager.STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"State save error: {e}")

    @staticmethod
    def load_state():
        try:
            if os.path.exists(BotStateManager.STATE_FILE):
                with open(BotStateManager.STATE_FILE, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {'running': False, 'pid': None, 'last_update': None}

    @staticmethod
    def is_process_running(pid):
        if pid is None:
            return False
        try:
            process = psutil.Process(pid)
            return process.is_running() and 'python' in process.name().lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    @staticmethod
    def get_real_status():
        state = BotStateManager.load_state()
        if not state['running']:
            return False, None
        if BotStateManager.is_process_running(state['pid']):
            return True, state['pid']
        else:
            BotStateManager.save_state(False, None)
            return False, None


class TradingDashboard:
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.symbol = os.getenv('SYMBOL', 'BTC/USDT')

        if 'use_testnet' not in st.session_state:
            st.session_state.use_testnet = os.getenv('USE_TESTNET', 'True').lower() == 'true'

        self.exchange = None
        if self.api_key and self.api_secret:
            try:
                self.exchange = ccxt.binance({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'enableRateLimit': True,
                })
                if st.session_state.use_testnet:
                    self.exchange.set_sandbox_mode(True)
            except Exception as e:
                st.error(f"❌ Connection error: {e}")

    def get_bot_status(self):
        return BotStateManager.get_real_status()

    def load_trades_history(self):
        try:
            with open('trades.json', 'r') as f:
                return json.load(f)
        except:
            return []

    def get_current_price(self):
        try:
            if self.exchange:
                ticker = self.exchange.fetch_ticker(self.symbol)
                return float(ticker['last'])
            return 0
        except:
            return 0

    def get_account_balance(self):
        try:
            if self.exchange:
                balance = self.exchange.fetch_balance()
                usdt = balance.get('USDT', {})
                return float(usdt.get('free', 0))
            return 0
        except:
            return 0

    def get_open_position(self):
        trades = self.load_trades_history()
        for trade in reversed(trades):
            if trade.get('action') == 'OPEN':
                trade_time = trade.get('timestamp')
                closed = False
                for t in trades:
                    if t.get('action') == 'CLOSE' and t.get('timestamp') > trade_time:
                        closed = True
                        break
                if not closed:
                    return trade
        return None

    def calculate_stats(self, trades):
        if not trades:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0, 'total_pnl': 0, 'avg_profit': 0,
                'avg_loss': 0, 'profit_factor': 0, 'total_volume': 0,
                'best_trade': 0, 'worst_trade': 0, 'open_trades': 0,
                'avg_duration': 0, 'total_fees': 0
            }

        closed = [t for t in trades if t.get('action') == 'CLOSE']
        open_trades = len([t for t in trades if t.get('action') == 'OPEN']) - len(closed)

        total = len(closed)
        winning = len([t for t in closed if t.get('pnl', 0) > 0])
        losing = len([t for t in closed if t.get('pnl', 0) < 0])
        total_pnl = sum([t.get('pnl', 0) for t in closed])

        profits = [t['pnl'] for t in closed if t.get('pnl', 0) > 0]
        losses = [abs(t['pnl']) for t in closed if t.get('pnl', 0) < 0]

        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = sum(profits) / sum(losses) if losses else (float('inf') if profits else 0)
        win_rate = (winning / total * 100) if total > 0 else 0

        total_volume = sum([t.get('quantity', 0) * t.get('entry_price', 0) for t in closed])
        best = max([t.get('pnl', 0) for t in closed]) if closed else 0
        worst = min([t.get('pnl', 0) for t in closed]) if closed else 0

        durations = [t.get('duration_seconds', 0) for t in closed if 'duration_seconds' in t]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            'total_trades': total,
            'winning_trades': winning,
            'losing_trades': losing,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_volume': total_volume,
            'best_trade': best,
            'worst_trade': worst,
            'open_trades': open_trades,
            'avg_duration': avg_duration
        }

    def plot_equity_curve(self, trades):
        closed = [t for t in trades if t.get('action') == 'CLOSE']
        if not closed:
            return None

        equity = [0]
        dates = []
        colors = []

        for trade in closed:
            pnl = trade.get('pnl', 0)
            equity.append(equity[-1] + pnl)
            try:
                dates.append(datetime.fromisoformat(trade['timestamp']))
            except:
                dates.append(datetime.now())
            colors.append('#10b981' if pnl > 0 else '#ef4444')

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Equity Curve', 'Trade Results'),
            vertical_spacing=0.12,
            row_heights=[0.7, 0.3]
        )

        fig.add_trace(go.Scatter(
            x=dates, y=equity,
            mode='lines+markers',
            name='Equity',
            line=dict(color='#667eea', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)',
            marker=dict(size=6, color=colors[1:] if len(colors) > 1 else colors)
        ), row=1, col=1)

        pnls = [t.get('pnl', 0) for t in closed]
        fig.add_trace(go.Bar(
            x=list(range(1, len(pnls) + 1)),
            y=pnls,
            marker_color=colors,
            name='P/L',
            showlegend=False
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_white",
            height=550,
            hovermode='x unified',
            font=dict(size=11),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        return fig

    def start_bot(self):
        try:
            is_running, _ = self.get_bot_status()
            if is_running:
                return False, "⚠️ Bot already running!"

            process = subprocess.Popen(
                [sys.executable, 'avto_savdo.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            BotStateManager.save_state(True, process.pid)
            return True, "✅ Bot started successfully!"

        except Exception as e:
            BotStateManager.save_state(False, None)
            return False, f"❌ Error: {str(e)}"

    def stop_bot(self):
        try:
            is_running, pid = self.get_bot_status()
            if not is_running:
                return False, "⚠️ Bot is not running"

            if pid:
                try:
                    process = psutil.Process(pid)
                    process.terminate()
                    process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    process.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            BotStateManager.save_state(False, None)
            return True, "🛑 Bot stopped!"

        except Exception as e:
            BotStateManager.save_state(False, None)
            return False, f"❌ Error: {str(e)}"


def main():
    st.markdown('<h1 class="main-header">💰 M1 SCALPING BOT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional Trading System | EMA + RSI + ATR + MACD</p>',
                unsafe_allow_html=True)

    dashboard = TradingDashboard()
    bot_running, bot_pid = dashboard.get_bot_status()

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        st.divider()

        # Connection status
        if dashboard.exchange:
            st.success("✅ Binance Connected")
        else:
            st.error("❌ Binance Disconnected")

        st.divider()

        # Bot status
        st.markdown("### 🤖 Bot Status")
        if bot_running:
            st.markdown('<p class="status-running">🟢 RUNNING</p>', unsafe_allow_html=True)
            if bot_pid:
                st.caption(f"Process ID: {bot_pid}")
        else:
            st.markdown('<p class="status-stopped">⚪ STOPPED</p>', unsafe_allow_html=True)

        st.divider()

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", disabled=bot_running, use_container_width=True, type="primary"):
                success, msg = dashboard.start_bot()
                if success:
                    st.success(msg)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(msg)

        with col2:
            if st.button("⏹️ Stop", disabled=not bot_running, use_container_width=True):
                success, msg = dashboard.stop_bot()
                if success:
                    st.success(msg)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(msg)

        st.divider()

        # Settings
        st.markdown("### 🛠️ Settings")

        symbol = st.text_input("Symbol", value=os.getenv('SYMBOL', 'BTC/USDT'),
                               disabled=bot_running)

        timeframe = st.selectbox("Timeframe", ['1m', '3m', '5m'],
                                 index=0, disabled=bot_running)

        trade_amount = st.number_input("Trade Amount (USDT)",
                                       value=float(os.getenv('TRADE_AMOUNT_USDT', '50')),
                                       min_value=10.0, step=10.0,
                                       disabled=bot_running)

        risk_pct = st.slider("Risk per Trade (%)", 0.1, 2.0,
                             float(os.getenv('RISK_PER_TRADE_PCT', '0.5')),
                             0.1, disabled=bot_running)

        use_testnet = st.checkbox("🔶 Demo Mode (Testnet)",
                                  value=st.session_state.use_testnet,
                                  disabled=bot_running)

        if st.button("💾 Save Settings", use_container_width=True, type="primary",
                     disabled=bot_running):
            env_content = f"""# Binance API
    BINANCE_API_KEY={os.getenv('BINANCE_API_KEY', '')}
    BINANCE_API_SECRET={os.getenv('BINANCE_API_SECRET', '')}

    # Trading
    SYMBOL={symbol}
    TIMEFRAME={timeframe}
    TRADE_AMOUNT_USDT={trade_amount}
    RISK_PER_TRADE_PCT={risk_pct}

    # Indicators
    EMA_FAST=20
    EMA_SLOW=50
    RSI_PERIOD=14
    ATR_PERIOD=14

    # Risk Management
    ATR_SL_MULTIPLIER=1.5
    RISK_REWARD_RATIO=2.0
    MAX_DAILY_LOSS_PCT=3.0
    MAX_TOTAL_LOSS_PCT=6.0
    MAX_CONSECUTIVE_LOSSES=2

    # Mode
    USE_TESTNET={'True' if use_testnet else 'False'}

    # Telegram
    TELEGRAM_BOT_TOKEN={os.getenv('TELEGRAM_BOT_TOKEN', '')}
    TELEGRAM_CHAT_ID={os.getenv('TELEGRAM_CHAT_ID', '')}
    """
            with open('.env', 'w') as f:
                f.write(env_content)
            st.success("✅ Settings saved!")
            time.sleep(1)
            st.rerun()

        st.divider()

        with st.expander("📖 Strategy Info"):
            st.markdown("""
    **Entry Conditions:**
    - BUY: Price > EMA50, EMA20 > EMA50, RSI 50-70, MACD > Signal
    - SELL: Price < EMA50, EMA20 < EMA50, RSI 30-50, MACD < Signal

    **Risk Management:**
    - SL = ATR × 1.5
    - TP = SL × 2 (R:R 1:2)
    - Max Daily DD: 3%
    - Max Total DD: 6%

    **Sessions:**
    - London: 08:00-16:00 UTC
    - New York: 13:00-21:00 UTC
                """)

    # Main content
    trades = dashboard.load_trades_history()
    stats = dashboard.calculate_stats(trades)
    balance = dashboard.get_account_balance()
    current_price = dashboard.get_current_price()
    open_position = dashboard.get_open_position()

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(dashboard.symbol,
                  f"${current_price:,.2f}" if current_price > 0 else "...")

    with col2:
        st.metric("Balance", f"${balance:,.2f}")

    with col3:
        pnl_delta = f"${stats['total_pnl']:+,.2f}" if stats['total_pnl'] != 0 else None
        st.metric("Total P/L", f"${stats['total_pnl']:,.2f}", delta=pnl_delta)

    with col4:
        st.metric("Win Rate", f"{stats['win_rate']:.1f}%",
                  delta=f"{stats['winning_trades']}/{stats['total_trades']}")

    with col5:
        st.metric("Total Trades", stats['total_trades'])

    st.divider()

    # Open position
    if open_position:
        st.markdown("### 🔥 Open Position")

        side = open_position.get('signal', 'BUY')
        card_class = '' if side == 'BUY' else 'sell'

        st.markdown(f'<div class="position-card {card_class}">', unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"**{'🟢' if side == 'BUY' else '🔴'} {side}**")
            st.caption("Direction")

        with col2:
            entry = open_position.get('entry_price', 0)
            st.markdown(f"**${entry:.2f}**")
            st.caption("Entry Price")

        with col3:
            sl = open_position.get('sl', 0)
            st.markdown(f"**${sl:.2f}**")
            st.caption("Stop Loss")

        with col4:
            tp = open_position.get('tp', 0)
            st.markdown(f"**${tp:.2f}**")
            st.caption("Take Profit")

        with col5:
            if current_price > 0 and entry > 0:
                if side == 'BUY':
                    unrealized_pnl = (current_price - entry) * open_position.get('quantity', 0)
                else:
                    unrealized_pnl = (entry - current_price) * open_position.get('quantity', 0)

                pnl_class = "metric-positive" if unrealized_pnl >= 0 else "metric-negative"
                st.markdown(f"<p class='{pnl_class}' style='font-size:1.2rem; margin:0;'>${unrealized_pnl:.2f}</p>",
                            unsafe_allow_html=True)
                st.caption("Unrealized P/L")

        st.markdown('</div>', unsafe_allow_html=True)

        # Position details
        try:
            entry_time = datetime.fromisoformat(open_position.get('timestamp'))
            duration = datetime.now() - entry_time
            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<span class="info-badge">⏰ Opened: {entry_time.strftime("%d.%m %H:%M")}</span>',
                            unsafe_allow_html=True)
            with col2:
                st.markdown(f'<span class="info-badge">⏱️ Duration: {hours}h {minutes}m</span>',
                            unsafe_allow_html=True)
            with col3:
                confidence = open_position.get('confidence', 0)
                st.markdown(f'<span class="info-badge">🎯 Confidence: {confidence * 100:.0f}%</span>',
                            unsafe_allow_html=True)
        except:
            pass

        st.divider()

    # Chart
    st.markdown("### 📈 Equity Curve")

    equity_fig = dashboard.plot_equity_curve(trades)
    if equity_fig:
        st.plotly_chart(equity_fig, use_container_width=True)
    else:
        st.info("📊 No trading history yet. Chart will appear after first trade.")

    st.divider()

    # Statistics
    st.markdown("### 📊 Performance Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Trading**")
        st.write(f"Total: **{stats['total_trades']}**")
        st.write(f"Wins: **{stats['winning_trades']}** ✅")
        st.write(f"Losses: **{stats['losing_trades']}** ❌")
        st.write(f"Win Rate: **{stats['win_rate']:.1f}%**")

    with col2:
        st.markdown("**Profit/Loss**")
        pnl_class = "metric-positive" if stats['total_pnl'] >= 0 else "metric-negative"
        st.markdown(f"<p class='{pnl_class}' style='font-size:1.3rem;'>${stats['total_pnl']:,.2f}</p>",
                    unsafe_allow_html=True)
        st.write(f"Best: **${stats['best_trade']:.2f}**")
        st.write(f"Worst: **${stats['worst_trade']:.2f}**")

    with col3:
        st.markdown("**Averages**")
        st.write(f"Avg Win: **${stats['avg_profit']:.2f}**")
        st.write(f"Avg Loss: **${stats['avg_loss']:.2f}**")
        st.write(f"Profit Factor: **{stats['profit_factor']:.2f}**")

    with col4:
        st.markdown("**Other**")
        avg_dur_min = int(stats['avg_duration'] / 60) if stats['avg_duration'] > 0 else 0
        st.write(f"Avg Duration: **{avg_dur_min}m**")
        st.write(f"Volume: **${stats['total_volume']:,.0f}**")
        st.write(f"Open: **{stats['open_trades']}**")

    st.divider()

    # Recent trades
    st.markdown("### 📋 Recent Trades (Last 20)")

    if trades:
        closed = [t for t in trades if t.get('action') == 'CLOSE'][-20:]

        if closed:
            df = pd.DataFrame(closed)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%d.%m %H:%M')

            df_display = pd.DataFrame({
                'Time': df['timestamp'],
                'Reason': df['reason'],
                'Entry': df['entry_price'].apply(lambda x: f"${x:.2f}"),
                'Exit': df['exit_price'].apply(lambda x: f"${x:.2f}"),
                'P/L': df['pnl'].apply(lambda x: f"${x:.2f}"),
                '%': df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
            })

            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("📭 No closed trades yet")
    else:
        st.info("🤖 Bot ready! Waiting for signals...")

    # Footer
    st.divider()
    st.markdown("""
            <div style='text-align: center; color: #94a3b8; padding: 15px;'>
                <p style='font-size: 0.85rem;'>⚠️ <strong>Risk Warning:</strong> Trading involves substantial risk. Only trade with capital you can afford to lose.</p>
                <p style='font-size: 0.8rem; margin-top: 8px;'>M1 Scalping Bot | Professional Trading System</p>
            </div>
        """, unsafe_allow_html=True)

    st.caption(f"Last update: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")


if __name__ == "__main__":
    main()

    # Auto-refresh every 10 seconds
    time.sleep(10)
    st.rerun()