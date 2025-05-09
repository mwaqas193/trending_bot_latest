import os
import time
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask
import threading

# Load environment variables
load_dotenv()

# Initialize Flask app for health checks
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Supertrend Bot is running", 200

def run_flask():
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 10000)))

RESULT_DIR = 'supertrend_result'
os.makedirs(RESULT_DIR, exist_ok=True)

class SupertrendBot:
    def __init__(
        self,
        mode: str = 'live',
        use_sandbox: bool = False,
        symbols: list = None,
        timeframes: list = None,
        max_positions: int = 3,
        position_size_pct: float = 0.99,
        position_size_usd: float = None,
        allocated_balance: float = None,
        discord_webhook_url: str = None,
        fee_pct: float = 0.001
    ):
        self.mode = mode
        self.use_sandbox = use_sandbox
        self.timeframes = timeframes or ['1d', '4h', '1h']
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.position_size_usd = position_size_usd
        self.allocated_balance = allocated_balance
        self.discord_webhook = discord_webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self.fee_pct = fee_pct

        STABLECOINS = {'USDT', 'BUSD', 'USDC', 'TUSD', 'DAI', 'PAX', 'USDP'}

        self.exchange = self._init_exchange()
        
        if symbols:
            raw = symbols
        else:
            markets = self.exchange.load_markets()
            raw = [
                s for s, m in markets.items()
                if m.get('quote') == 'USDT'
                and m.get('active')
                and m.get('base') not in STABLECOINS
            ]
        tickers = self.exchange.fetch_tickers()
        self.symbols = sorted(
            [s for s in raw if s in tickers],
            key=lambda s: tickers[s]['quoteVolume'],
            reverse=True
        )[:500]

        self.open_positions = {}
        self.closed_positions = []
        self.start_balance = None
        self.cycle_count = 0

    def _init_exchange(self):
        cfg = {
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'timeout': 30000
        }
        if self.mode == 'live':
            ex = ccxt.binance({
                **cfg,
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET'),
            })
            
            if self.use_sandbox:
                ex.set_sandbox_mode(True)
            try:
                ex.fetch_balance()
                print(f"[{datetime.now()}] ✅ Exchange connection successful")
                return ex
            except Exception as e:
                print(f"[{datetime.now()}] ❌ Exchange connection failed: {e}")
                raise
        else:
            return ccxt.binance(cfg)

    def calculate_supertrend(self, df, atr_period=10, multiplier=3):
        hl2 = (df['high'] + df['low']) / 2
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                (df['high'] - df['close'].shift(1)).abs(),
                (df['low'] - df['close'].shift(1)).abs()
            )
        )
        atr = tr.rolling(atr_period).mean()
        df['upper_band'] = hl2 + multiplier * atr
        df['lower_band'] = hl2 - multiplier * atr
        df['in_uptrend'] = True

        for i in range(1, len(df)):
            if df['close'].iat[i] > df['upper_band'].iat[i-1]:
                df.at[i, 'in_uptrend'] = True
            elif df['close'].iat[i] < df['lower_band'].iat[i-1]:
                df.at[i, 'in_uptrend'] = False
            else:
                df.at[i, 'in_uptrend'] = df['in_uptrend'].iat[i-1]
                if df.at[i, 'in_uptrend'] and df['lower_band'].iat[i] < df['lower_band'].iat[i-1]:
                    df.at[i, 'lower_band'] = df['lower_band'].iat[i-1]
                if not df.at[i, 'in_uptrend'] and df['upper_band'].iat[i] > df['upper_band'].iat[i-1]:
                    df.at[i, 'upper_band'] = df['upper_band'].iat[i-1]
        return df

    def compute_size(self, symbol: str, price: float) -> float:
        if self.position_size_usd is not None:
            usd = self.position_size_usd
        elif self.allocated_balance is not None:
            usd = self.allocated_balance / self.max_positions
        else:
            usd = self.exchange.fetch_balance()['free']['USDT'] * self.position_size_pct
        return usd / price

    def _print_open_positions(self):
        print("\n┌── Open Positions ────────────────────────")
        if not self.open_positions:
            print("│ None")
        else:
            for sym, pos in self.open_positions.items():
                curr = self.exchange.fetch_ticker(sym)['last']
                cost = pos['entry_price'] * pos['size']
                pnl = (curr - pos['entry_price']) * pos['size']
                roi = (pnl / cost * 100) if cost else 0
                print(f"│ {sym} | TF:{pos['entry_tf']} | Size:{pos['size']:.6f}")
                print(f"│ Entry:{pos['entry_price']:.4f} Curr:{curr:.4f} PnL:${pnl:.2f} ({roi:.2f}%)")
        print("└──────────────────────────────────────────")
   
    def _update_reports(self):
        start = self.start_balance or 0
        df = pd.DataFrame(self.closed_positions)

        overall = {
            'initial_balance': start,
            'current_balance': start + (df['pnl'].sum() if not df.empty else 0),
            'total_pnl': df['pnl'].sum() if not df.empty else 0,
            'return_pct': (df['pnl'].sum() / start * 100) if start else 0,
            'total_trades': len(df),
            'winning_trades': int((df['pnl'] > 0).sum()) if not df.empty else 0,
            'win_rate': (df['pnl'] > 0).mean() * 100 if not df.empty else 0,
            'avg_trade_duration': df['duration_min'].mean() if not df.empty else 0
        }
        pd.DataFrame([overall]).to_csv(os.path.join(RESULT_DIR, 'summary.csv'), index=False)

        if not df.empty:
            df = df.copy()
            df['size_usdt'] = df['entry_price'] * df['size']
            df['exit_usdt'] = df['exit_price'] * df['size']
            df['cumulative_pnl'] = df['pnl'].cumsum()
            df['cumulative_pnl_pct'] = df['cumulative_pnl'] / start * 100
            df.rename(columns={'fees': 'total_fees_usd'}, inplace=True)
            df['cumulative_fees_usd'] = df['total_fees_usd'].cumsum()

            col_order = [
                'symbol', 'entry_tf', 'entry_time', 'entry_price', 'size', 'size_usdt',
                'exit_time', 'exit_price', 'exit_usdt', 'pnl', 'cumulative_pnl', 
                'cumulative_pnl_pct', 'total_fees_usd', 'cumulative_fees_usd', 'duration_min'
            ]
            df[col_order].to_csv(os.path.join(RESULT_DIR, 'trades.csv'), index=False)

            df['exit_dt'] = pd.to_datetime(df['exit_time'], unit='ms')
            df['month'] = df['exit_dt'].dt.to_period('M').dt.to_timestamp()
            monthly = df.groupby('month').agg(
                total_pnl=('pnl', 'sum'),
                total_trades=('pnl', 'size'),
                winning_trades=('pnl', lambda x: int((x > 0).sum())),
                win_rate=('pnl', lambda x: float(x.gt(0).mean() * 100)),
                avg_trade_duration=('duration_min', 'mean'),
                total_fees_usd=('total_fees_usd', 'sum'),
            ).reset_index()
            monthly['cumulative_pnl'] = monthly['total_pnl'].cumsum()
            monthly['cumulative_pnl_pct'] = monthly['cumulative_pnl'] / start * 100
            monthly['cumulative_fees_usd'] = monthly['total_fees_usd'].cumsum()
            monthly['month'] = monthly['month'].dt.strftime('%Y-%m')
            monthly.to_csv(os.path.join(RESULT_DIR, 'monthly_summary.csv'), index=False)

            tf_df = df.copy()
            tf_summary = tf_df.groupby('entry_tf').agg(
                total_pnl_usd=('pnl', 'sum'),
                total_fees_usd=('total_fees_usd', 'sum'),
                total_trades=('pnl', 'size'),
                winning_trades=('pnl', lambda x: int((x > 0).sum())),
                losing_trades=('pnl', lambda x: int((x < 0).sum())),
                avg_trade_duration_min=('duration_min', 'mean'),
            ).reset_index().rename(columns={'entry_tf': 'timeframe'})

            order = {'1h': 0, '4h': 1, '1d': 2}
            tf_summary['order'] = tf_summary['timeframe'].map(order).fillna(99)
            tf_summary = tf_summary.sort_values('order').drop(columns='order')
            tf_summary['cumulative_pnl_usd'] = tf_summary['total_pnl_usd'].cumsum()
            tf_summary['cumulative_pnl_pct'] = tf_summary['cumulative_pnl_usd'] / start * 100
            tf_summary['cumulative_fees_usd'] = tf_summary['total_fees_usd'].cumsum()
            tf_summary['win_rate_pct'] = tf_summary['winning_trades'] / tf_summary['total_trades'] * 100

            tf_summary = tf_summary[
                ['timeframe', 'total_pnl_usd', 'cumulative_pnl_usd', 'cumulative_pnl_pct',
                 'total_fees_usd', 'cumulative_fees_usd', 'total_trades',
                 'winning_trades', 'losing_trades', 'win_rate_pct', 'avg_trade_duration_min']
            ]
            tf_summary.to_csv(os.path.join(RESULT_DIR, 'timeframe_summary.csv'), index=False)

    def send_discord_report(self):
        if not self.discord_webhook:
            return

        start = self.start_balance or 0
        df = pd.DataFrame(self.closed_positions)
        realised_pnl = df['pnl'].sum() if not df.empty and 'pnl' in df.columns else 0.0
        wins = int((df['pnl'] > 0).sum()) if not df.empty else 0
        losses = len(df) - wins if not df.empty else 0

        unrealised_pnl = 0.0
        for pos in self.open_positions.values():
            curr_price = self.exchange.fetch_ticker(pos['symbol'])['last']
            pnl = (curr_price - pos['entry_price']) * pos['size']
            unrealised_pnl += pnl

        current = start + realised_pnl
        content = (
            f"**Supertrend Bot Report**\n\n"
            f"**Balance:** ${start:.2f} → ${current:.2f} ({realised_pnl:+.2f})\n"
            f"**Performance:** {((current - start) / start * 100):.2f}%\n"
            f"**Realised PnL:** ${realised_pnl:.2f}\n"
            f"**Unrealised PnL:** ${unrealised_pnl:.2f}\n"
            f"**Trades:** {len(df)} (W:{wins}, L:{losses})\n"
            f"**Open Positions:** {len(self.open_positions)}/{self.max_positions}\n"
        )

        if self.open_positions:
            content += "\n**Positions Detail:**\n"
            for sym, pos in self.open_positions.items():
                curr = self.exchange.fetch_ticker(sym)['last']
                cost = pos['entry_price'] * pos['size']
                pnl = (curr - pos['entry_price']) * pos['size']
                roi = (pnl / cost * 100) if cost else 0
                content += (
                    f"• {sym} | TF:{pos['entry_tf']} | Size:{pos['size']:.6f}\n"
                    f"  Entry:{pos['entry_price']:.4f} Curr:{curr:.4f} "
                    f"PnL:${pnl:.2f} ({roi:.2f}%)\n"
                )

        try:
            requests.post(self.discord_webhook, json={'content': content}, timeout=10)
            print(f"[{datetime.now()}] 📢 Discord report sent")
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Discord report failed: {e}")

    def _trade_cycle(self):
        self.cycle_count += 1
        start_ts = time.time()
        print(f"\n[{datetime.now()}] 🔄 Trade Cycle #{self.cycle_count}")

        for sym, pos in list(self.open_positions.items()):
            try:
                o = self.exchange.fetch_ohlcv(sym, pos['entry_tf'], limit=50)
                df = pd.DataFrame(o, columns=['ts','open','high','low','close','vol'])
                df = self.calculate_supertrend(df)
                prev, curr = df.iloc[-2], df.iloc[-1]
                if prev['in_uptrend'] and not curr['in_uptrend']:
                    size = pos['size']
                    cost = pos['entry_price'] * size
                    exit_val = curr['close'] * size
                    fees = (cost + exit_val) * self.fee_pct
                    pnl = exit_val - cost - fees
                    if self.mode == 'live':
                        try:
                            self.exchange.create_market_sell_order(sym, size)
                            print(f"  ✅ Sold {sym} @ {curr['close']:.4f}")
                        except Exception as e:
                            print(f"  ❌ Sell failed {sym}: {e}")
                            continue
                    trade = {
                        **pos,
                        'exit_price': curr['close'],
                        'exit_time': curr['ts'],
                        'pnl': pnl,
                        'duration_min': (curr['ts'] - pos['entry_time']) / 60000,
                        'fees': fees
                    }
                    self.closed_positions.append(trade)
                    del self.open_positions[sym]
                    print(f"  ⏹ Closed {sym} PnL:${pnl:.2f}")
            except Exception as e:
                print(f"  ❌ Error closing {sym}: {e}")

        open_cnt = len(self.open_positions)
        if open_cnt < self.max_positions:
            needed = self.max_positions - open_cnt
            print(f"  🔍 Scanning for buys ({open_cnt}/{self.max_positions}) – need {needed}")
            tasks = [
                (s, tf)
                for s in self.symbols
                if s not in self.open_positions
                for tf in self.timeframes
            ]
            signals = []
            def check(pair):
                s, tf = pair
                try:
                    ohlcv = self.exchange.fetch_ohlcv(s, tf, limit=50)
                    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','vol'])
                    df = self.calculate_supertrend(df)
                    prev, curr = df.iloc[-2], df.iloc[-1]
                    if not prev['in_uptrend'] and curr['in_uptrend']:
                        return (s, tf, curr['close'], curr['ts'])
                except Exception:
                    pass
                return None

            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = {ex.submit(check, t): t for t in tasks}
                for fut in as_completed(futures):
                    r = fut.result()
                    if r:
                        signals.append(r)
                        if len(signals) >= needed:
                            break

            if signals:
                tf_priority = {'1d': 0, '4h': 1, '1h': 2}
                signals.sort(key=lambda sig: tf_priority.get(sig[1], 99))
                signals = signals[:needed]
                pretty = ", ".join(f"{s}@{tf}" for s,tf,_,_ in signals)
                print(f"  ✅ Picking top {len(signals)} by timeframe: {pretty}")
                for sym, tf, price, ts in signals:
                    size = self.compute_size(sym, price)
                    if self.mode == 'live':
                        try:
                            self.exchange.create_market_buy_order(sym, size)
                            print(f"  ✅ Bought {sym}@{tf} @ {price:.4f}")
                        except Exception as e:
                            print(f"  ❌ Buy failed {sym}: {e}")
                            continue
                    self.open_positions[sym] = {
                        'symbol': sym,
                        'entry_price': price,
                        'size': size,
                        'entry_tf': tf,
                        'entry_time': ts
                    }
            else:
                print("  ⚠️ No buy signals this cycle")
        else:
            print(f"  🚫 Max positions reached ({self.max_positions})")

        self._update_reports()
        self._print_open_positions()
        print(f"[{datetime.now()}] 🔄 Cycle done in {time.time() - start_ts:.2f}s")

    def run_live(self):
        try:
            if self.allocated_balance is not None:
                self.start_balance = self.allocated_balance
            else:
                self.start_balance = self.exchange.fetch_balance()['free']['USDT']
            print(f"\n[{datetime.now()}] 🚀 Starting Supertrend Bot")
            print(f"  Mode: {'SANDBOX' if self.use_sandbox else 'LIVE'}")
            print(f"  Symbols: {len(self.symbols)} | Timeframes: {', '.join(self.timeframes)}")
            print(f"  Max Positions: {self.max_positions}")
            print(f"  Start Balance: ${self.start_balance:.2f}")
            print(f"  Discord: {'ENABLED' if self.discord_webhook else 'DISABLED'}")

            last_cycle = time.time()
            last_report = time.time()
            last_hb = time.time()

            self._trade_cycle()
            print(f"\n[{datetime.now()}] 🏁 Bot is running. Ctrl+C to stop.\n")

            while True:
                now = time.time()
                if now - last_cycle >= 60:
                    try:
                        self._trade_cycle()
                    except Exception as e:
                        print(f"[{datetime.now()}] ❌ Cycle error: {e}")
                    last_cycle = now
                if self.discord_webhook and now - last_report >= 300:
                    try:
                        self.send_discord_report()
                    except Exception as e:
                        print(f"[{datetime.now()}] ❌ Report error: {e}")
                    last_report = now
                if now - last_hb >= 30:
                    nt = max(0, 60 - (now - last_cycle))
                    nr = max(0, 300 - (now - last_report))
                    print(f"[{datetime.now()}] ♻️ Next cycle in {int(nt)}s | Next report in {int(nr)}s")
                    last_hb = now
                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n[{datetime.now()}] ⏸ Stopping…")
        except Exception as e:
            print(f"\n[{datetime.now()}] ❌ Fatal: {e}")
        finally:
            print(f"[{datetime.now()}] 📊 Final reports…")
            self._update_reports()
            if self.discord_webhook:
                try:
                    self.send_discord_report()
                except:
                    pass
            print(f"[{datetime.now()}] 🛑 Bot stopped")

if __name__ == '__main__':
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
     
    
    # Initialize and run the trading bot
    bot = SupertrendBot(
        mode='live',
        use_sandbox=True,
        max_positions=3,
        allocated_balance=5000,
        fee_pct=0.001,
        discord_webhook_url=os.getenv('DISCORD_WEBHOOK_URL')
    )
    bot.run_live()  