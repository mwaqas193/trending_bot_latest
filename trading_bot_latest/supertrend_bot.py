import os
import time
import requests
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
from flask import Flask, jsonify

load_dotenv()

RESULT_DIR = 'supertrend_result'
os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__)

@app.route('/healthz')
def healthz():
    return jsonify(status='ok')

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

        # stablecoin tickers to exclude as bases
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

    # ... (all your existing methods: calculate_supertrend, compute_size, _print_open_positions,
    #      _update_reports, send_discord_report, _trade_cycle remain unchanged) ...

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

            self._trade_cycle()
            print(f"\n[{datetime.now()}] 🏁 Bot is running. Ctrl+C to stop.\n")

            last_cycle = time.time()
            last_report = time.time()
            last_hb = time.time()

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


def start_bot():
    bot = SupertrendBot(
        mode='live',
        use_sandbox=True,
        max_positions=3,
        allocated_balance=5000,
        fee_pct=0.001,
        discord_webhook_url=os.getenv('DISCORD_WEBHOOK_URL')
    )
    bot.run_live()

if __name__ == '__main__':
    # Launch the bot in a background thread
    Thread(target=start_bot, daemon=True).start()
    # Bind Flask server to Render's PORT
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port) 
 