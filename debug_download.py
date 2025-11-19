import ccxt
import pandas as pd
from datetime import datetime, timezone
import sys

print("Starting debug download...")
try:
    ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    print("Exchange initialized.")
    
    symbol = "BTCUSDT"
    timeframe = "1m"
    start = "2024-01-01"
    
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    print(f"Fetching {symbol} {timeframe} from {start_ts}...")
    
    candles = ex.fetch_ohlcv(symbol, timeframe, since=start_ts, limit=10)
    print(f"Fetched {len(candles)} candles.")
    print(candles[0])
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
