import ccxt
import pandas as pd
from datetime import datetime, timezone
import sys
import os
import shutil
import glob

LOG_FILE = "debug_log.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

log("Starting debug/fix script...")

# 1. Cleanup Data Directory
data_dir = "data"
log(f"Cleaning {data_dir}...")
if os.path.exists(data_dir):
    for file in glob.glob(os.path.join(data_dir, "*")):
        try:
            os.remove(file)
            log(f"Deleted {file}")
        except Exception as e:
            log(f"Failed to delete {file}: {e}")
else:
    os.makedirs(data_dir)

# 2. Initialize Exchange
try:
    ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    log("Exchange initialized.")
except Exception as e:
    log(f"Failed to init exchange: {e}")
    sys.exit(1)

# 3. Download Function
def download(symbol, timeframe, start_str, end_str):
    log(f"Downloading {symbol} {timeframe} from {start_str} to {end_str}...")
    try:
        start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        all_candles = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            candles = ex.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not candles: break
            all_candles.extend(candles)
            current_ts = candles[-1][0] + 1
            log(f"  Fetched {len(candles)} candles...")
            
        if not all_candles:
            log("No data fetched.")
            return
            
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        filename = f"{data_dir}/{symbol}_{timeframe}_{start_str}_{end_str}.parquet"
        df.to_parquet(filename)
        log(f"Saved {len(df)} rows to {filename}")
        
    except Exception as e:
        log(f"Error downloading: {e}")
        import traceback
        log(traceback.format_exc())

# 4. Run Downloads
download("BTCUSDT", "1h", "2024-01-01", "2024-01-07")
download("BTCUSDT", "1m", "2024-01-01", "2024-01-02")

log("Done.")
