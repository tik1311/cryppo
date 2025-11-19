import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

# 1. Create 1h Data
dates_1h = pd.date_range(start="2024-01-01", end="2024-01-07", freq="1h")
df_1h = pd.DataFrame({
    "timestamp": dates_1h,
    "open": np.random.rand(len(dates_1h)) * 1000 + 40000,
    "high": np.random.rand(len(dates_1h)) * 1000 + 41000,
    "low": np.random.rand(len(dates_1h)) * 1000 + 39000,
    "close": np.random.rand(len(dates_1h)) * 1000 + 40000,
    "volume": np.random.rand(len(dates_1h)) * 100 + 100
})
df_1h.to_parquet("data/BTCUSDT_1h_2024-01-01_2024-01-07.parquet")
print(f"Created 1h data: {len(df_1h)} rows")

# 2. Create 1m Data
dates_1m = pd.date_range(start="2024-01-01", end="2024-01-07", freq="1min")
df_1m = pd.DataFrame({
    "timestamp": dates_1m,
    "open": np.random.rand(len(dates_1m)) * 1000 + 40000,
    "high": np.random.rand(len(dates_1m)) * 1000 + 41000,
    "low": np.random.rand(len(dates_1m)) * 1000 + 39000,
    "close": np.random.rand(len(dates_1m)) * 1000 + 40000,
    "volume": np.random.rand(len(dates_1m)) * 100 + 100
})
df_1m.to_parquet("data/BTCUSDT_1m_2024-01-01_2024-01-07.parquet")
print(f"Created 1m data: {len(df_1m)} rows")
