"""
Data Loader for Market Data
Loads and preprocesses OHLCV data for training
"""

import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path
from typing import Optional, Tuple
from .features import add_technical_indicators, normalize_features


def get_project_root() -> Path:
    """Get the project root directory (parent of server/)"""
    current_file = Path(__file__).resolve()
    # __file__ is in: project/server/src/data/loader.py
    # So go up 3 levels to get to project root
    return current_file.parent.parent.parent.parent


def load_market_data(
    data_dir: str = "data",
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load market data from parquet files

    Args:
        data_dir: Directory containing market data (relative to project root)
        symbol: Trading pair symbol
        timeframe: Timeframe (1h, 4h, etc.)
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    # Convert data_dir to absolute path relative to project root
    project_root = get_project_root()
    if not os.path.isabs(data_dir):
        data_dir = str(project_root / data_dir)

    print(f"[DEBUG] Looking for data in: {data_dir}")
    print(f"[DEBUG] Pattern: {symbol}_{timeframe}*.parquet")

    # Look for parquet files (try both direct and recursive)
    patterns = [
        f"{data_dir}/{symbol}_{timeframe}*.parquet",  # Direct in data/
        f"{data_dir}/**/{symbol}_{timeframe}*.parquet",  # In subdirectories
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    # Remove duplicates
    files = list(set(files))

    if not files:
        # Try CSV files
        patterns = [
            f"{data_dir}/{symbol}_{timeframe}*.csv",
            f"{data_dir}/**/{symbol}_{timeframe}*.csv",
        ]
        for pattern in patterns:
            files.extend(glob.glob(pattern, recursive=True))
        files = list(set(files))

    if not files:
        raise FileNotFoundError(
            f"[ERROR] No data files found for {symbol} {timeframe} in {data_dir}\n"
            f"Please download data first using the API:\n"
            f"  POST /api/data/download\n"
            f"  {{'symbol': '{symbol}', 'timeframe': '{timeframe}', 'start': '2023-01-01', 'end': '2024-12-31'}}\n\n"
            f"[INFO] To download real market data:\n"
            f"   1. Start the server: uvicorn server.main:app\n"
            f"   2. Download data via API:\n"
            f"      POST /api/data/download\n"
            f"      {{'symbol': '{symbol}', 'timeframe': '{timeframe}', 'start': '2023-01-01', 'end': '2024-12-31'}}\n\n"
            f"   Or use --sample-data flag to train with synthetic data"
        )

    print(f"[INFO] Found {len(files)} data file(s) for {symbol} {timeframe}")
    for f in files:
        print(f"  - {f}")

    # Load all files
    dfs = []
    for file in files:
        print(f"  [INFO] Loading: {file}")
        if file.endswith('.parquet'):
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(file)

        # If timestamp is in index, reset it to column
        if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            print(f"    [DEBUG] Timestamp found in index, resetting to column")
            df = df.reset_index()
            # After reset_index(), the index becomes a column
            # Check if it needs to be renamed
            if df.columns[0] not in ['timestamp', 'time', 'datetime']:
                print(f"    [DEBUG] Renaming column '{df.columns[0]}' to 'timestamp'")
                df = df.rename(columns={df.columns[0]: 'timestamp'})
        elif 'timestamp' not in df.columns:
            # Try to find timestamp column with different names
            timestamp_cols = [col for col in df.columns if col.lower() in ['time', 'datetime', 'date', 'open_time']]
            if timestamp_cols:
                print(f"    [DEBUG] Found timestamp column: '{timestamp_cols[0]}', renaming to 'timestamp'")
                df = df.rename(columns={timestamp_cols[0]: 'timestamp'})

        dfs.append(df)

    # Combine all dataframes
    if len(dfs) == 1:
        df = dfs[0]
    else:
        df = pd.concat(dfs, ignore_index=True)

    # Ensure proper columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    print(f"  [DEBUG] Columns before mapping: {list(df.columns)}")

    # Try to map common column names
    column_mapping = {
        'time': 'timestamp',
        'open_time': 'timestamp',
        'datetime': 'timestamp',
        'date': 'timestamp',
        'index': 'timestamp',  # From reset_index()
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }

    df = df.rename(columns=column_mapping)

    print(f"  [DEBUG] Columns after mapping: {list(df.columns)}")

    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  [ERROR] Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert timestamp to datetime
    if df['timestamp'].dtype == 'object' or df['timestamp'].dtype == 'int64':
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms' if df['timestamp'].dtype == 'int64' else None)

    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Filter by date range
    if start_date:
        df = df[df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['timestamp'] <= pd.to_datetime(end_date)]

    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'], keep='last')

    # Reset index
    df = df.reset_index(drop=True)

    print(f"  [OK] Data loaded successfully!")
    print(f"    - Shape: {df.shape}")
    print(f"    - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"    - Columns: {list(df[required_cols].columns)}\n")

    return df[required_cols]


def prepare_env_data(
    df: pd.DataFrame,
    add_indicators: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Prepare data for trading environment

    Args:
        df: DataFrame with OHLCV data
        add_indicators: Whether to add technical indicators
        normalize: Whether to normalize features

    Returns:
        Numpy array suitable for trading environment
    """
    if add_indicators:
        # Add technical indicators
        df = add_technical_indicators(df)

        # Drop NaN values from indicator calculations
        df = df.dropna()

    price_cols = ['open', 'high', 'low', 'close', 'volume']

    # Select features for environment
    if add_indicators:
        indicator_cols = [
            'returns', 'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'stoch_k', 'stoch_d',
            'volume_ratio', 'volatility'
        ]
        feature_cols = price_cols + indicator_cols
    else:
        feature_cols = price_cols

    feature_cols = [col for col in feature_cols if col in df.columns]

    # Normalize only non-price features
    if normalize and add_indicators:
        indicator_norm_cols = [col for col in feature_cols if col not in price_cols]
        df = normalize_features(df, indicator_norm_cols, method='minmax')

    data = df[feature_cols].values
    data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
    return data.astype(np.float32)


def split_train_test(
    data: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/val/test sets

    Args:
        data: Full dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set

    Returns:
        (train_data, val_data, test_data)
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def create_sample_data(
    symbol: str = "BTCUSDT",
    n_samples: int = 10000,
    save_dir: str = "data"
) -> pd.DataFrame:
    """
    Create sample synthetic data for testing
    (Use real data from API for actual training!)

    Args:
        symbol: Symbol name
        n_samples: Number of samples
        save_dir: Directory to save data (relative to project root)

    Returns:
        DataFrame with synthetic OHLCV data
    """
    # Convert save_dir to absolute path relative to project root
    project_root = get_project_root()
    if not os.path.isabs(save_dir):
        save_dir = str(project_root / save_dir)

    print(f"Creating {n_samples} samples of synthetic data...")
    print("! WARNING: This is synthetic data for testing only!")
    print("  Use real market data for actual training.\n")

    # Generate timestamps (1 hour intervals)
    start_time = pd.Timestamp('2023-01-01')
    timestamps = [start_time + pd.Timedelta(hours=i) for i in range(n_samples)]

    # Generate price data with realistic patterns
    initial_price = 30000.0
    prices = [initial_price]

    for i in range(1, n_samples):
        # Random walk with drift
        change = np.random.randn() * 100 + 0.5  # Slight upward bias
        new_price = max(prices[-1] + change, 100)  # Prevent negative prices
        prices.append(new_price)

    # Create OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        volatility = np.random.uniform(0.001, 0.01)
        high = close * (1 + abs(np.random.randn()) * volatility)
        low = close * (1 - abs(np.random.randn()) * volatility)
        open_price = (high + low) / 2 + np.random.randn() * volatility * close
        volume = np.random.uniform(100, 1000)

        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data)

    # Save to parquet
    os.makedirs(save_dir, exist_ok=True)
    filepath = f"{save_dir}/{symbol}_{n_samples}_synthetic.parquet"
    df.to_parquet(filepath, index=False)
    print(f"[OK] Saved synthetic data to: {filepath}")

    return df


def download_and_save_data(
    symbol: str,
    timeframe: str = "1h",
    start: str = "2023-01-01",
    end: str = "2024-12-31",
    exchange: str = "binance",
    save_dir: str = "data"
) -> pd.DataFrame:
    """
    Download market data from exchange and save to parquet

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        timeframe: Candle timeframe (1h, 4h, 1d)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        exchange: Exchange name (binance, etc.)
        save_dir: Directory to save data (relative to project root)

    Returns:
        DataFrame with OHLCV data
    """
    import ccxt
    from datetime import datetime, timezone

    # Convert save_dir to absolute path relative to project root
    project_root = get_project_root()
    if not os.path.isabs(save_dir):
        save_dir = str(project_root / save_dir)

    print(f"\nDownloading {symbol} {timeframe} data from {exchange}...")
    print(f"Period: {start} to {end}")
    print(f"[DEBUG] Will save to: {save_dir}")

    # Initialize exchange
    if exchange.lower() == "binance":
        ex = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}  # Use futures for more liquidity
        })
    else:
        raise ValueError(f"Exchange {exchange} not supported yet")

    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    # Download data in chunks
    all_candles = []
    current_ts = start_ts

    while current_ts < end_ts:
        try:
            candles = ex.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_ts,
                limit=1000  # Max per request
            )

            if not candles:
                break

            all_candles.extend(candles)
            current_ts = candles[-1][0] + 1  # Move to next timestamp

            print(f"  Downloaded {len(all_candles)} candles...", end='\r')

        except Exception as e:
            print(f"\n  Error downloading: {e}")
            break

    print(f"\n  Total candles downloaded: {len(all_candles)}")

    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]

    # Sort by timestamp
    df.sort_index(inplace=True)

    # Save to parquet
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{symbol}_{timeframe}_{start}_{end}.parquet"
    filepath = os.path.join(save_dir, filename)
    df.to_parquet(filepath)

    print(f"  [OK] Saved to: {filepath}")
    print(f"  Shape: {df.shape}")
    print(f"  Period: {df.index[0]} to {df.index[-1]}\n")

    return df
