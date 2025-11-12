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
from .features import add_technical_indicators


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
        data_dir: Directory containing market data
        symbol: Trading pair symbol
        timeframe: Timeframe (1h, 4h, etc.)
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    # Look for parquet files
    pattern = f"{data_dir}/**/{symbol}_{timeframe}*.parquet"
    files = glob.glob(pattern, recursive=True)

    if not files:
        # Try CSV files
        pattern = f"{data_dir}/**/{symbol}_{timeframe}*.csv"
        files = glob.glob(pattern, recursive=True)

    if not files:
        raise FileNotFoundError(
            f"No data files found for {symbol} {timeframe} in {data_dir}\n"
            f"Please download data first using the API:\n"
            f"  POST /api/data/download\n"
            f"  {{'symbol': '{symbol}', 'timeframe': '{timeframe}', 'start': '2023-01-01', 'end': '2024-12-31'}}"
        )

    # Load all files
    dfs = []
    for file in files:
        if file.endswith('.parquet'):
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(file)
        dfs.append(df)

    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True)

    # Ensure proper columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    # Try to map common column names
    column_mapping = {
        'time': 'timestamp',
        'open_time': 'timestamp',
        'datetime': 'timestamp',
        'date': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }

    df = df.rename(columns=column_mapping)

    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
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

    # Select features for environment
    # Basic OHLCV + key indicators
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'returns', 'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_middle', 'bb_lower',
        'atr', 'stoch_k', 'stoch_d',
        'volume_ratio', 'volatility'
    ] if add_indicators else ['open', 'high', 'low', 'close', 'volume']

    # Filter to available columns
    feature_cols = [col for col in feature_cols if col in df.columns]

    # Extract features as numpy array
    data = df[feature_cols].values

    # Handle any remaining NaN or inf values
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
        save_dir: Directory to save data

    Returns:
        DataFrame with synthetic OHLCV data
    """
    print(f"Creating {n_samples} samples of synthetic data...")
    print("⚠️  WARNING: This is synthetic data for testing only!")
    print("   Use real market data for actual training.\n")

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
    print(f"✓ Saved synthetic data to: {filepath}")

    return df
