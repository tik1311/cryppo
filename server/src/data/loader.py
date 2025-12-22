"""
Data Loader for Market Data
Loads and preprocesses OHLCV data for training (Multi-Timeframe Support)

UPDATED: December 2025
- Proper train/test split BEFORE normalization (no data leakage)
- Uses OnlineScaler for consistent normalization
- Supports scaler saving/loading for live trading
"""

import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from .features import add_technical_indicators, add_minimal_indicators, normalize_features, OnlineScaler


def get_project_root() -> Path:
    """Get the project root directory (parent of server/)"""
    current_file = Path(__file__).resolve()
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
    """
    project_root = get_project_root()
    if not os.path.isabs(data_dir):
        data_dir = str(project_root / data_dir)

    print(f"[DEBUG] Looking for data in: {data_dir}")
    
    # Look for parquet files
    patterns = [
        f"{data_dir}/{symbol}_{timeframe}*.parquet",
        f"{data_dir}/**/{symbol}_{timeframe}*.parquet",
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
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
        raise FileNotFoundError(f"No data files found for {symbol} {timeframe} in {data_dir}. Please run 'python cryppo.py data download' first.")

    print(f"[INFO] Found {len(files)} data file(s) for {symbol} {timeframe}")

    # Load all files
    dfs = []
    for file in files:
        if file.endswith('.parquet'):
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(file)
        
        # Standardize timestamp column
        if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        if 'timestamp' not in df.columns:
            for col in df.columns:
                if col.lower() in ['time', 'datetime', 'date', 'open_time']:
                    df = df.rename(columns={col: 'timestamp'})
                    break
        
        dfs.append(df)

    if len(dfs) == 1:
        df = dfs[0]
    else:
        df = pd.concat(dfs, ignore_index=True)

    # Standardize columns
    column_mapping = {
        'time': 'timestamp', 'open_time': 'timestamp', 'datetime': 'timestamp',
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    }
    df = df.rename(columns=column_mapping)
    
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert timestamp
    if df['timestamp'].dtype == 'object' or df['timestamp'].dtype == 'int64':
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms' if df['timestamp'].dtype == 'int64' else None)

    # Sort and filter
    df = df.sort_values('timestamp')
    
    if start_date:
        df = df[df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['timestamp'] <= pd.to_datetime(end_date)]

    df = df.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
    
    return df[required_cols]


def load_multi_timeframe_data(
    symbol: str = "BTCUSDT",
    main_timeframe: str = "1h",
    detail_timeframe: str = "1m",
    data_dir: str = "data",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load synchronized data for two timeframes (Main + Detail)
    """
    print(f"\n[INFO] Loading Multi-Timeframe Data: {symbol} ({main_timeframe} + {detail_timeframe})")
    
    # Load both datasets
    df_main = load_market_data(data_dir, symbol, main_timeframe, start_date, end_date)
    df_detail = load_market_data(data_dir, symbol, detail_timeframe, start_date, end_date)
    
    print(f"  - Main ({main_timeframe}): {len(df_main)} candles")
    print(f"  - Detail ({detail_timeframe}): {len(df_detail)} candles")
    
    # Ensure detail data covers main data range
    min_ts = df_main['timestamp'].min()
    max_ts = df_main['timestamp'].max()
    
    df_detail = df_detail[(df_detail['timestamp'] >= min_ts) & (df_detail['timestamp'] <= max_ts + pd.Timedelta(main_timeframe))]
    
    return df_main, df_detail


def prepare_multi_timeframe_env_data(
    df_main: pd.DataFrame,
    df_detail: pd.DataFrame,
    main_timeframe: str = "1h",
    detail_timeframe: str = "1m",
    add_indicators: bool = True,
    normalize: bool = True,
    use_efficient_mode: bool = True,
    verbose: bool = True,
    scaler: Optional[OnlineScaler] = None,  # NEW: Use pre-fitted scaler for val/test
    return_scaler: bool = False  # NEW: Return fitted scaler for saving
) -> Dict[str, np.ndarray]:
    """
    Prepare nested data structure for environment

    UPDATED: Now properly handles normalization with OnlineScaler to prevent data leakage.

    Args:
        df_main: Main timeframe DataFrame
        df_detail: Detail timeframe DataFrame
        main_timeframe: Main timeframe string (e.g., '1h')
        detail_timeframe: Detail timeframe string (e.g., '1m')
        add_indicators: Whether to add technical indicators
        normalize: Whether to normalize features
        scaler: Pre-fitted OnlineScaler (for validation/test data)
        return_scaler: Whether to return the fitted scaler (for training data)

    Returns:
        Dict with:
        - 'main': [N, features] array
        - 'detail': [N, detail_len, features] array (nested sequences)
        - 'scaler': OnlineScaler (only if return_scaler=True)
    """
    if verbose:
        print("\n[INFO] Preprocessing Multi-Timeframe Data...")

    # 1. Add indicators
    if add_indicators:
        df_main = add_technical_indicators(df_main, verbose=verbose)
        df_detail = add_minimal_indicators(df_detail, verbose=verbose)
        df_main = df_main.dropna()
        df_detail = df_detail.dropna()

    # 2. Select features
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    main_indicator_cols = [
        'returns', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
        'atr', 'stoch_k', 'stoch_d', 'volume_ratio', 'volatility'
    ]

    main_cols = [c for c in (price_cols + main_indicator_cols) if c in df_main.columns]
    detail_minimal_cols = ['open', 'high', 'low', 'close', 'volume', 'atr']
    detail_cols = [c for c in detail_minimal_cols if c in df_detail.columns]

    if verbose:
        print(f"  - Main timeframe features: {len(main_cols)} ({', '.join(main_cols[:5])}...)")
        print(f"  - Detail timeframe features: {len(detail_cols)} (OHLCV + ATR only)")

    # 3. Normalize (UPDATED: Proper handling with OnlineScaler)
    fitted_scaler = None
    if normalize:
        norm_cols_main = [c for c in main_cols if c not in price_cols]
        norm_cols_detail = [c for c in detail_cols if c not in price_cols]

        if scaler is not None and scaler.fitted:
            # Use pre-fitted scaler (for validation/test)
            if verbose:
                print("  - Using pre-fitted scaler (no leakage)")
            df_main = scaler.transform(df_main)
            # Detail uses same scaler for consistency (ATR normalization)
            df_detail = scaler.transform(df_detail)
        else:
            # Create and fit new scaler (for training)
            if verbose:
                print("  - Fitting new scaler on this data")
            fitted_scaler = OnlineScaler(method='minmax')
            fitted_scaler.fit(df_main, norm_cols_main)
            df_main = fitted_scaler.transform(df_main)
            df_detail = fitted_scaler.transform(df_detail)

    # 4. Align Data
    tf_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
    main_mins = tf_map.get(main_timeframe, 60)
    detail_mins = tf_map.get(detail_timeframe, 1)
    expected_detail_len = main_mins // detail_mins

    if verbose:
        print(f"  - Mapping: 1 {main_timeframe} candle = {expected_detail_len} {detail_timeframe} candles")

    main_data = df_main[main_cols].values.astype(np.float32)
    n_main = len(df_main)
    n_detail_features = len(detail_cols)
    detail_data = np.zeros((n_main, expected_detail_len, n_detail_features), dtype=np.float32)

    main_timestamps = df_main['timestamp'].values
    detail_timestamps = df_detail['timestamp'].values
    detail_values = df_detail[detail_cols].values

    main_ts_int = main_timestamps.astype(np.int64)
    detail_ts_int = detail_timestamps.astype(np.int64)
    main_duration = pd.Timedelta(main_timeframe).value
    main_end_ts_int = main_ts_int + main_duration

    start_indices = np.searchsorted(detail_ts_int, main_ts_int)
    end_indices = np.searchsorted(detail_ts_int, main_end_ts_int)

    valid_rows = []
    for i in range(n_main):
        start = start_indices[i]
        end = end_indices[i]
        slice_data = detail_values[start:end]
        current_len = len(slice_data)

        if current_len == expected_detail_len:
            detail_data[i] = slice_data
            valid_rows.append(i)
        elif current_len > 0:
            if current_len > expected_detail_len:
                detail_data[i] = slice_data[:expected_detail_len]
            else:
                detail_data[i, :current_len] = slice_data
                detail_data[i, current_len:] = slice_data[-1]
            valid_rows.append(i)

    final_main = main_data[valid_rows]
    final_detail = detail_data[valid_rows]

    if verbose:
        print(f"  [OK] Aligned {len(final_main)}/{n_main} candles successfully.")

    result = {
        'main': final_main,
        'detail': final_detail
    }

    if return_scaler and fitted_scaler is not None:
        result['scaler'] = fitted_scaler

    return result


def split_train_test_multi(
    data: Dict[str, np.ndarray],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[Dict, Dict, Dict]:
    """Split multi-timeframe data dictionary"""
    n = len(data['main'])
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = {'main': data['main'][:train_end], 'detail': data['detail'][:train_end]}
    val_data = {'main': data['main'][train_end:val_end], 'detail': data['detail'][train_end:val_end]}
    test_data = {'main': data['main'][val_end:], 'detail': data['detail'][val_end:]}

    return train_data, val_data, test_data


def download_and_save_data(
    symbol: str,
    timeframe: str = "1h",
    start: str = "2023-01-01",
    end: str = "2024-12-31",
    exchange: str = "binance",
    save_dir: str = "data"
) -> pd.DataFrame:
    """Download data (Wrapper around CCXT)"""
    import ccxt
    from datetime import datetime, timezone

    project_root = get_project_root()
    if not os.path.isabs(save_dir):
        save_dir = str(project_root / save_dir)

    print(f"Downloading {symbol} {timeframe}...")
    
    ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    all_candles = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            candles = ex.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not candles: break
            all_candles.extend(candles)
            current_ts = candles[-1][0] + 1
            print(f"  Downloaded {len(all_candles)}...", end='\r')
        except Exception as e:
            print(f"Error: {e}")
            break
            
    if not all_candles:
        raise ValueError(f"No data downloaded for {symbol} {timeframe}. Check your internet connection or date range.")

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    os.makedirs(save_dir, exist_ok=True)
    filepath = f"{save_dir}/{symbol}_{timeframe}_{start}_{end}.parquet"
    df.to_parquet(filepath)
    print(f"\nSaved to {filepath}")
    return df
