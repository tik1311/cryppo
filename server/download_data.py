"""
Data Download Script for 1H Crypto Data
Downloads historical 1-hour candlestick data from Binance
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def load_config():
    """Load configuration from YAML"""
    config_path = os.path.join(os.path.dirname(__file__), 'src', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_binance_data(symbol: str, start_date: str, end_date: str, timeframe: str = '1h'):
    """
    Download OHLCV data from Binance

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        timeframe: Candle timeframe (default: '1h')

    Returns:
        DataFrame with OHLCV data
    """
    # Initialize Binance exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}  # Use futures for more data
    })

    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    print(f"Downloading {symbol} data from {start_date} to {end_date}")
    print(f"Timeframe: {timeframe}")

    all_candles = []
    current_ts = start_ts

    # Calculate total iterations for progress bar
    timeframe_ms = {
        '1h': 3600000,
        '1m': 60000,
        '5m': 300000,
        '15m': 900000,
        '30m': 1800000,
        '4h': 14400000,
        '1d': 86400000
    }

    ms_per_candle = timeframe_ms.get(timeframe, 3600000)
    limit = 1000  # Binance limit per request
    total_iterations = int((end_ts - start_ts) / (ms_per_candle * limit)) + 1

    with tqdm(total=total_iterations, desc=f"Downloading {symbol}") as pbar:
        while current_ts < end_ts:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=limit
                )

                if not candles:
                    break

                all_candles.extend(candles)
                current_ts = candles[-1][0] + ms_per_candle

                pbar.update(1)

            except Exception as e:
                print(f"\nError downloading data: {e}")
                print("Retrying...")
                continue

    # Convert to DataFrame
    df = pd.DataFrame(
        all_candles,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]

    # Sort by index
    df.sort_index(inplace=True)

    print(f"\nDownloaded {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    return df


def save_data(df: pd.DataFrame, symbol: str, output_dir: str, format: str = 'npz'):
    """
    Save data to file

    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol name for filename
        output_dir: Output directory
        format: File format ('npz' or 'csv')
    """
    os.makedirs(output_dir, exist_ok=True)

    # Clean symbol name for filename
    clean_symbol = symbol.replace('/', '').replace(':', '')

    if format == 'npz':
        # Save as compressed numpy array
        output_file = os.path.join(output_dir, f"{clean_symbol}_1h.npz")

        np.savez_compressed(
            output_file,
            timestamp=df.index.values,
            open=df['open'].values,
            high=df['high'].values,
            low=df['low'].values,
            close=df['close'].values,
            volume=df['volume'].values
        )

        print(f"Saved to: {output_file}")

    elif format == 'csv':
        # Save as CSV
        output_file = os.path.join(output_dir, f"{clean_symbol}_1h.csv")
        df.to_csv(output_file)
        print(f"Saved to: {output_file}")

    # Print file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Download 1H crypto data from Binance')
    parser.add_argument('--symbols', nargs='+', default=['BTC/USDT', 'ETH/USDT'],
                        help='Symbols to download (default: BTC/USDT ETH/USDT)')
    parser.add_argument('--start', type=str, default='2019-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='./data',
                        help='Output directory')
    parser.add_argument('--format', type=str, choices=['npz', 'csv'], default='npz',
                        help='Output format')

    args = parser.parse_args()

    print("=" * 60)
    print("RL Crypto Trading - Data Download")
    print("Timeframe: 1H Only")
    print("=" * 60)

    for symbol in args.symbols:
        try:
            df = download_binance_data(symbol, args.start, args.end, timeframe='1h')
            save_data(df, symbol, args.output, format=args.format)
            print()
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue


if __name__ == "__main__":
    main()
