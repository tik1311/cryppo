"""
Real-time Data Fetcher for Live Paper Trading
Fetches data from Binance API in real-time
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
from .features import add_technical_indicators, add_minimal_indicators


class RealtimeFetcher:
    """
    Fetch real-time market data from Binance API
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        main_timeframe: str = "1h",
        detail_timeframe: str = "1m",
        window_size: int = 36
    ):
        self.symbol = symbol
        self.main_timeframe = main_timeframe
        self.detail_timeframe = detail_timeframe
        self.window_size = window_size
        self.base_url = "https://api.binance.com/api/v3"

        # Timeframe mapping to Binance API format
        self.tf_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h', '1d': '1d'
        }

        # Timeframe to minutes
        self.tf_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }

    def _fetch_klines(
        self,
        timeframe: str,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch klines (candlestick) data from Binance API

        Args:
            timeframe: Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: Number of candles to fetch (max 1000)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        endpoint = f"{self.base_url}/klines"
        params = {
            'symbol': self.symbol,
            'interval': self.tf_map[timeframe],
            'limit': min(limit, 1000)  # Binance API limit
        }

        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Parse response
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Select and convert columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

            return df

        except Exception as e:
            print(f"[ERROR] Failed to fetch data: {e}")
            return pd.DataFrame()

    def get_latest_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Get latest data for both main and detail timeframes

        Returns:
            Tuple of (main_df, detail_df)
        """
        # Fetch main timeframe (1h)
        # Need window_size + some buffer for indicators
        main_limit = self.window_size + 100
        df_main = self._fetch_klines(self.main_timeframe, limit=main_limit)

        if df_main.empty:
            return None, None

        # Fetch detail timeframe (1m)
        # Need enough 1m candles to cover window_size 1h candles
        detail_ratio = self.tf_minutes[self.main_timeframe] // self.tf_minutes[self.detail_timeframe]
        detail_limit = main_limit * detail_ratio
        df_detail = self._fetch_klines(self.detail_timeframe, limit=min(detail_limit, 1000))

        if df_detail.empty:
            return None, None

        # Add indicators
        df_main = add_technical_indicators(df_main, verbose=False)
        df_detail = add_minimal_indicators(df_detail, verbose=False)

        # Drop NaNs
        df_main = df_main.dropna()
        df_detail = df_detail.dropna()

        return df_main, df_detail

    def wait_for_candle_close(self, timeframe: str = "1h") -> datetime:
        """
        Wait for the next candle to close

        Args:
            timeframe: Timeframe to wait for (default: 1h)

        Returns:
            Timestamp when candle closed
        """
        tf_seconds = self.tf_minutes[timeframe] * 60

        # Calculate time until next candle close
        now = datetime.utcnow()
        seconds_since_epoch = now.timestamp()
        seconds_into_candle = seconds_since_epoch % tf_seconds
        seconds_until_close = tf_seconds - seconds_into_candle

        print(f"[INFO] Waiting {seconds_until_close:.0f}s for {timeframe} candle to close...")

        # Wait
        time.sleep(seconds_until_close + 1)  # +1 second buffer

        return datetime.utcnow()

    def is_candle_just_closed(self, timeframe: str = "1h", tolerance_seconds: int = 10) -> bool:
        """
        Check if a candle just closed (within tolerance)

        Args:
            timeframe: Timeframe to check
            tolerance_seconds: Tolerance in seconds (default: 10s)

        Returns:
            True if candle just closed
        """
        tf_seconds = self.tf_minutes[timeframe] * 60

        now = datetime.utcnow()
        seconds_since_epoch = now.timestamp()
        seconds_into_candle = seconds_since_epoch % tf_seconds

        # Check if we're within tolerance of candle close
        return seconds_into_candle <= tolerance_seconds
