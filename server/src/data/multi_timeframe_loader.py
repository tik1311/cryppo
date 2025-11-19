"""
Multi-Timeframe Data Loader for Hierarchical RL Trading
Phase 6: Combines primary timeframe (5m/15m/1h) with 1m detail candles

Key Concept:
- 1m candles are NOT separate data, but DETAILS within each primary candle
- Example: 1 candle 5m contains 5 candles 1m as microstructure
- Example: 1 candle 1h contains 60 candles 1m as microstructure
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
import ccxt


class MultiTimeframeLoader:
    """
    Loads and aligns multi-timeframe data for hierarchical RL training

    Structure:
        Primary TF (e.g., 5m): [N, features] - Trading timeframe
        Detail TF (1m): [N, detail_ratio, features] - Microstructure within each primary candle

    Where detail_ratio = primary_timeframe / 1m
    - 5m: 5 candles 1m per primary candle
    - 15m: 15 candles 1m per primary candle
    - 1h: 60 candles 1m per primary candle
    """

    TIMEFRAME_MINUTES = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }

    def __init__(
        self,
        primary_timeframe: str = '5m',
        detail_timeframe: str = '1m',
        window_size: int = 72
    ):
        """
        Initialize Multi-Timeframe Loader

        Args:
            primary_timeframe: Main trading timeframe (5m, 15m, 1h)
            detail_timeframe: Detail timeframe for microstructure (1m)
            window_size: Number of primary candles in observation window
        """
        self.primary_tf = primary_timeframe
        self.detail_tf = detail_timeframe
        self.window_size = window_size

        # Calculate detail ratio
        primary_min = self.TIMEFRAME_MINUTES[primary_timeframe]
        detail_min = self.TIMEFRAME_MINUTES[detail_timeframe]

        if primary_min % detail_min != 0:
            raise ValueError(f"Primary TF {primary_timeframe} must be divisible by detail TF {detail_timeframe}")

        self.detail_ratio = primary_min // detail_min
        print(f"[MultiTF] Primary: {primary_timeframe}, Detail: {detail_timeframe}, Ratio: {self.detail_ratio}:1")
        print(f"[MultiTF] Each {primary_timeframe} candle contains {self.detail_ratio} x {detail_timeframe} candles")

    def download_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        data_dir: Path = Path("data")
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download both primary and detail timeframe data

        Returns:
            primary_df: Primary timeframe OHLCV data
            detail_df: Detail timeframe OHLCV data
        """
        exchange = ccxt.binance({'enableRateLimit': True})

        # Download primary timeframe
        print(f"\n[Download] {symbol} {self.primary_tf} from {start_date} to {end_date}")
        primary_ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe=self.primary_tf,
            since=exchange.parse8601(f"{start_date}T00:00:00Z"),
            limit=1000
        )

        # Download detail timeframe
        print(f"[Download] {symbol} {self.detail_tf} from {start_date} to {end_date}")
        detail_ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe=self.detail_tf,
            since=exchange.parse8601(f"{start_date}T00:00:00Z"),
            limit=1000
        )

        # Convert to DataFrames
        primary_df = pd.DataFrame(
            primary_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        primary_df['timestamp'] = pd.to_datetime(primary_df['timestamp'], unit='ms')

        detail_df = pd.DataFrame(
            detail_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        detail_df['timestamp'] = pd.to_datetime(detail_df['timestamp'], unit='ms')

        # Save to parquet
        data_dir.mkdir(parents=True, exist_ok=True)
        primary_path = data_dir / f"{symbol.replace('/', '_')}_{self.primary_tf}.parquet"
        detail_path = data_dir / f"{symbol.replace('/', '_')}_{self.detail_tf}.parquet"

        primary_df.to_parquet(primary_path)
        detail_df.to_parquet(detail_path)

        print(f"[Saved] Primary: {len(primary_df)} candles → {primary_path}")
        print(f"[Saved] Detail: {len(detail_df)} candles → {detail_path}")

        return primary_df, detail_df

    def align_timeframes(
        self,
        primary_df: pd.DataFrame,
        detail_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align detail candles within primary candles

        Returns:
            primary_data: [N, features] - Primary timeframe data
            detail_data: [N, detail_ratio, features] - Detail candles within each primary

        Example for 5m primary, 1m detail:
            primary_data[100] = 5m candle at index 100
            detail_data[100] = [5, features] = 5x 1m candles inside that 5m candle
        """
        print(f"\n[Align] Aligning {self.detail_tf} candles within {self.primary_tf} candles")

        # Convert to numpy
        primary_data = primary_df[['open', 'high', 'low', 'close', 'volume']].values
        detail_values = detail_df[['open', 'high', 'low', 'close', 'volume']].values

        # Create aligned detail data structure
        n_primary = len(primary_df)
        n_features = 5  # OHLCV

        detail_data = np.zeros((n_primary, self.detail_ratio, n_features))

        # For each primary candle, find corresponding detail candles
        for i in range(n_primary):
            primary_ts = primary_df.iloc[i]['timestamp']

            # Find detail candles within this primary candle's time window
            # Detail candles should fall within [primary_start, primary_start + primary_period)
            detail_start_idx = i * self.detail_ratio
            detail_end_idx = (i + 1) * self.detail_ratio

            if detail_end_idx <= len(detail_values):
                detail_data[i] = detail_values[detail_start_idx:detail_end_idx]
            else:
                # Pad with last available data if insufficient detail candles
                available = detail_values[detail_start_idx:]
                detail_data[i, :len(available)] = available
                if len(available) < self.detail_ratio:
                    # Repeat last candle to fill
                    detail_data[i, len(available):] = available[-1]

        print(f"[Aligned] Primary: {primary_data.shape}, Detail: {detail_data.shape}")
        print(f"[Aligned] Each primary candle has {self.detail_ratio} detail candles")

        return primary_data, detail_data

    def add_technical_indicators(
        self,
        primary_data: np.ndarray,
        detail_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add technical indicators to both timeframes

        Primary TF: Full set of indicators (RSI, MACD, Bollinger, ATR, etc.)
        Detail TF: Simpler indicators (momentum, volume-weighted price, spread)
        """
        # Primary indicators (same as before)
        primary_enriched = self._add_primary_indicators(primary_data)

        # Detail indicators (simpler, focused on microstructure)
        n_primary, detail_ratio, _ = detail_data.shape
        detail_enriched = np.zeros((n_primary, detail_ratio, 8))  # OHLCV + 3 indicators

        for i in range(n_primary):
            detail_candles = detail_data[i]  # [detail_ratio, 5]
            detail_enriched[i] = self._add_detail_indicators(detail_candles)

        return primary_enriched, detail_enriched

    def _add_primary_indicators(self, data: np.ndarray) -> np.ndarray:
        """Add full technical indicators for primary timeframe"""
        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()

        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_std'] = df['volume'].rolling(20).std()

        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Fill NaN
        df = df.fillna(method='bfill').fillna(0)

        return df.values

    def _add_detail_indicators(self, detail_candles: np.ndarray) -> np.ndarray:
        """
        Add microstructure indicators for detail timeframe
        Focused on: momentum, spread, volume profile within primary candle
        """
        df = pd.DataFrame(detail_candles, columns=['open', 'high', 'low', 'close', 'volume'])

        # Price momentum within primary candle
        df['momentum'] = df['close'].pct_change().fillna(0)

        # Volume-weighted price
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap'] = df['vwap'].fillna(df['close'])

        # Spread (high-low range as % of close)
        df['spread'] = (df['high'] - df['low']) / df['close']
        df['spread'] = df['spread'].fillna(0)

        return df[['open', 'high', 'low', 'close', 'volume', 'momentum', 'vwap', 'spread']].values

    def prepare_for_training(
        self,
        primary_df: pd.DataFrame,
        detail_df: pd.DataFrame,
        normalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Prepare multi-timeframe data for RL training

        Returns:
            {
                'primary': [N, features] - Primary timeframe with indicators
                'detail': [N, detail_ratio, detail_features] - Detail candles within primary
                'primary_norm_params': {mean, std} - For denormalization
                'detail_norm_params': {mean, std}
            }
        """
        # Align timeframes
        primary_data, detail_data = self.align_timeframes(primary_df, detail_df)

        # Add indicators
        primary_enriched, detail_enriched = self.add_technical_indicators(
            primary_data, detail_data
        )

        # Normalize if requested
        if normalize:
            primary_mean = np.mean(primary_enriched, axis=0)
            primary_std = np.std(primary_enriched, axis=0) + 1e-8
            primary_normalized = (primary_enriched - primary_mean) / primary_std

            # Normalize detail (across all detail candles)
            detail_flat = detail_enriched.reshape(-1, detail_enriched.shape[-1])
            detail_mean = np.mean(detail_flat, axis=0)
            detail_std = np.std(detail_flat, axis=0) + 1e-8
            detail_normalized = (detail_enriched - detail_mean) / detail_std
        else:
            primary_normalized = primary_enriched
            detail_normalized = detail_enriched
            primary_mean = primary_std = None
            detail_mean = detail_std = None

        print(f"\n[Prepared] Primary: {primary_normalized.shape}")
        print(f"[Prepared] Detail: {detail_normalized.shape}")
        print(f"[Prepared] Ready for multi-timeframe RL training")

        return {
            'primary': primary_normalized,
            'detail': detail_normalized,
            'primary_norm_params': {'mean': primary_mean, 'std': primary_std},
            'detail_norm_params': {'mean': detail_mean, 'std': detail_std}
        }


if __name__ == '__main__':
    # Test multi-timeframe loader
    loader = MultiTimeframeLoader(
        primary_timeframe='5m',
        detail_timeframe='1m',
        window_size=72
    )

    # Download data
    primary_df, detail_df = loader.download_data(
        symbol='BTC/USDT',
        start_date='2024-01-01',
        end_date='2024-12-31'
    )

    # Prepare for training
    data = loader.prepare_for_training(primary_df, detail_df, normalize=True)

    print("\n[Test] Multi-timeframe data prepared successfully!")
    print(f"Primary shape: {data['primary'].shape}")
    print(f"Detail shape: {data['detail'].shape}")
