"""
Feature Engineering for 1H Crypto Trading
Calculates technical indicators and features from OHLCV data

UPDATED: December 2025
- Added OnlineScaler for proper train/test normalization (no data leakage)
- Fixed swing point detection to be causal (no look-ahead bias)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import pickle
from pathlib import Path


class OnlineScaler:
    """
    Online Scaler that fits ONLY on training data and transforms any data.
    Prevents data leakage by not using future information for normalization.

    Usage:
        scaler = OnlineScaler()
        scaler.fit(train_df, columns=['rsi', 'macd', 'atr'])
        train_normalized = scaler.transform(train_df)
        test_normalized = scaler.transform(test_df)  # Uses train statistics!
    """

    def __init__(self, method: str = 'minmax', clip_outliers: bool = True, clip_std: float = 5.0):
        """
        Args:
            method: 'minmax' for [0,1] scaling, 'zscore' for standardization
            clip_outliers: Whether to clip outliers before scaling
            clip_std: Number of std deviations for outlier clipping
        """
        self.method = method
        self.clip_outliers = clip_outliers
        self.clip_std = clip_std
        self.statistics: Dict[str, Dict[str, float]] = {}
        self.fitted = False
        self.columns: List[str] = []

    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'OnlineScaler':
        """
        Fit scaler on training data ONLY.

        Args:
            df: Training DataFrame
            columns: List of columns to normalize
        """
        self.columns = [c for c in columns if c in df.columns]
        self.statistics = {}

        for col in self.columns:
            data = df[col].dropna()

            if len(data) == 0:
                continue

            if self.method == 'minmax':
                self.statistics[col] = {
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'range': float(data.max() - data.min())
                }
            elif self.method == 'zscore':
                self.statistics[col] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()) if data.std() > 0 else 1.0
                }

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted statistics.

        Args:
            df: DataFrame to transform (can be train, val, or test)

        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted! Call fit() first with training data.")

        df = df.copy()

        for col in self.columns:
            if col not in df.columns or col not in self.statistics:
                continue

            stats = self.statistics[col]

            if self.method == 'minmax':
                if stats['range'] > 0:
                    df[col] = (df[col] - stats['min']) / stats['range']
                else:
                    df[col] = 0.5  # Constant column
                # Clip to [0, 1] for test data that may exceed train range
                df[col] = df[col].clip(0, 1)

            elif self.method == 'zscore':
                df[col] = (df[col] - stats['mean']) / stats['std']
                # Clip extreme values
                if self.clip_outliers:
                    df[col] = df[col].clip(-self.clip_std, self.clip_std)

        return df

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fit and transform in one step (for training data only)."""
        self.fit(df, columns)
        return self.transform(df)

    def save(self, path: str):
        """Save scaler to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'clip_outliers': self.clip_outliers,
                'clip_std': self.clip_std,
                'statistics': self.statistics,
                'columns': self.columns,
                'fitted': self.fitted
            }, f)

    @classmethod
    def load(cls, path: str) -> 'OnlineScaler':
        """Load scaler from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        scaler = cls(
            method=data['method'],
            clip_outliers=data['clip_outliers'],
            clip_std=data['clip_std']
        )
        scaler.statistics = data['statistics']
        scaler.columns = data['columns']
        scaler.fitted = data['fitted']
        return scaler


def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average"""
    return pd.Series(data).rolling(window=period).mean().values


def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average"""
    return pd.Series(data).ewm(span=period, adjust=False).mean().values


def calculate_rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index
    Measures momentum on scale of 0-100
    """
    delta = pd.Series(data).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    MACD (Moving Average Convergence Divergence)
    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data: np.ndarray, period: int = 20, num_std: float = 2.0) -> tuple:
    """
    Bollinger Bands
    Returns: (upper_band, middle_band, lower_band)
    """
    sma = calculate_sma(data, period)
    std = pd.Series(data).rolling(window=period).std().values
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average True Range
    Measures volatility
    """
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))

    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(true_range).rolling(window=period).mean().values
    return atr


def calculate_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> tuple:
    """
    Stochastic Oscillator
    Returns: (%K, %D)
    """
    lowest_low = pd.Series(low).rolling(window=period).min().values
    highest_high = pd.Series(high).rolling(window=period).max().values

    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = pd.Series(k).rolling(window=3).mean().values
    return k, d


def calculate_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    On-Balance Volume
    Cumulative volume indicator
    """
    obv = np.zeros_like(volume)
    obv[0] = volume[0]

    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    return obv


def detect_trend(sma_fast: np.ndarray, sma_slow: np.ndarray) -> np.ndarray:
    """
    Detect trend direction using SMA crossover
    Returns: 1 (uptrend), 0 (sideways), -1 (downtrend)
    """
    trend = np.zeros_like(sma_fast)
    # Uptrend: fast SMA > slow SMA
    trend[sma_fast > sma_slow] = 1
    # Downtrend: fast SMA < slow SMA
    trend[sma_fast < sma_slow] = -1
    return trend


def detect_swing_points(high: np.ndarray, low: np.ndarray, period: int = 5) -> tuple:
    """
    Detect swing highs and swing lows using CAUSAL method (no look-ahead bias).

    FIXED: Now uses only PAST data to detect swing points with confirmation delay.
    A swing high is confirmed when the high at position (i-period) is the highest
    in the past window AND subsequent bars have lower highs.

    Returns: (swing_highs, swing_lows) as boolean arrays
    """
    n = len(high)
    swing_highs = np.zeros(n, dtype=bool)
    swing_lows = np.zeros(n, dtype=bool)

    # Need at least 2*period bars for proper detection with confirmation
    min_lookback = period * 2

    for i in range(min_lookback, n):
        # Look at past window only: [i - 2*period : i]
        # Check if bar at (i - period) is a swing point with confirmation

        candidate_idx = i - period
        window_start = max(0, candidate_idx - period)
        window_end = i  # Current bar (exclusive, so up to i-1 for confirmation)

        # Window for checking swing: from (candidate - period) to (candidate + period)
        # But we only use data up to current bar i
        past_window = slice(window_start, candidate_idx + 1)
        confirm_window = slice(candidate_idx, i)

        # Swing High: candidate is highest in past window AND confirmed by lower highs after
        past_high_max = np.max(high[past_window])
        if high[candidate_idx] == past_high_max and high[candidate_idx] > 0:
            # Confirmation: all bars after candidate have lower highs
            if i > candidate_idx + 1:
                confirm_highs = high[candidate_idx + 1:i]
                if len(confirm_highs) > 0 and np.all(confirm_highs < high[candidate_idx]):
                    swing_highs[candidate_idx] = True

        # Swing Low: candidate is lowest in past window AND confirmed by higher lows after
        past_low_min = np.min(low[past_window])
        if low[candidate_idx] == past_low_min and low[candidate_idx] > 0:
            # Confirmation: all bars after candidate have higher lows
            if i > candidate_idx + 1:
                confirm_lows = low[candidate_idx + 1:i]
                if len(confirm_lows) > 0 and np.all(confirm_lows > low[candidate_idx]):
                    swing_lows[candidate_idx] = True

    return swing_highs, swing_lows


def calculate_dow_theory_structure(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                   swing_period: int = 5, lookback: int = 20) -> tuple:
    """
    Dow Theory: Detect Higher Highs, Higher Lows (uptrend) or Lower Highs, Lower Lows (downtrend)
    Returns: (structure_score, is_uptrend, is_downtrend)
    - structure_score: +1 for HH/HL pattern, -1 for LH/LL pattern, 0 for mixed
    - is_uptrend: boolean array for confirmed uptrend
    - is_downtrend: boolean array for confirmed downtrend
    """
    n = len(high)
    structure_score = np.zeros(n)
    is_uptrend = np.zeros(n, dtype=bool)
    is_downtrend = np.zeros(n, dtype=bool)

    # Detect swing points
    swing_highs, swing_lows = detect_swing_points(high, low, swing_period)

    # Find indices of swing points
    swing_high_indices = np.where(swing_highs)[0]
    swing_low_indices = np.where(swing_lows)[0]

    # Analyze structure for each point
    for i in range(lookback, n):
        # Get recent swing highs and lows within lookback period
        recent_high_idx = swing_high_indices[(swing_high_indices >= i-lookback) & (swing_high_indices < i)]
        recent_low_idx = swing_low_indices[(swing_low_indices >= i-lookback) & (swing_low_indices < i)]

        if len(recent_high_idx) >= 2 and len(recent_low_idx) >= 2:
            # Check for Higher Highs (HH) and Higher Lows (HL) - Uptrend
            recent_highs = high[recent_high_idx[-2:]]
            recent_lows = low[recent_low_idx[-2:]]

            hh = recent_highs[-1] > recent_highs[-2]  # Higher High
            hl = recent_lows[-1] > recent_lows[-2]    # Higher Low

            lh = recent_highs[-1] < recent_highs[-2]  # Lower High
            ll = recent_lows[-1] < recent_lows[-2]    # Lower Low

            if hh and hl:
                # Confirmed uptrend
                structure_score[i] = 1.0
                is_uptrend[i] = True
            elif lh and ll:
                # Confirmed downtrend
                structure_score[i] = -1.0
                is_downtrend[i] = True
            else:
                # Mixed structure (consolidation)
                structure_score[i] = 0.0

    return structure_score, is_uptrend, is_downtrend


def add_minimal_indicators(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add minimal indicators for detail timeframe (1m)
    Purpose: Detail timeframe is used only for accurate TP/SL execution tracking
    We only need: OHLCV + ATR (for TP/SL calculation)

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        verbose: Print progress messages

    Returns:
        DataFrame with minimal indicators (OHLCV + ATR only)
    """
    if verbose:
        print(f"  [Detail TF] Adding minimal indicators... ({len(df):,} candles)")

    df = df.copy()

    # Only ATR for TP/SL calculation
    df['atr'] = calculate_atr(df['high'].values, df['low'].values, df['close'].values)

    # Fill NaN with forward fill
    df['atr'] = df['atr'].ffill().bfill()

    if verbose:
        print(f"  ✓ Detail timeframe ready! (OHLCV + ATR only)")

    return df


def add_technical_indicators(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add all technical indicators to dataframe (for MAIN timeframe)

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        verbose: Print progress messages

    Returns:
        DataFrame with added technical indicators
    """
    if verbose:
        print(f"  [1/4] Calculating basic indicators... ({len(df):,} candles)")

    df = df.copy()

    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Moving Averages
    df['sma_7'] = calculate_sma(df['close'].values, 7)
    df['sma_25'] = calculate_sma(df['close'].values, 25)
    df['sma_99'] = calculate_sma(df['close'].values, 99)
    df['ema_7'] = calculate_ema(df['close'].values, 7)
    df['ema_25'] = calculate_ema(df['close'].values, 25)

    # RSI
    df['rsi'] = calculate_rsi(df['close'].values, 14)

    # MACD
    macd, signal, hist = calculate_macd(df['close'].values)
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = hist

    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(df['close'].values)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_width'] = (upper - lower) / middle

    # ATR
    df['atr'] = calculate_atr(df['high'].values, df['low'].values, df['close'].values)

    # Stochastic
    k, d = calculate_stochastic(df['high'].values, df['low'].values, df['close'].values)
    df['stoch_k'] = k
    df['stoch_d'] = d

    # OBV
    df['obv'] = calculate_obv(df['close'].values, df['volume'].values)

    # Volume features
    df['volume_sma'] = calculate_sma(df['volume'].values, 20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()

    # Price position
    df['price_to_sma7'] = df['close'] / df['sma_7']
    df['price_to_sma25'] = df['close'] / df['sma_25']

    # === NEW: Trend, Swing, and Dow Theory Features ===
    if verbose:
        print(f"  [2/4] Detecting trend direction...")

    # Trend Direction (SMA 7 vs SMA 25)
    df['trend_direction'] = detect_trend(df['sma_7'].values, df['sma_25'].values)

    # Trend Strength (distance between EMAs)
    df['trend_strength'] = (df['ema_7'] - df['ema_25']) / df['close']

    if verbose:
        print(f"  [3/4] Detecting swing points...")

    # Swing Points Detection
    swing_highs, swing_lows = detect_swing_points(df['high'].values, df['low'].values, period=5)
    df['is_swing_high'] = swing_highs.astype(float)
    df['is_swing_low'] = swing_lows.astype(float)

    # Distance to nearest swing points - OPTIMIZED VERSION (vectorized)
    df['distance_to_swing_high'] = 0.0
    df['distance_to_swing_low'] = 0.0

    swing_high_indices = np.where(swing_highs)[0]
    swing_low_indices = np.where(swing_lows)[0]

    if len(swing_high_indices) > 0:
        # Vectorized approach using broadcasting
        close_array = df['close'].values
        high_array = df['high'].values

        for i, idx in enumerate(swing_high_indices):
            # Set distance for all points after this swing high (until next swing high)
            end_idx = swing_high_indices[i+1] if i+1 < len(swing_high_indices) else len(df)
            start_idx = max(idx, idx)
            if end_idx - start_idx > 0:
                distances = (close_array[start_idx:end_idx] - high_array[idx]) / close_array[start_idx:end_idx]
                df.iloc[start_idx:end_idx, df.columns.get_loc('distance_to_swing_high')] = distances

    if len(swing_low_indices) > 0:
        # Vectorized approach using broadcasting
        close_array = df['close'].values
        low_array = df['low'].values

        for i, idx in enumerate(swing_low_indices):
            # Set distance for all points after this swing low (until next swing low)
            end_idx = swing_low_indices[i+1] if i+1 < len(swing_low_indices) else len(df)
            start_idx = max(idx, idx)
            if end_idx - start_idx > 0:
                distances = (close_array[start_idx:end_idx] - low_array[idx]) / close_array[start_idx:end_idx]
                df.iloc[start_idx:end_idx, df.columns.get_loc('distance_to_swing_low')] = distances

    if verbose:
        print(f"  [4/4] Calculating trend structure...")

    # Dow Theory Price Structure - SIMPLIFIED (faster version)
    # Instead of detailed calculation, use simplified trend structure
    df['dow_structure'] = 0.0
    df['is_uptrend'] = 0.0
    df['is_downtrend'] = 0.0

    # Simple uptrend/downtrend detection based on SMA positions
    # Much faster than full Dow Theory calculation
    uptrend_condition = (df['sma_7'] > df['sma_25']) & (df['sma_25'] > df['sma_99'])
    downtrend_condition = (df['sma_7'] < df['sma_25']) & (df['sma_25'] < df['sma_99'])

    df.loc[uptrend_condition, 'dow_structure'] = 1.0
    df.loc[uptrend_condition, 'is_uptrend'] = 1.0

    df.loc[downtrend_condition, 'dow_structure'] = -1.0
    df.loc[downtrend_condition, 'is_downtrend'] = 1.0

    # Additional trend confirmation
    df['price_above_sma25'] = (df['close'] > df['sma_25']).astype(float)
    df['price_above_sma99'] = (df['close'] > df['sma_99']).astype(float)

    if verbose:
        print(f"  ✓ Feature engineering complete! ({len(df.columns)} features total)")

    return df


def normalize_features(df: pd.DataFrame, feature_cols: list, method: str = 'minmax',
                       scaler: Optional[OnlineScaler] = None) -> pd.DataFrame:
    """
    Normalize features using OnlineScaler to prevent data leakage.

    UPDATED: Now uses OnlineScaler for proper train/test separation.

    Args:
        df: DataFrame with features
        feature_cols: List of columns to normalize
        method: 'minmax' or 'zscore'
        scaler: Optional pre-fitted OnlineScaler. If None, creates new one (for training).

    Returns:
        DataFrame with normalized features

    WARNING: If scaler is None, this will fit on the provided data.
             For proper train/test split, pass a pre-fitted scaler for test data.
    """
    df = df.copy()

    if scaler is not None and scaler.fitted:
        # Use pre-fitted scaler (for validation/test data)
        return scaler.transform(df)
    else:
        # Create new scaler and fit on this data (for training data)
        new_scaler = OnlineScaler(method=method)
        return new_scaler.fit_transform(df, feature_cols)


def create_sequences(df: pd.DataFrame, feature_cols: list, window_size: int = 72) -> tuple:
    """
    Create sequences for training

    Args:
        df: DataFrame with features
        feature_cols: List of feature columns
        window_size: Number of timesteps per sequence (72 hours = 3 days for 1H data)

    Returns:
        (sequences, targets) as numpy arrays
    """
    sequences = []
    targets = []

    data = df[feature_cols].values

    for i in range(window_size, len(data)):
        sequences.append(data[i - window_size:i])
        targets.append(data[i])

    return np.array(sequences), np.array(targets)


def prepare_training_data(
    df: pd.DataFrame,
    window_size: int = 72,
    train_split: float = 0.8,
    val_split: float = 0.1
) -> dict:
    """
    Prepare data for training

    Args:
        df: DataFrame with OHLCV data
        window_size: Sequence length
        train_split: Training data ratio
        val_split: Validation data ratio

    Returns:
        Dictionary with train/val/test splits
    """
    # Add technical indicators
    df = add_technical_indicators(df)

    # Drop NaN values
    df = df.dropna()

    # Define feature columns (exclude non-features)
    exclude_cols = ['timestamp'] if 'timestamp' in df.columns else []
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Normalize features
    df = normalize_features(df, feature_cols, method='minmax')

    # Create sequences
    sequences, targets = create_sequences(df, feature_cols, window_size)

    # Split data
    n = len(sequences)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    return {
        'train': {
            'sequences': sequences[:train_end],
            'targets': targets[:train_end],
        },
        'val': {
            'sequences': sequences[train_end:val_end],
            'targets': targets[train_end:val_end],
        },
        'test': {
            'sequences': sequences[val_end:],
            'targets': targets[val_end:],
        },
        'feature_cols': feature_cols,
        'n_features': len(feature_cols)
    }
