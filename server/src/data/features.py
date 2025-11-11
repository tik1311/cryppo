"""
Feature Engineering for 1H Crypto Trading
Calculates technical indicators and features from OHLCV data
"""

import numpy as np
import pandas as pd
from typing import Optional


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


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to dataframe

    Args:
        df: DataFrame with columns [open, high, low, close, volume]

    Returns:
        DataFrame with added technical indicators
    """
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

    return df


def normalize_features(df: pd.DataFrame, feature_cols: list, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize features

    Args:
        df: DataFrame with features
        feature_cols: List of columns to normalize
        method: 'minmax' or 'zscore'

    Returns:
        DataFrame with normalized features
    """
    df = df.copy()

    for col in feature_cols:
        if col not in df.columns:
            continue

        if method == 'minmax':
            # Min-Max scaling to [0, 1]
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val > 0:
                df[col] = (df[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            # Z-score normalization
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[col] = (df[col] - mean_val) / std_val

    return df


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
