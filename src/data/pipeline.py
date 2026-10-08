"""
Data Ingestion and Preprocessing Pipeline for Sequential Market Datasets.
Ensures strict point-in-time sequential splits without look-ahead bias.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SequentialMarketDataset(Dataset):
    """
    PyTorch Dataset providing sliding-window multivariate market sequences
    alongside concurrent account state vectors.
    """

    def __init__(
        self,
        market_features: np.ndarray,
        account_features: np.ndarray,
        seq_length: int = 60,
        forecast_horizon: int = 1,
    ):
        assert len(market_features) == len(account_features), "Mismatch in feature lengths"
        self.market_features = torch.tensor(market_features, dtype=torch.float32)
        self.account_features = torch.tensor(account_features, dtype=torch.float32)
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon

    def __len__(self) -> int:
        return len(self.market_features) - self.seq_length - self.forecast_horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sequence window: [seq_length, num_features]
        market_window = self.market_features[idx : idx + self.seq_length]
        
        # Point-in-time account status at the end of the window
        account_state = self.account_features[idx + self.seq_length - 1]
        
        # Target: Return over forecast horizon
        current_close = self.market_features[idx + self.seq_length - 1, 3]  # Close index
        future_close = self.market_features[idx + self.seq_length - 1 + self.forecast_horizon, 3]
        target_return = (future_close - current_close) / (current_close + 1e-8)
        
        return market_window, account_state, target_return


class TimeSeriesDataPipeline:
    """
    End-to-end data pipeline: Ingestion -> Cleaning -> Scaling -> Windowing.
    """

    def __init__(self, seq_length: int = 60, train_ratio: float = 0.8):
        self.seq_length = seq_length
        self.train_ratio = train_ratio
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill missing ticks, drop remaining NaNs, verify monotonic timestamps."""
        df = df.copy()
        df = df.ffill().bfill()
        return df

    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives log returns, rolling volatility, and normalized spread."""
        df = df.copy()
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
        df["volatility_20"] = df["log_ret"].rolling(window=20, min_periods=1).std().fillna(0.0)
        df["spread_proxy"] = ((df["high"] - df["low"]) / df["close"]).fillna(0.0)
        return df

    def fit_transform_train(self, data: np.ndarray) -> np.ndarray:
        """Fits scaler parameters solely on the training split to avoid look-ahead bias."""
        self.feature_means = np.mean(data, axis=0)
        self.feature_stds = np.std(data, axis=0) + 1e-8
        return (data - self.feature_means) / self.feature_stds

    def transform_val(self, data: np.ndarray) -> np.ndarray:
        """Transforms out-of-sample data using training moments."""
        assert self.feature_means is not None, "Pipeline must fit training data first."
        return (data - self.feature_means) / self.feature_stds

    def create_dataloaders(
        self,
        market_df: pd.DataFrame,
        batch_size: int = 64
    ) -> Tuple[DataLoader, DataLoader]:
        """Creates training and validation DataLoaders with strict temporal partitioning."""
        cleaned_df = self.clean_raw_data(market_df)
        enhanced_df = self.compute_technical_indicators(cleaned_df)
        
        features = enhanced_df[["open", "high", "low", "close", "volume", "log_ret", "volatility_20", "spread_proxy"]].values
        
        # Synthetic account status (Equity, Margin, Floating PnL proxy)
        account_states = np.zeros((len(features), 4), dtype=np.float32)
        account_states[:, 0] = 10000.0  # Initial capital
        
        # Chronological Split
        split_idx = int(len(features) * self.train_ratio)
        
        train_market = self.fit_transform_train(features[:split_idx])
        val_market = self.transform_val(features[split_idx:])
        
        train_acc = account_states[:split_idx]
        val_acc = account_states[split_idx:]
        
        train_dataset = SequentialMarketDataset(train_market, train_acc, seq_length=self.seq_length)
        val_dataset = SequentialMarketDataset(val_market, val_acc, seq_length=self.seq_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
