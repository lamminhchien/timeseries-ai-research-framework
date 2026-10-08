"""
Core Model Training Engine.
Features learning rate scheduling, gradient clipping, checkpointing, and evaluation tracking.
"""

from typing import Dict, Optional
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


class TimeSeriesTrainer:
    """
    Supervised/Self-Supervised sequential model trainer with strict numerical safeguards.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        max_grad_norm: float = 1.0,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Head for return prediction proxy
        self.regressor = nn.Linear(model.features_dim, 1).to(self.device)
        
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.regressor.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.SmoothL1Loss()  # Robust to market outliers (Huber)
        self.max_grad_norm = max_grad_norm

    def train_epoch(self) -> float:
        """Trains for one full epoch over sequential batches."""
        self.model.train()
        self.regressor.train()
        total_loss = 0.0
        
        for mkt_batch, acc_batch, target_batch in self.train_loader:
            mkt_batch = mkt_batch.to(self.device)
            acc_batch = acc_batch.to(self.device)
            target_batch = target_batch.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            features = self.model(mkt_batch, acc_batch)
            predictions = self.regressor(features)
            loss = self.criterion(predictions, target_batch)

            # Check for numerical instability (Silent Failure Hunter)
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: Detected NaN/Inf loss. Skipping step.")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.regressor.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Runs out-of-sample validation evaluation."""
        self.model.eval()
        self.regressor.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for mkt_batch, acc_batch, target_batch in self.val_loader:
            mkt_batch = mkt_batch.to(self.device)
            acc_batch = acc_batch.to(self.device)
            target_batch = target_batch.to(self.device).unsqueeze(1)

            features = self.model(mkt_batch, acc_batch)
            predictions = self.regressor(features)
            loss = self.criterion(predictions, target_batch)

            total_loss += loss.item()
            all_preds.append(predictions.cpu())
            all_targets.append(target_batch.cpu())

        val_loss = total_loss / max(len(self.val_loader), 1)
        
        preds_tensor = torch.cat(all_preds, dim=0).numpy().squeeze()
        targets_tensor = torch.cat(all_targets, dim=0).numpy().squeeze()
        
        # Directional Accuracy (Hit Rate)
        hit_rate = float(np.mean((preds_tensor > 0) == (targets_tensor > 0)))

        return {
            "val_loss": val_loss,
            "hit_rate": hit_rate
        }

    def save_checkpoint(self, filepath: str) -> None:
        """Saves weights and optimizer state to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "regressor_state_dict": self.regressor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, filepath)
        print(f"Saved checkpoint to: {filepath}")
