# AI-Powered Time-Series & Sequential Decision Framework

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Distributed-Ray](https://img.shields.io/badge/Distributed-Ray%20Core-028CF0.svg?style=flat&logo=ray)](https://www.ray.io/)
[![Optuna-HPO](https://img.shields.io/badge/HPO-Optuna-3B5998.svg?style=flat)](https://optuna.org/)
[![Gymnasium](https://img.shields.io/badge/RL-Gymnasium-008080.svg?style=flat)](https://gymnasium.farama.org/)
[![License-MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

An end-to-end deep learning and reinforcement learning research framework dedicated to **multivariate time-series forecasting**, **distributed hyperparameter optimization (HPO)**, and **sequential quantitative decision-making under uncertainty**.

---

## 📌 Executive Summary

Predicting sequential market patterns is one of the most challenging domains in applied Machine Learning due to non-stationarity, low signal-to-noise ratio, and complex temporal dependencies. 

This repository documents the evolution of a rigorous research framework:
1. **Phase 1: Project Tom (Baseline Foundations & Distributed HPO)** — Hybrid TCN-Transformer feature extractors, curriculum-driven reinforcement learning, and large-scale parallel hyperparameter searches using Ray and Optuna.
2. **Phase 2: Project G3 (Advanced High-Efficiency Architecture)** — Extension toward ultra-low latency inference, memory-mapped (`mmap`) dataset streaming, strict prevention of look-ahead bias, and sub-18MB VRAM footprint on edge/consumer hardware.

---

## 🏛️ System Architecture

```text
                           [ Multi-Timeframe Raw Sequential Feeds ]
                                              │
                                 (ETL & Feature Pipeline)
                                 - Rolling Window Normalization
                                 - Strict Point-in-Time Splits
                                              │
                 ┌────────────────────────────┴───────────────────────────┐
                 ▼                                                        ▼
      [ Market History Features ]                              [ Real-Time State Vector ]
                 │                                                        │
                 ▼                                                        ▼
    ┌─────────────────────────┐                              ┌─────────────────────────┐
    │ Temporal ConvNet (TCN)  │                              │ Multi-Layer Perceptron  │
    │ Dilated Causal Convs    │                              │ State Embedding Network │
    └────────────┬────────────┘                              └────────────┬────────────┘
                 ▼                                                        │
    ┌─────────────────────────┐                                           │
    │ Transformer Encoder     │                                           │
    │ Multi-Head Self-Attn    │                                           │
    └────────────┬────────────┘                                           │
                 │                                                        │
                 └────────────────────────────┬───────────────────────────┘
                                              ▼
                               [ Fused Latent Representation ]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
           [ Actor Network (Policy) ]                    [ Critic Network (Value) ]
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                                 [ Gymnasium Simulated Arena ]
                                 - Realistic Slippage & Spreads
                                 - Zero-Sum Evaluation Metrics
```

---

## 🔬 Research Methodology & Key Components

### 1. Robust Data Engineering & Point-in-Time Pipeline
* **Mitigating Look-Ahead Leakage:** Strict enforcement of point-in-time sequential splits and rolling window normalizations (Welford's algorithm) to prevent future statistical moments from leaking into historical feature representations.
* **Feature Curation:** Integrated multi-timeframe OHLCV tensors, moving volatility estimators, and order-flow proxies across high-frequency datasets.

### 2. ExtractorTransformer: Hybrid Temporal Representation
* Combines **Dilated Temporal Convolutional Networks (TCN)** for local causal pattern detection with a **Multi-Head Transformer Encoder** for global sequence context.
* Dynamic state fusion: Integrates sequence market features with continuous internal portfolio state vectors into a unified latent space.

### 3. Distributed Hyperparameter Search (Ray + Optuna)
* Built a multi-worker HPO engine capable of distributing trials across available GPUs using asynchronous task queues.
* Implemented automatic trial pruning (`optuna.exceptions.TrialPruned`) based on intermediate validation loss and Sharpe Ratio thresholds to maximize compute efficiency.

### 4. Curriculum Learning in Simulated Arena
* Structured multi-tiered difficulty levels ("Curriculum Training") where agents face progressively wider spreads, higher slippage, and adverse volatility regimes.
* Zero-sum reward formulations to stabilize policy gradient updates in continuous action spaces.

### 5. Architectural Scaling: From Project Tom to Project G3
* **Memory Optimization:** Leveraged `numpy.memmap` arrays to stream gigabyte-scale datasets directly to PyTorch without RAM memory saturation.
* **Hardware Efficiency:** Streamlined network tensor computations to sustain continuous inference with peak VRAM capped at **< 18 MB**, eliminating memory leaks.
* **Walk-Forward Validation:** Evaluated strategy degradation using walk-forward out-of-sample (OOS) testing windows.

---

## 💻 Technical Showcases

### Showcase 1: Hybrid TCN-Transformer Feature Extractor

Modular PyTorch implementation featuring dilated causal convolutions for local dependencies and self-attention for cross-temporal representations.

```python
import torch
import torch.nn as nn
from gymnasium import spaces

class PositionalEncoding(nn.Module):
    """Injects sequence position information into feature representations."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CausalConv1dBlock(nn.Module):
    """
    Dilated Causal 1D Convolution with strict past-only horizon.
    Pads (kernel_size - 1) * dilation on both sides, then trims future elements.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=self.pad)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]  # Strict causal past-only trim
        return self.drop(self.act(out))


class ExtractorTransformer(nn.Module):
    """
    Hybrid Temporal Feature Extractor:
    1. Causal Conv1d blocks with exponential dilation (2^i) for multi-scale pattern extraction.
    2. Positional Encoding + Transformer Encoder captures long-horizon context.
    3. State projection MLP processes instantaneous internal state vectors.
    """
    def __init__(
        self,
        market_feature_dim: int,
        seq_length: int,
        account_dim: int,
        tcn_channels: list[int],
        tcn_kernel_size: int = 3,
        tfm_nhead: int = 4,
        tfm_layers: int = 2,
        tfm_dim_ff: int = 128,
        account_embed_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        tcn_out_dim = tcn_channels[-1]
        
        # 1. Temporal Convolutional Backbone (Exponential Dilation)
        layers = []
        in_c = market_feature_dim
        for i, out_c in enumerate(tcn_channels):
            dilation = 2 ** i
            layers.append(CausalConv1dBlock(in_c, out_c, kernel_size=tcn_kernel_size, dilation=dilation, dropout=dropout))
            in_c = out_c
        self.tcn = nn.Sequential(*layers)

        # 2. Transformer Contextual Encoder
        self.pos_encoder = PositionalEncoding(tcn_out_dim, dropout=dropout, max_len=seq_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tcn_out_dim,
            nhead=tfm_nhead,
            dim_feedforward=tfm_dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=tfm_layers)

        # 3. State Vector Processor
        self.account_mlp = nn.Sequential(
            nn.Linear(account_dim, 64),
            nn.ReLU(),
            nn.Linear(64, account_embed_dim),
            nn.Tanh()
        )
        self.total_output_dim = tcn_out_dim + account_embed_dim

    def forward(self, market_data: torch.Tensor, account_data: torch.Tensor) -> torch.Tensor:
        # market_data: [Batch, Seq_Len, Features] -> Transpose for Conv1d: [Batch, Features, Seq_Len]
        x = market_data.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)  # [Batch, Seq_Len, Channels] - Exact length preserved!

        # Positional Encoding + Attention
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        temporal_summary = x.mean(dim=1)  # Sequence Pooling

        # Latent Fusion
        account_summary = self.account_mlp(account_data)
        return torch.cat([temporal_summary, account_summary], dim=-1)
```

---

### Showcase 2: Distributed Hyperparameter Worker (Ray + Optuna)

Asynchronous trial executor ensuring zero GPU idle time, dynamic exception tracking, and deterministic random state management.

```python
import traceback
import optuna

def objective_worker(
    trial: optuna.Trial,
    gpu_queue,
    training_params: dict,
    train_data_arrays: dict,
    val_data_arrays: dict,
    hparam_definer_func=None,
    model_compiler_func=None,
    objective_metric: str = "sharpe",
    seed_override: int = 42,
    eval_episodes: int = 5
) -> float:
    """
    Thread-safe worker executing an isolated trial across allocated GPU devices.
    Guarantees resource release back to gpu_queue upon trial completion or exception.
    """
    gpu_id = -1
    try:
        assert gpu_queue is not None, "Error: gpu_queue must be provided for distributed execution."
        gpu_id = gpu_queue.get(timeout=120)

        # Execute training & validation trial on target device
        score = execute_training_pipeline(
            trial=trial,
            device_id=gpu_id,
            training_params=training_params,
            train_data=train_data_arrays,
            val_data=val_data_arrays,
            hparam_fn=hparam_definer_func,
            model_fn=model_compiler_func,
            metric=objective_metric,
            seed=seed_override,
            eval_episodes=eval_episodes
        )
        return score

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as exc:
        traceback.print_exc()
        trial.set_user_attr("worker_error", traceback.format_exc())
        raise exc
    finally:
        # Guarantee device index is restored to worker pool
        if gpu_id != -1 and gpu_queue is not None:
            gpu_queue.put(gpu_id)
```

---

## 📊 Experimental Results & Empirical Validation

Below are quantitative results obtained across multiple optimization and evaluation checkpoints during research runs on **Project Tom**:

### 1. Cumulative Return Trajectory & Policy Convergence
The learned continuous policy demonstrates consistent risk-adjusted returns during out-of-sample backtesting under friction conditions.

<p align="center">
  <img width="880" alt="Cumulative Returns" src="https://github.com/user-attachments/assets/078927f8-cfb3-4474-b07d-8b5424f4ab53" />
</p>

<p align="center">
  <img width="880" alt="Evaluation Performance" src="https://github.com/user-attachments/assets/9853dc6f-3638-43f8-bbd4-de9e52e498a6" />
</p>

---

### 2. Hyperparameter Exploration Landscape (Optuna Studies)
Parallel coordinate plots and hyperparameter importance graphs demonstrating convergence toward optimal learning rates, entropy coefficients, and network depths.

<p align="center">
  <img width="420" alt="Optimization Frontier" src="https://github.com/user-attachments/assets/50d6818d-c166-43ad-b69b-7fdc6bf9139c" />
  <img width="480" alt="Hyperparameter Correlations" src="https://github.com/user-attachments/assets/c3360a21-0395-45a4-b6ae-9cc30e8397b0" />
</p>

<p align="center">
  <img width="880" alt="Slice Plot HPO" src="https://github.com/user-attachments/assets/5391838c-d1b5-4c76-901d-6064f588feed" />
</p>

---

### 3. Out-of-Sample Stability & Distribution Analysis
Evaluation across multi-episode rollouts verifying drawdown containment and win-rate stability.

<p align="center">
  <img width="480" alt="Drawdown Analysis" src="https://github.com/user-attachments/assets/3ee94910-6f78-4e7e-a166-9a98b048192e" />
  <img width="880" alt="Step Performance Profile" src="https://github.com/user-attachments/assets/64b335f4-82e2-4a94-9af8-e790087ca9d6" />
</p>

<p align="center">
  <img width="880" alt="Full Backtest Evaluation" src="https://github.com/user-attachments/assets/127e9f51-5357-48a5-b074-57e5de1643f4" />
</p>

<p align="center">
  <img width="880" alt="Validation Horizon Comparison" src="https://github.com/user-attachments/assets/f8469ac3-76f8-4245-b038-837208b35ad6" />
</p>

---

## 🛠️ Tech Stack & Requirements

* **Core Language:** Python 3.10+
* **Deep Learning:** PyTorch 2.x, TorchVision
* **Distributed Computing:** Ray Core, Ray Tune
* **Hyperparameter Search:** Optuna
* **Environment:** Gymnasium, NumPy, Pandas, Scikit-learn
* **Hardware Profiles Tested:** NVIDIA Tesla T4 / P100 (Kaggle Cloud), NVIDIA GTX 1660 Ti (Local Edge Inference)

---

## ⚖️ License & Disclaimer

This project is licensed under the MIT License. The code and models presented here are for **academic and quantitative research purposes only**. They do not constitute financial advice or real-money trading endorsements.
