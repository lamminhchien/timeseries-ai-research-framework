"""
Quantitative Backtesting and Out-of-Sample Performance Evaluator.
Computes risk-adjusted performance metrics (Sharpe, Sortino, Calmar, Max Drawdown).
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class StrategyEvaluator:
    """
    Rigorously evaluates sequential decision policies under realistic transaction friction.
    """

    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252 * 24):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def compute_equity_curve(
        self,
        returns: np.ndarray,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.0004,
        slippage_bps: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates equity trajectory with transaction fees and slippage deduction.
        """
        friction_deduction = fee_rate + (slippage_bps * 1e-4)
        net_returns = returns - np.where(returns != 0, friction_deduction, 0.0)
        
        equity = [initial_capital]
        for r in net_returns:
            equity.append(equity[-1] * (1.0 + r))
            
        equity_curve = np.array(equity)
        drawdowns = (np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve)
        return equity_curve, drawdowns

    def calculate_performance_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Generates standard statistical and risk-adjusted metrics.
        """
        if len(returns) == 0 or np.all(returns == 0):
            return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}

        mean_ret = np.mean(returns)
        std_ret = np.std(returns) + 1e-8

        # Annualized Sharpe Ratio
        annualized_sharpe = float((mean_ret - self.risk_free_rate) / std_ret * np.sqrt(self.periods_per_year))

        # Downside Deviation & Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) + 1e-8 if len(downside_returns) > 0 else 1e-8
        sortino = float(mean_ret / downside_std * np.sqrt(self.periods_per_year))

        # Equity Curve & Max Drawdown
        _, drawdowns = self.compute_equity_curve(returns)
        max_dd = float(np.max(drawdowns))

        # Win Rate & Profit Factor
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        win_rate = float(len(winning_trades) / max(len(returns[returns != 0]), 1))
        
        gross_profit = float(np.sum(winning_trades)) if len(winning_trades) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losing_trades))) if len(losing_trades) > 0 else 1e-8
        profit_factor = float(gross_profit / gross_loss)

        return {
            "annualized_sharpe": round(annualized_sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "total_samples": len(returns),
        }
