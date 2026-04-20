from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from core.config import ProfessionalConfig


class MonteCarloEngine:
    def __init__(self, config: ProfessionalConfig):
        self.config = config

    def simulate_terminal_values(
        self,
        portfolio_returns: pd.Series,
        initial_capital: float,
        horizon_days: int = 252,
        n_sims: int = 1000,
        random_seed: int = 42,
    ) -> Dict[str, object]:
        rng = np.random.default_rng(random_seed)
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()

        sims = np.zeros((horizon_days, n_sims), dtype=float)
        sims[0, :] = initial_capital

        for j in range(n_sims):
            shocks = rng.normal(mu, sigma, size=horizon_days - 1)
            path = np.empty(horizon_days, dtype=float)
            path[0] = initial_capital
            for t in range(1, horizon_days):
                path[t] = path[t - 1] * (1 + shocks[t - 1])
            sims[:, j] = path

        terminal = sims[-1, :]
        return {
            "paths": sims,
            "terminal_values": terminal,
            "mean_terminal": float(np.mean(terminal)),
            "median_terminal": float(np.median(terminal)),
            "p05_terminal": float(np.quantile(terminal, 0.05)),
            "p95_terminal": float(np.quantile(terminal, 0.95)),
            "prob_loss": float(np.mean(terminal < initial_capital)),
            "prob_gain": float(np.mean(terminal > initial_capital)),
        }

    def simulate_correlated_paths(
        self,
        mean_returns: pd.Series,
        covariance: pd.DataFrame,
        initial_capital: float,
        weights: Dict[str, float],
        horizon_days: int = 252,
        n_sims: int = 1000,
        random_seed: int = 42,
    ) -> Dict[str, object]:
        rng = np.random.default_rng(random_seed)
        assets = list(mean_returns.index)
        w = pd.Series(weights).reindex(assets).fillna(0.0).values
        mu = mean_returns.values / self.config.annual_trading_days
        cov_daily = covariance.values / self.config.annual_trading_days

        sims = np.zeros((horizon_days, n_sims), dtype=float)
        sims[0, :] = initial_capital

        for j in range(n_sims):
            asset_path = np.ones(len(assets), dtype=float) * initial_capital
            series = [initial_capital]
            draws = rng.multivariate_normal(mean=mu, cov=cov_daily, size=horizon_days - 1)
            portfolio_value = initial_capital

            for t in range(horizon_days - 1):
                port_ret = float(np.dot(w, draws[t]))
                portfolio_value *= (1 + port_ret)
                series.append(portfolio_value)

            sims[:, j] = np.array(series)

        terminal = sims[-1, :]
        return {
            "paths": sims,
            "terminal_values": terminal,
            "mean_terminal": float(np.mean(terminal)),
            "median_terminal": float(np.median(terminal)),
            "p05_terminal": float(np.quantile(terminal, 0.05)),
            "p95_terminal": float(np.quantile(terminal, 0.95)),
            "prob_loss": float(np.mean(terminal < initial_capital)),
            "prob_gain": float(np.mean(terminal > initial_capital)),
        }
