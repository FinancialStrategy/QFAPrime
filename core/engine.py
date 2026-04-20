from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.analytics import AnalyticsEngine
from core.config import ProfessionalConfig, RunDiagnostics
from core.data_loader import ProfessionalDataManager
from core.expected_returns import ExpectedReturnBuilder
from core.monte_carlo import MonteCarloEngine
from core.optimizers import ProfessionalOptimizer, StrategyResult
from core.risk_models import RiskModelBuilder
from core.scenarios import run_historical_stress_tests, run_hypothetical_shocks
from core.universes import flatten_universe_dict, get_universe_definition


class ProfessionalPortfolioEngine:
    def __init__(self, config: Optional[ProfessionalConfig] = None):
        self.config = config or ProfessionalConfig()
        self.diagnostics = RunDiagnostics()

        self.data = ProfessionalDataManager(self.config, self.diagnostics)
        self.expected_return_builder = ExpectedReturnBuilder(self.config, self.diagnostics)
        self.risk_builder = RiskModelBuilder(self.config, self.diagnostics)
        self.analytics = AnalyticsEngine(self.config)
        self.monte_carlo = MonteCarloEngine(self.config)

        self.mu: pd.Series | None = None
        self.cov: pd.DataFrame | None = None
        self.corr: pd.DataFrame | None = None

        self.strategies: Dict[str, StrategyResult] = {}
        self.metrics: Dict[str, Dict] = {}
        self.metrics_df = pd.DataFrame()
        self.strategy_df = pd.DataFrame()
        self.risk_contributions: Dict[str, pd.DataFrame] = {}
        self.historical_stress: Dict[str, pd.DataFrame] = {}
        self.hypothetical_shocks: Dict[str, pd.DataFrame] = {}
        self.monte_carlo_results: Dict[str, Dict] = {}

    def run(self, current_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        self.data.load()

        self.mu = self.expected_return_builder.build(
            self.data.asset_returns,
            self.data.benchmark_returns,
        )
        self.cov = self.risk_builder.build_covariance(self.data.asset_prices)
        self.corr = self.risk_builder.correlation_matrix(self.data.asset_returns)

        asset_categories = {
            meta["ticker"]: meta["bucket"]
            for meta in flatten_universe_dict(
                get_universe_definition(self.config.selected_universe)
            ).values()
        }

        optimizer = ProfessionalOptimizer(
            mu=self.mu,
            cov=self.cov,
            returns=self.data.asset_returns,
            benchmark_returns=self.data.benchmark_returns,
            config=self.config,
            diagnostics=self.diagnostics,
            asset_categories=asset_categories,
            current_weights=current_weights,
        )
        self.strategies = optimizer.run_all()

        for name, result in self.strategies.items():
            pr = self.analytics.portfolio_returns(self.data.asset_returns, result.weights)
            metrics = self.analytics.calculate_all_metrics(
                pr,
                self.data.benchmark_returns,
                self.config.initial_capital,
            )
            self.metrics[name] = metrics

            self.risk_contributions[name] = self.analytics.risk_contribution_table(
                result.weights,
                self.cov,
            )

            self.historical_stress[name] = run_historical_stress_tests(
                metrics["portfolio_returns"],
                metrics["benchmark_returns"],
            )

            self.hypothetical_shocks[name] = run_hypothetical_shocks(result.weights)

            self.monte_carlo_results[name] = self.monte_carlo.simulate_terminal_values(
                metrics["portfolio_returns"],
                initial_capital=self.config.initial_capital,
                horizon_days=self.config.annual_trading_days,
                n_sims=1000,
                random_seed=42,
            )

        self.metrics_df = pd.DataFrame(self.metrics).T.sort_values("sharpe_ratio", ascending=False)
        self.strategy_df = pd.DataFrame([
            {
                "strategy": name,
                "method": result.method,
                "num_assets": len([w for w in result.weights.values() if w > 0.001]),
                "max_weight": max(result.weights.values()),
                "turnover": result.diagnostics.get("turnover", 0.0),
                "transaction_cost": result.diagnostics.get("estimated_transaction_cost_usd", 0.0),
                "tracking_error_target": result.diagnostics.get("tracking_error_target", np.nan),
                "ex_ante_tracking_error": result.diagnostics.get("ex_ante_tracking_error", np.nan),
            }
            for name, result in self.strategies.items()
        ])

        return self.metrics_df

    def best_strategy_name(self) -> str:
        if self.metrics_df.empty:
            raise ValueError("Engine has not been run yet.")
        return self.metrics_df.index[0]

    def tracking_error_strategy_name(self) -> str:
        for strategy_name, strategy_obj in self.strategies.items():
            if strategy_obj.method == "tracking_error_optimal":
                return strategy_name
        return self.best_strategy_name()
