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
from core.scenarios import (
    detect_sharp_fluctuation_windows,
    extract_scenario_path,
    run_historical_stress_tests,
    run_hypothetical_shocks,
)
from core.universes import flatten_universe_dict, get_universe_definition


class ProfessionalPortfolioEngine:
    def __init__(
        self,
        config: Optional[ProfessionalConfig] = None,
        bl_controls: Optional[Dict] = None,
        scenario_controls: Optional[Dict] = None,
    ):
        self.config = config or ProfessionalConfig()
        self.bl_controls = bl_controls or {
            "enabled": False,
            "view_mode": "ticker",
            "views_payload": [],
        }
        self.scenario_controls = scenario_controls or {
            "selected_family": "All",
            "minimum_severity_threshold": 0.0,
            "quick_view": "All",
        }
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
        self.historical_stress_paths: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.hypothetical_shocks: Dict[str, pd.DataFrame] = {}
        self.sharp_fluctuation_windows: Dict[str, pd.DataFrame] = {}
        self.monte_carlo_results: Dict[str, Dict] = {}

        self.bl_prior_returns: pd.Series | None = None
        self.bl_posterior_returns: pd.Series | None = None
        self.bl_posterior_cov: pd.DataFrame | None = None
        self.bl_weights: Dict[str, float] | None = None

        self.pca_results: Dict = {}

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
            asset_metadata=self.data.asset_metadata,
            current_weights=current_weights,
            bl_controls=self.bl_controls,
        )
        self.strategies = optimizer.run_all()

        for name, result in self.strategies.items():
            pr = self.analytics.portfolio_returns(self.data.asset_returns, result.weights)

            metrics = self.analytics.calculate_all_metrics(
                pr,
                self.data.benchmark_returns,
                self.config.initial_capital,
            )
            if not metrics:
                continue

            self.metrics[name] = metrics

            self.risk_contributions[name] = self.analytics.risk_contribution_table(
                result.weights,
                self.cov,
            )

            stress_df = run_historical_stress_tests(
                metrics["portfolio_returns"],
                metrics["benchmark_returns"],
            )
            self.historical_stress[name] = stress_df

            path_map: Dict[str, pd.DataFrame] = {}
            if stress_df is not None and not stress_df.empty:
                for _, row in stress_df.iterrows():
                    path_map[row["scenario"]] = extract_scenario_path(
                        metrics["portfolio_returns"],
                        metrics["benchmark_returns"],
                        row["start_date"],
                        row["end_date"],
                    )
            self.historical_stress_paths[name] = path_map

            self.hypothetical_shocks[name] = run_hypothetical_shocks(result.weights)

            self.sharp_fluctuation_windows[name] = detect_sharp_fluctuation_windows(
                metrics["portfolio_returns"],
                window=21,
                top_n=5,
            )

            self.monte_carlo_results[name] = self.monte_carlo.simulate_terminal_values(
                metrics["portfolio_returns"],
                initial_capital=self.config.initial_capital,
                horizon_days=self.config.annual_trading_days,
                n_sims=1000,
                random_seed=42,
            )

            if result.method == "black_litterman":
                prior = result.diagnostics.get("prior_returns")
                posterior = result.diagnostics.get("posterior_returns")
                bl_w = result.diagnostics.get("bl_weight_output")

                if prior is not None:
                    self.bl_prior_returns = pd.Series(prior)

                if posterior is not None:
                    self.bl_posterior_returns = pd.Series(posterior)

                if bl_w is not None:
                    self.bl_weights = bl_w

        if self.bl_posterior_returns is not None and self.cov is not None:
            self.bl_posterior_cov = self.cov.copy()

        self.pca_results = self.analytics.pca_factor_analysis(
            self.data.asset_returns,
            n_components=5,
        )

        if self.metrics:
            self.metrics_df = pd.DataFrame(self.metrics).T.sort_values(
                "sharpe_ratio",
                ascending=False,
            )
        else:
            self.metrics_df = pd.DataFrame()

        strategy_rows = []
        for name, result in self.strategies.items():
            weights_values = list(result.weights.values()) if result.weights else []
            strategy_rows.append({
                "strategy": name,
                "method": result.method,
                "num_assets": len([w for w in weights_values if w > 0.001]),
                "max_weight": max(weights_values) if weights_values else 0.0,
                "turnover": result.diagnostics.get("turnover", 0.0),
                "transaction_cost": result.diagnostics.get("estimated_transaction_cost_usd", 0.0),
                "tracking_error_target": result.diagnostics.get("tracking_error_target", np.nan),
                "ex_ante_tracking_error": result.diagnostics.get("ex_ante_tracking_error", np.nan),
            })

        self.strategy_df = pd.DataFrame(strategy_rows)
        return self.metrics_df

    def best_strategy_name(self) -> str:
        if self.metrics_df.empty:
            raise ValueError("Engine has not been run yet.")
        return str(self.metrics_df.index[0])

    def tracking_error_strategy_name(self) -> str:
        for strategy_name, strategy_obj in self.strategies.items():
            if strategy_obj.method == "tracking_error_optimal":
                return strategy_name
        return self.best_strategy_name()

    def filter_stress_dataframe(self, stress_df: pd.DataFrame) -> pd.DataFrame:
        if stress_df is None or stress_df.empty:
            return pd.DataFrame()

        out = stress_df.copy()

        selected_family = self.scenario_controls.get("selected_family", "All")
        min_severity = float(self.scenario_controls.get("minimum_severity_threshold", 0.0))
        quick_view = self.scenario_controls.get("quick_view", "All")

        quick_view_map = {
            "Crisis Only": "Crisis",
            "Inflation Only": "Inflation",
            "Banking Stress Only": "Banking_Stress",
            "Sharp Rally Only": "Sharp_Rally",
            "Sharp Selloff Only": "Sharp_Selloff",
        }

        if quick_view in quick_view_map and "family" in out.columns:
            out = out[out["family"] == quick_view_map[quick_view]]

        if selected_family != "All" and "family" in out.columns:
            out = out[out["family"] == selected_family]

        if "severity_score" in out.columns:
            out = out[out["severity_score"] >= min_severity]

        return out.reset_index(drop=True)
