from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

try:
    from pypfopt import EfficientFrontier, expected_returns, objective_functions
    from pypfopt import risk_models as pypfopt_risk_models
    from pypfopt.black_litterman import BlackLittermanModel, market_implied_risk_aversion
    HAS_PYPORTFOLIOOPT = True
except Exception:
    HAS_PYPORTFOLIOOPT = False


logger = logging.getLogger("QFA_QUANT_PLATFORM")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# =========================================================
# Diagnostics / helpers
# =========================================================
@dataclass
class RunDiagnostics:
    dropped_assets: Dict[str, str] = field(default_factory=dict)
    info: List[str] = field(default_factory=list)
    warnings_list: List[str] = field(default_factory=list)
    errors_list: List[str] = field(default_factory=list)

    benchmark_used: Optional[str] = None
    covariance_repaired: bool = False
    covariance_method_used: Optional[str] = None
    expected_return_method_used: Optional[str] = None
    strategy_diagnostics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    data_load_time: float = 0.0
    optimization_time: float = 0.0
    universe_used: Optional[str] = None

    def add_info(self, message: str) -> None:
        logger.info(message)
        self.info.append(message)

    def add_warning(self, message: str) -> None:
        logger.warning(message)
        self.warnings_list.append(message)

    def add_error(self, message: str) -> None:
        logger.error(message)
        self.errors_list.append(message)

    def summary(self) -> Dict[str, List[str]]:
        return {
            "warnings": self.warnings_list,
            "errors": self.errors_list,
            "info": self.info,
        }


@dataclass
class StrategyResult:
    weights: Dict[str, float]
    method: str
    description: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _normalize_weights_dict(weights: Dict[str, float]) -> Dict[str, float]:
    s = pd.Series(weights, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total = float(s.sum())
    if abs(total) < 1e-12:
        n = len(s)
        return {k: 1.0 / n for k in s.index} if n > 0 else {}
    s = s / total
    return {k: float(v) for k, v in s.items()}


def _drawdown_series(return_series: pd.Series) -> pd.Series:
    wealth = (1.0 + return_series.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


def _annual_factor(config) -> int:
    return int(getattr(config, "annual_trading_days", 252))


# =========================================================
# Data manager
# =========================================================
class ProfessionalDataManager:
    """
    Resilient Yahoo Finance loader with universe fallback logic.
    """

    def __init__(self, config, diagnostics: RunDiagnostics):
        self.config = config
        self.diagnostics = diagnostics

        self.asset_prices = pd.DataFrame()
        self.asset_returns = pd.DataFrame()
        self.benchmark_prices = pd.Series(dtype=float)
        self.benchmark_returns = pd.Series(dtype=float)
        self.data_quality = pd.DataFrame()
        self.asset_metadata = pd.DataFrame()
        self.instruments_list: List[str] = []

    # -----------------------------------------------------
    # Download helpers
    # -----------------------------------------------------
    def _download_single(self, ticker: str) -> tuple[pd.Series, Dict[str, Any]]:
        last_error = None

        for attempt in range(4):
            try:
                t = yf.Ticker(ticker)
                hist = t.history(
                    start=self.config.start_date,
                    end=self.config.end_date,
                    auto_adjust=True,
                )
                info = getattr(t, "info", {}) or {}

                if hist.empty or "Close" not in hist.columns:
                    raise ValueError("No close history returned")

                close = hist["Close"].copy()
                if getattr(close.index, "tz", None) is not None:
                    close.index = close.index.tz_localize(None)

                if close.dropna().empty:
                    raise ValueError("Close history is empty after cleaning")

                meta = {
                    "ticker": ticker,
                    "name": info.get("longName") or info.get("shortName") or ticker,
                    "exchange": info.get("exchange") or "",
                    "currency": info.get("currency") or "",
                    "type": info.get("quoteType") or "",
                    "source": "yahoo",
                }
                return close, meta

            except Exception as exc:
                last_error = exc
                time.sleep(1.0 + attempt * 1.2 + random.uniform(0.2, 0.8))

        raise ValueError(str(last_error))

    def _try_load_universe(self, tickers: List[str], universe_name: str) -> bool:
        rows = []
        meta_rows = []
        price_map = {}

        self.diagnostics.add_info(
            f"Attempting universe '{universe_name}' with {len(tickers)} instruments."
        )

        for ticker in tickers:
            try:
                close, meta = self._download_single(ticker)

                valid_ratio = float(close.notna().mean())
                if valid_ratio < 0.70:
                    self.diagnostics.dropped_assets[ticker] = f"Insufficient history ({valid_ratio:.1%})"
                    continue

                price_map[ticker] = close.rename(ticker)

                rows.append(
                    {
                        "ticker": ticker,
                        "valid_ratio": valid_ratio,
                        "observations": int(close.notna().sum()),
                    }
                )

                meta_rows.append(
                    {
                        "category": self.config.asset_categories.get(ticker, "SelectedUniverse"),
                        "ticker": ticker,
                        "name": meta.get("name", ticker),
                        "exchange": meta.get("exchange", ""),
                        "currency": meta.get("currency", ""),
                        "type": meta.get("type", ""),
                        "source": meta.get("source", ""),
                    }
                )

                time.sleep(0.35 + random.uniform(0.05, 0.25))

            except Exception as exc:
                self.diagnostics.dropped_assets[ticker] = str(exc)
                meta_rows.append(
                    {
                        "category": self.config.asset_categories.get(ticker, "SelectedUniverse"),
                        "ticker": ticker,
                        "name": ticker,
                        "exchange": "",
                        "currency": "",
                        "type": "",
                        "source": "failed",
                    }
                )

        if len(price_map) < 2:
            self.diagnostics.add_warning(
                f"Universe '{universe_name}' failed at download stage. "
                f"Usable assets: {len(price_map)}"
            )
            return False

        prices = pd.concat(price_map.values(), axis=1).sort_index()
        prices = prices.ffill(limit=3).dropna(how="all")

        returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
        returns = returns.dropna(axis=1, thresh=max(20, int(0.6 * len(returns))))

        if returns.shape[1] < 2:
            self.diagnostics.add_warning(
                f"Universe '{universe_name}' failed after return cleaning. "
                f"Usable assets: {returns.shape[1]}"
            )
            return False

        prices = prices[returns.columns].copy()

        self.asset_prices = prices.loc[returns.index]
        self.asset_returns = returns
        self.data_quality = pd.DataFrame(rows).sort_values("ticker") if rows else pd.DataFrame()
        self.asset_metadata = pd.DataFrame(meta_rows).sort_values(["category", "ticker"]) if meta_rows else pd.DataFrame()
        self.instruments_list = list(returns.columns)
        self.diagnostics.universe_used = universe_name

        self.diagnostics.add_info(
            f"Universe '{universe_name}' loaded successfully with {returns.shape[1]} assets."
        )
        return True

    def _fallback_candidates(self) -> List[tuple[str, List[str]]]:
        """
        Ordered fallback list. Starts from currently selected universe, then
        progressively safer alternatives.
        """
        selected_name = getattr(self.config, "selected_universe", "institutional_multi_asset")
        selected_assets = list(getattr(self.config, "assets", []))

        candidates = [(selected_name, selected_assets)]

        safe_defaults = [
            "institutional_multi_asset",
            "balanced_60_40_plus",
            "major_indices_proxy",
        ]

        for name in safe_defaults:
            if name != selected_name:
                try:
                    from core.universes import get_universe_tickers
                    tickers = get_universe_tickers(name)
                    if len(tickers) >= 2:
                        candidates.append((name, tickers))
                except Exception:
                    continue

        # Remove duplicates by universe name
        seen = set()
        unique = []
        for name, tickers in candidates:
            if name not in seen and len(tickers) >= 2:
                seen.add(name)
                unique.append((name, tickers))
        return unique

    def load(self) -> None:
        start_time = time.time()

        candidates = self._fallback_candidates()
        success = False

        for universe_name, tickers in candidates:
            if self._try_load_universe(tickers, universe_name):
                success = True
                self.config.selected_universe = universe_name
                self.config.asset_universe = {"SelectedUniverse": list(self.instruments_list)}
                break

        if not success:
            raise ValueError(
                "No usable asset price series could be downloaded from Yahoo Finance. "
                "Likely causes: temporary Yahoo throttling, too many requested instruments, "
                "or unstable cloud-side requests. Please retry with a smaller universe."
            )

        self._load_benchmark()
        self._align_all()

        if self.asset_returns.shape[0] < int(getattr(self.config, "min_observations", 60)):
            raise ValueError(
                f"Not enough observations after alignment: {self.asset_returns.shape[0]} < {self.config.min_observations}"
            )

        self.diagnostics.data_load_time = time.time() - start_time
        self.diagnostics.add_info(
            f"Loaded final universe '{self.config.selected_universe}' with {self.asset_returns.shape[1]} valid assets."
        )

    def _load_benchmark(self) -> None:
        try:
            b = yf.Ticker(self.config.benchmark).history(
                start=self.config.start_date,
                end=self.config.end_date,
                auto_adjust=True,
            )["Close"]

            if getattr(b.index, "tz", None) is not None:
                b.index = b.index.tz_localize(None)

            self.benchmark_prices = b
            self.benchmark_returns = b.pct_change().dropna()
            self.diagnostics.benchmark_used = self.config.benchmark

        except Exception:
            proxy = self.asset_returns.mean(axis=1)
            self.benchmark_returns = proxy
            self.benchmark_prices = (1 + proxy).cumprod()
            self.diagnostics.benchmark_used = "EqualWeightProxy"
            self.diagnostics.add_warning("Benchmark download failed. EqualWeightProxy is being used.")

    def _align_all(self) -> None:
        common = self.asset_returns.index.intersection(self.benchmark_returns.index)
        self.asset_returns = self.asset_returns.loc[common]
        self.asset_prices = self.asset_prices.loc[common]
        self.benchmark_returns = self.benchmark_returns.loc[common]

        if not self.benchmark_prices.empty:
            self.benchmark_prices = self.benchmark_prices.loc[self.benchmark_prices.index.intersection(common)]


# =========================================================
# Inputs
# =========================================================
class ModelInputBuilder:
    def __init__(self, config, diagnostics: RunDiagnostics):
        self.config = config
        self.diagnostics = diagnostics

    def build_expected_returns(self, returns: pd.DataFrame, benchmark_returns: pd.Series) -> pd.Series:
        self.diagnostics.expected_return_method_used = self.config.expected_return_method
        price_like = (1 + returns).cumprod()

        if HAS_PYPORTFOLIOOPT:
            if self.config.expected_return_method == "historical_mean":
                mu = expected_returns.mean_historical_return(price_like, frequency=_annual_factor(self.config))
            elif self.config.expected_return_method == "capm":
                mu = expected_returns.capm_return(
                    price_like,
                    market_prices=(1 + benchmark_returns).cumprod().to_frame("benchmark"),
                    risk_free_rate=self.config.risk_free_rate,
                    frequency=_annual_factor(self.config),
                )
            else:
                mu = expected_returns.ema_historical_return(price_like, frequency=_annual_factor(self.config))
            return mu.replace([np.inf, -np.inf], np.nan).dropna()

        if self.config.expected_return_method == "ema_historical":
            return returns.ewm(span=60).mean().iloc[-1] * _annual_factor(self.config)

        return returns.mean() * _annual_factor(self.config)

    def build_covariance(self, prices: pd.DataFrame) -> pd.DataFrame:
        self.diagnostics.covariance_method_used = self.config.covariance_method

        if HAS_PYPORTFOLIOOPT:
            if self.config.covariance_method in {"sample_cov", "sample"}:
                cov = pypfopt_risk_models.sample_cov(prices, frequency=_annual_factor(self.config))
            elif self.config.covariance_method == "shrinkage":
                cov = pypfopt_risk_models.CovarianceShrinkage(prices, frequency=_annual_factor(self.config)).shrunk_covariance(0.2)
            else:
                cov = pypfopt_risk_models.CovarianceShrinkage(prices, frequency=_annual_factor(self.config)).ledoit_wolf()
        else:
            cov = prices.pct_change().dropna().cov() * _annual_factor(self.config)

        vals, vecs = np.linalg.eigh(cov.values)
        if vals.min() < 0:
            vals = np.clip(vals, 1e-10, None)
            cov = pd.DataFrame(vecs @ np.diag(vals) @ vecs.T, index=cov.index, columns=cov.columns)
            self.diagnostics.covariance_repaired = True

        return cov


# =========================================================
# Optimizer
# =========================================================
class ProfessionalOptimizer:
    def __init__(self, mu, cov, returns, benchmark_returns, config, diagnostics, current_weights=None):
        common = mu.index.intersection(cov.index).intersection(returns.columns)
        self.mu = mu.loc[common]
        self.cov = cov.loc[common, common]
        self.returns = returns[common]
        self.benchmark_returns = benchmark_returns.reindex(self.returns.index).dropna()
        self.returns = self.returns.loc[self.benchmark_returns.index]
        self.config = config
        self.diagnostics = diagnostics
        self.category_map = {k: v for k, v in config.asset_categories.items() if k in common}
        self.current_weights = pd.Series(current_weights or {a: 1 / len(common) for a in common}, dtype=float).reindex(common).fillna(0.0)
        self.current_weights = self.current_weights / self.current_weights.sum()

    def _make_ef(self, mu, cov):
        if not HAS_PYPORTFOLIOOPT:
            raise RuntimeError("PyPortfolioOpt not available")
        bounds = (-1.0, 1.0) if self.config.allow_short else (self.config.min_weight, self.config.max_weight)
        ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
        ef.add_objective(objective_functions.L2_reg, gamma=0.02)
        return ef

    def _norm(self, w):
        cleaned = {k: float(v) for k, v in w.items() if abs(v) > 1e-8}
        return _normalize_weights_dict(cleaned)

    def _attach_costs(self, result):
        wn = pd.Series(result.weights).reindex(self.mu.index).fillna(0.0)
        turnover = float(np.abs(wn - self.current_weights).sum())
        tc = turnover * (self.config.transaction_cost_bps / 10000.0) * self.config.initial_capital
        result.diagnostics.update({"turnover": turnover, "estimated_transaction_cost_usd": tc})
        return result

    def _validate(self, name, result):
        total = sum(result.weights.values())
        if not np.isclose(total, 1.0, atol=1e-4):
            raise ValueError(f"{name}: weights sum to {total:.6f}")

    def max_sharpe(self):
        if HAS_PYPORTFOLIOOPT:
            ef = self._make_ef(self.mu, self.cov)
            ef.max_sharpe(risk_free_rate=self.config.risk_free_rate)
            return StrategyResult(self._norm(ef.clean_weights()), "max_sharpe", "Sharpe-optimal portfolio")

        n = len(self.mu)
        x0 = np.ones(n) / n
        bounds = [(self.config.min_weight, self.config.max_weight)] * n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def obj(w):
            pret = float(np.dot(w, self.mu.values))
            pvol = float(np.sqrt(max(w @ self.cov.values @ w, 1e-12)))
            return -((pret - self.config.risk_free_rate) / pvol)

        res = minimize(obj, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
        if not res.success:
            raise RuntimeError(res.message)
        return StrategyResult(self._norm({a: float(w) for a, w in zip(self.mu.index, res.x)}), "max_sharpe", "Sharpe-optimal portfolio")

    def min_volatility(self):
        if HAS_PYPORTFOLIOOPT:
            ef = self._make_ef(self.mu, self.cov)
            ef.min_volatility()
            return StrategyResult(self._norm(ef.clean_weights()), "min_volatility", "Minimum volatility portfolio")

        n = len(self.mu)
        x0 = np.ones(n) / n
        bounds = [(self.config.min_weight, self.config.max_weight)] * n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def obj(w):
            return float(w @ self.cov.values @ w)

        res = minimize(obj, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
        if not res.success:
            raise RuntimeError(res.message)
        return StrategyResult(self._norm({a: float(w) for a, w in zip(self.mu.index, res.x)}), "min_volatility", "Minimum volatility portfolio")

    def equal_weight(self):
        n = len(self.mu)
        return StrategyResult({a: 1 / n for a in self.mu.index}, "equal_weight", "Equal-weight benchmark")

    def inverse_volatility(self):
        vol = np.sqrt(np.diag(self.cov.values))
        inv = 1 / np.maximum(vol, 1e-12)
        inv = inv / inv.sum()
        return StrategyResult({a: float(w) for a, w in zip(self.mu.index, inv)}, "inverse_volatility", "Inverse-volatility allocation")

    def equal_risk_contribution(self):
        n = len(self.mu)
        x0 = np.ones(n) / n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n

        def obj(w):
            s = self.cov.values
            pv = float(w @ s @ w)
            if pv <= 0:
                return 1e6
            rc = w * (s @ w) / np.sqrt(pv)
            target = np.mean(rc)
            turnover = np.sum(np.abs(w - self.current_weights.values))
            return float(np.sum((rc - target) ** 2) + self.config.turnover_penalty * turnover)

        res = minimize(obj, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
        if not res.success:
            raise RuntimeError(res.message)

        return StrategyResult(self._norm({a: float(w) for a, w in zip(self.mu.index, res.x)}), "erc", "Equal risk contribution", {"optimizer_status": res.message})

    def maximum_diversification(self):
        n = len(self.mu)
        vols = np.sqrt(np.diag(self.cov.values))
        x0 = np.ones(n) / n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n

        def obj(w):
            denom = float(np.sqrt(max(w @ self.cov.values @ w, 1e-12)))
            turnover = np.sum(np.abs(w - self.current_weights.values))
            return float(-(w @ vols) / denom + self.config.turnover_penalty * turnover)

        res = minimize(obj, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
        if not res.success:
            raise RuntimeError(res.message)

        return StrategyResult(self._norm({a: float(w) for a, w in zip(self.mu.index, res.x)}), "max_diversification", "Maximum diversification", {"optimizer_status": res.message})

    def hrp(self):
        corr = self.returns.corr().clip(-1, 1)
        dist = np.sqrt((1 - corr) / 2)
        condensed = squareform(dist.values, checks=False)
        link = hierarchy.linkage(condensed, method="ward")
        ordered_idx = self._get_quasi_diag(link)
        ordered_assets = corr.index[ordered_idx].tolist()
        weights = pd.Series(1.0 / len(ordered_assets), index=ordered_assets)

        def cluster_variance(items):
            cov_slice = self.cov.loc[items, items]
            inv_diag = 1 / np.maximum(np.diag(cov_slice.values), 1e-12)
            parity_w = inv_diag / inv_diag.sum()
            return float(parity_w @ cov_slice.values @ parity_w)

        def bisect(items, w):
            if len(items) <= 1:
                return w
            mid = len(items) // 2
            left = items[:mid]
            right = items[mid:]
            v_left = cluster_variance(left)
            v_right = cluster_variance(right)
            alpha = 1 - v_left / (v_left + v_right) if (v_left + v_right) > 0 else 0.5
            w[left] *= alpha
            w[right] *= 1 - alpha
            w = bisect(left, w)
            w = bisect(right, w)
            return w

        weights = bisect(ordered_assets, weights)
        weights = weights / weights.sum()
        return StrategyResult(self._norm(weights.to_dict()), "hrp", "Hierarchical Risk Parity", {"hrp_order": ordered_assets})

    def black_litterman(self):
        if not HAS_PYPORTFOLIOOPT:
            return self.max_sharpe()

        benchmark_prices = (1 + self.benchmark_returns).cumprod().rename("benchmark")
        delta = market_implied_risk_aversion(benchmark_prices.to_frame(), frequency=self.config.annual_trading_days)
        w_mkt = pd.Series(1.0, index=self.mu.index) / len(self.mu)
        pi = delta * (self.cov @ w_mkt)
        views = {}
        if "GLD" in self.mu.index:
            views["GLD"] = float(max(self.mu.loc["GLD"], 0.03))
        if "QQQ" in self.mu.index:
            views["QQQ"] = float(max(self.mu.loc["QQQ"], 0.05))

        bl = BlackLittermanModel(self.cov, pi=pi, absolute_views=views if views else None, tau=0.05)
        ef = self._make_ef(bl.bl_returns(), bl.bl_cov())
        ef.max_sharpe(risk_free_rate=self.config.risk_free_rate)

        return StrategyResult(self._norm(ef.clean_weights()), "black_litterman", "Black-Litterman", {"views_used": views})

    def tracking_error_optimal(self):
        assets = list(self.mu.index)
        benchmark_proxy = self._benchmark_proxy_weights(assets)
        x0 = benchmark_proxy.values.copy()
        bounds = [(self.config.min_weight, self.config.max_weight)] * len(assets)
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        b = self.benchmark_returns.values
        r = self.returns[assets].values
        mu = self.mu.values

        def obj(w):
            port = r @ w
            te = np.std(port - b) * np.sqrt(self.config.annual_trading_days)
            active_return = float(np.dot(w - benchmark_proxy.values, mu))
            turnover = np.sum(np.abs(w - self.current_weights.values))
            te_penalty = 0.0
            if te > self.config.tracking_error_target:
                te_penalty = 50.0 * (te - self.config.tracking_error_target) ** 2
            return -active_return + te_penalty + self.config.turnover_penalty * turnover

        res = minimize(obj, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
        if not res.success:
            raise RuntimeError(res.message)

        realized_te = float(np.std((r @ res.x) - b) * np.sqrt(self.config.annual_trading_days))
        weights = {a: float(w) for a, w in zip(assets, res.x)}
        return StrategyResult(
            self._norm(weights),
            "tracking_error_optimal",
            "Tracking-error-constrained active portfolio",
            {
                "tracking_error_target": self.config.tracking_error_target,
                "ex_ante_tracking_error": realized_te
            }
        )

    def _benchmark_proxy_weights(self, assets):
        if self.config.benchmark in assets:
            w = pd.Series(0.0, index=assets)
            w[self.config.benchmark] = 1.0
            return w
        if "SPY" in assets:
            w = pd.Series(0.0, index=assets)
            w["SPY"] = 1.0
            return w
        return pd.Series(1 / len(assets), index=assets)

    def _get_quasi_diag(self, link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = int(link[-1, 3])

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix.loc[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()

    def run_all(self):
        start_time = time.time()

        methods = {
            "Max Sharpe": self.max_sharpe,
            "Minimum Volatility": self.min_volatility,
            "Equal Weight": self.equal_weight,
            "Inverse Volatility": self.inverse_volatility,
            "Equal Risk Contribution": self.equal_risk_contribution,
            "Maximum Diversification": self.maximum_diversification,
            "Hierarchical Risk Parity": self.hrp,
            "Black-Litterman": self.black_litterman,
            "Tracking Error Optimal": self.tracking_error_optimal,
        }

        out = {}
        for name, fn in methods.items():
            try:
                r = self._attach_costs(fn())
                self._validate(name, r)
                out[name] = r
                self.diagnostics.strategy_diagnostics[name] = r.diagnostics
                logger.info(f"✅ {name}")
            except Exception as exc:
                logger.warning(f"⚠️ {name}: {exc}")

        if not out:
            raise ValueError("All strategies failed")

        self.diagnostics.optimization_time = time.time() - start_time
        return out


# =========================================================
# Analytics
# =========================================================
class AnalyticsEngine:
    def __init__(self, config):
        self.config = config

    def portfolio_returns(self, returns, weights):
        w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
        w = w / w.sum()
        return returns.mul(w, axis=1).sum(axis=1)

    def portfolio_values(self, returns, initial_capital):
        return (1 + returns).cumprod() * initial_capital

    def calculate_var_family(self, portfolio_returns, benchmark_returns):
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]
        active = aligned["portfolio"] - aligned["benchmark"]

        out = {}
        for cl in self.config.confidence_levels:
            q_port = np.quantile(aligned["portfolio"], 1 - cl)
            q_act = np.quantile(active, 1 - cl)
            tail_port = aligned["portfolio"][aligned["portfolio"] <= q_port]
            tail_act = active[active <= q_act]

            out[f"var_{int(cl*100)}"] = float(-q_port)
            out[f"cvar_{int(cl*100)}"] = float(-tail_port.mean()) if len(tail_port) else np.nan
            out[f"relative_var_{int(cl*100)}"] = float(-q_act)
            out[f"relative_cvar_{int(cl*100)}"] = float(-tail_act.mean()) if len(tail_act) else np.nan

        return out

    def historical_stress_tests(self, portfolio_returns, benchmark_returns):
        scenarios = {
            "COVID Crash": ("2020-02-19", "2020-03-23"),
            "2022 Inflation Shock": ("2022-01-03", "2022-10-14"),
            "2023 Banking Stress": ("2023-03-08", "2023-03-31"),
            "2024 Q1 Rally": ("2024-01-02", "2024-03-28"),
        }

        rows = []
        pr = portfolio_returns.copy()
        br = benchmark_returns.copy()

        for name, (s, e) in scenarios.items():
            mask = (pr.index >= pd.Timestamp(s)) & (pr.index <= pd.Timestamp(e))
            p = pr.loc[mask]
            b = br.reindex(p.index).dropna()
            p = p.reindex(b.index)

            if len(p) < 5:
                continue

            p_total = float((1 + p).prod() - 1)
            b_total = float((1 + b).prod() - 1)
            rows.append({
                "scenario": name,
                "portfolio_return": p_total,
                "benchmark_return": b_total,
                "relative_return": p_total - b_total,
                "duration_days": len(p),
            })

        return pd.DataFrame(rows)

    def calculate_all_metrics(self, portfolio_returns, benchmark_returns, initial_capital):
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]

        portfolio_values = self.portfolio_values(aligned["portfolio"], initial_capital)
        benchmark_values = self.portfolio_values(aligned["benchmark"], initial_capital)

        final_portfolio_value = float(portfolio_values.iloc[-1])
        final_benchmark_value = float(benchmark_values.iloc[-1])

        total_return_portfolio = final_portfolio_value / initial_capital - 1
        total_return_benchmark = final_benchmark_value / initial_capital - 1

        n_years = len(aligned) / self.config.annual_trading_days
        annual_return_portfolio = (1 + total_return_portfolio) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        annual_return_benchmark = (1 + total_return_benchmark) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        daily_rf = self.config.risk_free_rate / self.config.annual_trading_days
        excess_returns = aligned["portfolio"] - daily_rf
        volatility = float(aligned["portfolio"].std() * np.sqrt(self.config.annual_trading_days))
        sharpe = float((excess_returns.mean() / aligned["portfolio"].std()) * np.sqrt(self.config.annual_trading_days)) if aligned["portfolio"].std() > 0 else 0.0

        downside_returns = aligned["portfolio"][aligned["portfolio"] < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.config.annual_trading_days) if len(downside_returns) > 0 else volatility
        sortino = (annual_return_portfolio - self.config.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0

        cumulative = (1 + aligned["portfolio"]).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1
        max_drawdown = float(drawdown.min())
        calmar = annual_return_portfolio / abs(max_drawdown) if max_drawdown < 0 else 0.0

        covariance = aligned["portfolio"].cov(aligned["benchmark"])
        benchmark_variance = aligned["benchmark"].var()
        beta = float(covariance / benchmark_variance) if benchmark_variance > 0 else 1.0
        alpha = float(annual_return_portfolio - (self.config.risk_free_rate + beta * (annual_return_benchmark - self.config.risk_free_rate)))

        tracking_diff = aligned["portfolio"] - aligned["benchmark"]
        tracking_error = float(tracking_diff.std() * np.sqrt(self.config.annual_trading_days))
        information_ratio = float((tracking_diff.mean() * self.config.annual_trading_days) / tracking_error) if tracking_error > 0 else 0.0

        winning_days = (aligned["portfolio"] > 0).sum()
        losing_days = (aligned["portfolio"] < 0).sum()
        win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0

        outperformance_days = (aligned["portfolio"] > aligned["benchmark"]).sum()
        total_days = len(aligned)
        win_rate_vs_benchmark = outperformance_days / total_days if total_days > 0 else 0

        avg_win = aligned["portfolio"][aligned["portfolio"] > 0].mean() if winning_days > 0 else 0
        avg_loss = aligned["portfolio"][aligned["portfolio"] < 0].mean() if losing_days > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        var_family = self.calculate_var_family(aligned["portfolio"], aligned["benchmark"])

        return {
            "initial_capital": initial_capital,
            "final_portfolio_value": final_portfolio_value,
            "final_benchmark_value": final_benchmark_value,
            "total_profit_loss": final_portfolio_value - initial_capital,
            "total_return_pct": total_return_portfolio,
            "total_return_benchmark_pct": total_return_benchmark,
            "excess_return_vs_benchmark_pct": total_return_portfolio - total_return_benchmark,
            "annual_return": annual_return_portfolio,
            "annual_return_benchmark": annual_return_benchmark,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_drawdown,
            "alpha": alpha,
            "beta": beta,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "win_rate": win_rate,
            "win_rate_vs_benchmark": win_rate_vs_benchmark,
            "profit_factor": profit_factor,
            "portfolio_values": portfolio_values,
            "benchmark_values": benchmark_values,
            "drawdown_series": drawdown,
            "portfolio_returns": aligned["portfolio"],
            "benchmark_returns": aligned["benchmark"],
            **var_family,
        }

    def compute_risk_contributions(self, weights: Dict[str, float], cov: pd.DataFrame) -> pd.DataFrame:
        assets = list(weights.keys())
        w = np.array([weights[a] for a in assets])
        S = cov.loc[assets, assets].values
        port_var = w @ S @ w

        if port_var <= 0:
            return pd.DataFrame({
                "Asset": assets,
                "Weight": w,
                "Marginal Risk": 0.0,
                "Total Risk Contribution": 0.0,
                "Contribution %": 0.0
            })

        port_vol = np.sqrt(port_var)
        marginal_risk = (S @ w) / port_vol
        total_contrib = w * marginal_risk
        total_contrib = total_contrib / total_contrib.sum() * port_vol if total_contrib.sum() != 0 else total_contrib

        df = pd.DataFrame({
            "Asset": assets,
            "Weight": w,
            "Marginal Risk": marginal_risk,
            "Total Risk Contribution": total_contrib,
            "Contribution %": total_contrib / port_vol if port_vol > 0 else 0
        })
        return df.sort_values("Total Risk Contribution", ascending=False)

    def rolling_beta_table(self, returns_df: pd.DataFrame, benchmark_returns: pd.Series, window: int) -> pd.DataFrame:
        aligned = pd.concat([returns_df, benchmark_returns.rename("benchmark")], axis=1).dropna()
        if aligned.empty:
            return pd.DataFrame()

        out = {}
        for col in returns_df.columns:
            pair = aligned[[col, "benchmark"]].dropna()
            if len(pair) < window:
                continue
            cov = pair[col].rolling(window).cov(pair["benchmark"])
            varb = pair["benchmark"].rolling(window).var()
            beta = cov / varb.replace(0, np.nan)
            out[col] = beta
        return pd.DataFrame(out)

    def rolling_beta_summary(self, rolling_beta_df: pd.DataFrame) -> pd.DataFrame:
        if rolling_beta_df.empty:
            return pd.DataFrame()
        return pd.DataFrame({
            "mean_beta": rolling_beta_df.mean(),
            "min_beta": rolling_beta_df.min(),
            "max_beta": rolling_beta_df.max(),
            "latest_beta": rolling_beta_df.iloc[-1],
        })

    def factor_pca(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        clean = returns_df.dropna(how="any")
        if clean.shape[1] < 2 or clean.shape[0] < 20:
            return pd.DataFrame()

        n_components = min(3, clean.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(clean.values)

        loading_df = pd.DataFrame(
            pca.components_.T,
            index=clean.columns,
            columns=[f"PC{i+1}" for i in range(n_components)],
        ).reset_index().rename(columns={"index": "factor_or_asset"})

        explained_rows = []
        for i in range(n_components):
            row = {"factor_or_asset": f"PC{i+1}_explained"}
            for j in range(n_components):
                row[f"PC{j+1}"] = pca.explained_variance_ratio_[i] if i == j else np.nan
            explained_rows.append(row)

        return pd.concat([loading_df, pd.DataFrame(explained_rows)], ignore_index=True)


# =========================================================
# Visualization
# =========================================================
class VisualizationEngine:
    def __init__(self, config):
        self.config = config

    def info_hub_table(self, asset_metadata):
        if asset_metadata.empty:
            return go.Figure()
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(asset_metadata.columns), fill_color="#1a3a5c", font=dict(color="white", size=12), align="left"),
            cells=dict(values=[asset_metadata[c] for c in asset_metadata.columns], fill_color="white", align="left", font=dict(size=11))
        )])
        fig.update_layout(title="Investment Universe Identity Map", height=max(420, 28 * len(asset_metadata) + 80))
        return fig

    def equity_curve_chart(self, portfolio_values, benchmark_values, strategy_name, initial_capital):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_values.index, y=portfolio_values.values, mode="lines", name=f"{strategy_name} Portfolio"))
        fig.add_trace(go.Scatter(x=benchmark_values.index, y=benchmark_values.values, mode="lines", name=f"Benchmark ({self.config.benchmark})"))
        fig.update_layout(title="Portfolio vs Benchmark Equity Curve", template="plotly_white", height=500)
        return fig

    def benchmark_vs_tracking_error_curve(self, strategy_returns, benchmark_returns, strategy_name):
        aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]
        rolling_te = (aligned["portfolio"] - aligned["benchmark"]).rolling(63).std() * np.sqrt(252)
        cum_port = (1 + aligned["portfolio"]).cumprod() - 1
        cum_bench = (1 + aligned["benchmark"]).cumprod() - 1

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port.values, mode="lines", name=f"{strategy_name} Cum Return"), secondary_y=False)
        fig.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench.values, mode="lines", name="Benchmark Cum Return"), secondary_y=False)
        fig.add_trace(go.Scatter(x=rolling_te.index, y=rolling_te.values, mode="lines", name="Rolling Tracking Error (63D)"), secondary_y=True)
        fig.update_layout(title="Benchmark vs Tracking-Error Optimal Dynamic Curve", template="plotly_white", height=500)
        fig.update_yaxes(tickformat=".0%", secondary_y=False)
        fig.update_yaxes(tickformat=".0%", secondary_y=True)
        return fig

    def drawdown_chart(self, drawdown_series, strategy_name):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series.values, mode="lines", fill="tozeroy", name=f"{strategy_name} Drawdown"))
        fig.update_layout(title="Drawdown Analysis", template="plotly_white", height=450)
        fig.update_yaxes(tickformat=".0%")
        return fig

    def allocation_chart(self, weights):
        s = pd.Series(weights).sort_values(ascending=False)
        fig = go.Figure([go.Bar(x=s.index[:15], y=s.values[:15], text=[f"{v:.1%}" for v in s.values[:15]], textposition="auto")])
        fig.update_layout(title="Top Strategy Allocation", template="plotly_white", height=450)
        fig.update_yaxes(tickformat=".0%")
        return fig

    def risk_contributions_chart(self, risk_df: pd.DataFrame):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=risk_df["Asset"], y=risk_df["Contribution %"], text=[f"{v:.2%}" for v in risk_df["Contribution %"]], textposition="auto"))
        fig.update_layout(title="Risk Contributions", template="plotly_white", height=500)
        fig.update_yaxes(tickformat=".0%")
        return fig

    def performance_dashboard(self, metrics_df):
        cols = [c for c in ["annual_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", "information_ratio", "win_rate_vs_benchmark"] if c in metrics_df.columns]
        fig = make_subplots(rows=2, cols=3, subplot_titles=tuple(cols[:6]))
        for idx, col in enumerate(cols[:6], start=1):
            row = 1 if idx <= 3 else 2
            c = idx if idx <= 3 else idx - 3
            fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df[col], name=col), row=row, col=c)
        fig.update_layout(title="Executive Strategy Dashboard", template="plotly_white", height=800, showlegend=False)
        return fig

    def improved_radar_chart(self, metrics_df):
        radar_metrics = [c for c in ["sharpe_ratio", "sortino_ratio", "information_ratio", "calmar_ratio", "win_rate_vs_benchmark", "annual_return"] if c in metrics_df.columns]
        if not radar_metrics:
            return go.Figure()
        df_norm = metrics_df[radar_metrics].copy()
        for col in radar_metrics:
            minv = df_norm[col].min()
            maxv = df_norm[col].max()
            df_norm[col] = (df_norm[col] - minv) / (maxv - minv) if maxv > minv else 0.5

        fig = go.Figure()
        for idx, row in df_norm.head(8).iterrows():
            fig.add_trace(go.Scatterpolar(r=row.values, theta=radar_metrics, fill="toself", name=idx))
        fig.update_layout(title="Strategy Performance Radar (Normalized)", template="plotly_white", height=700)
        return fig

    def optimization_chart(self, mu, cov, strategies, risk_free_rate):
        assets = list(mu.index)
        rng = np.random.default_rng(42)
        vols, rets = [], []
        for _ in range(700):
            w = rng.random(len(assets))
            w /= w.sum()
            vols.append(float(np.sqrt(w @ cov.values @ w)))
            rets.append(float(np.dot(w, mu.values)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vols, y=rets, mode="markers", name="Feasible Portfolios", opacity=0.25))
        for name, result in strategies.items():
            w = pd.Series(result.weights).reindex(assets).fillna(0.0).values
            v = float(np.sqrt(w @ cov.values @ w))
            r = float(np.dot(w, mu.values))
            fig.add_trace(go.Scatter(x=[v], y=[r], mode="markers+text", text=[name], textposition="top center", name=name))

        fig.update_layout(title="Portfolio Optimization & Efficient Frontier", template="plotly_white", height=580)
        fig.update_xaxes(tickformat=".0%", title="Annual Volatility")
        fig.update_yaxes(tickformat=".0%", title="Annual Return")
        return fig

    def tracking_error_chart(self, metrics_df):
        if "tracking_error" not in metrics_df.columns:
            return go.Figure()
        fig = go.Figure([go.Bar(x=metrics_df.index, y=metrics_df["tracking_error"], text=[f"{v:.2%}" for v in metrics_df["tracking_error"]], textposition="auto")])
        fig.update_layout(title="Tracking Error by Strategy", template="plotly_white", height=450)
        fig.update_yaxes(tickformat=".0%")
        return fig

    def relative_frontier_chart(self, mu, cov, strategies, benchmark_proxy):
        assets = list(mu.index)
        b = benchmark_proxy.reindex(assets).fillna(0.0).values
        rng = np.random.default_rng(42)
        xs, ys = [], []
        for _ in range(700):
            w = rng.random(len(assets))
            w /= w.sum()
            active_vol = float(np.sqrt(max((w - b) @ cov.values @ (w - b), 0.0)))
            excess_ret = float(np.dot(w - b, mu.values))
            xs.append(active_vol)
            ys.append(excess_ret)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Feasible Relative Portfolios", opacity=0.25))
        for name, result in strategies.items():
            w = pd.Series(result.weights).reindex(assets).fillna(0.0).values
            active_vol = float(np.sqrt(max((w - b) @ cov.values @ (w - b), 0.0)))
            excess_ret = float(np.dot(w - b, mu.values))
            fig.add_trace(go.Scatter(x=[active_vol], y=[excess_ret], mode="markers+text", text=[name], textposition="top center", name=name))
        fig.update_layout(title="Benchmark-Relative Efficient Frontier", template="plotly_white", height=580)
        fig.update_xaxes(tickformat=".0%", title="Tracking Error")
        fig.update_yaxes(tickformat=".0%", title="Excess Return")
        return fig

    def stress_test_chart(self, stress_df):
        if stress_df.empty:
            return go.Figure()
        xcol = "scenario" if "scenario" in stress_df.columns else "scenario_name"
        fig = go.Figure()
        fig.add_trace(go.Bar(x=stress_df[xcol], y=stress_df["portfolio_return"], name="Portfolio"))
        fig.add_trace(go.Bar(x=stress_df[xcol], y=stress_df["benchmark_return"], name="Benchmark"))
        fig.add_trace(go.Scatter(x=stress_df[xcol], y=stress_df["relative_return"], mode="lines+markers", name="Relative Return"))
        fig.update_layout(title="Historical Stress Testing", barmode="group", template="plotly_white", height=500)
        fig.update_yaxes(tickformat=".0%")
        return fig

    def var_family_chart(self, metrics_df, kind="absolute"):
        if kind == "absolute":
            cols = [c for c in ["var_95", "cvar_95"] if c in metrics_df.columns]
            title = "VaR / CVaR Figures (Absolute)"
        else:
            cols = [c for c in ["relative_var_95", "relative_cvar_95"] if c in metrics_df.columns]
            title = "Relative VaR / CVaR Figures (vs Benchmark)"
        if len(cols) < 2:
            return go.Figure()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df[cols[0]], name=cols[0]))
        fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df[cols[1]], name=cols[1]))
        fig.update_layout(title=title, template="plotly_white", height=500)
        fig.update_yaxes(tickformat=".0%")
        return fig

    def rolling_beta_chart(self, rolling_beta_df: pd.DataFrame):
        if rolling_beta_df.empty:
            return go.Figure()
        fig = go.Figure()
        for col in rolling_beta_df.columns[: min(5, len(rolling_beta_df.columns))]:
            fig.add_trace(go.Scatter(x=rolling_beta_df.index, y=rolling_beta_df[col], mode="lines", name=col))
        fig.update_layout(title="Rolling 60-Day Beta vs Benchmark", template="plotly_white", height=500)
        return fig

    def finquant_ef_chart(self, returns_df, mean_returns, cov_matrix, risk_free_rate, mc_trials=6000):
        assets = mean_returns.index.tolist()
        mean_returns = pd.Series(mean_returns, dtype=float).reindex(assets)
        cov_matrix = cov_matrix.loc[assets, assets].astype(float)

        rng = np.random.default_rng(42)
        num_portfolios = int(min(max(mc_trials, 500), 6000))
        results = np.zeros((3, num_portfolios), dtype=float)

        for i in range(num_portfolios):
            w = rng.random(len(assets))
            w = w / np.sum(w)
            port_return = float(np.dot(w, mean_returns.values))
            port_vol = float(np.sqrt(max(w @ cov_matrix.values @ w, 0.0)))
            sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else np.nan
            results[0, i] = port_vol
            results[1, i] = port_return
            results[2, i] = sharpe if np.isfinite(sharpe) else np.nan

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results[0], y=results[1],
            mode="markers",
            name=f"Random Portfolios ({num_portfolios:,})",
            marker=dict(size=4, color=results[2], colorscale="Viridis", showscale=True),
        ))

        asset_vols = np.sqrt(np.diag(cov_matrix.values))
        fig.add_trace(go.Scatter(
            x=asset_vols, y=mean_returns.values,
            mode="markers+text",
            name="Individual Assets",
            text=assets,
            textposition="top center",
        ))

        fig.update_layout(
            title="FinQuant Efficient Frontier & Monte Carlo",
            xaxis_title="Annual Volatility (Risk)",
            yaxis_title="Annual Return",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
            template="plotly_white",
            height=650,
        )
        return fig

    def finquant_weights_table(self, weights_dict: Dict[str, float], title: str):
        df = pd.DataFrame(list(weights_dict.items()), columns=["Asset", "Weight"])
        df = df.sort_values("Weight", ascending=False)
        df["Weight %"] = df["Weight"].apply(lambda x: f"{x:.2%}")
        fig = go.Figure(data=[go.Table(
            header=dict(values=["Asset", "Weight"], fill_color="#1a3a5c", font=dict(color="white", size=12), align="left"),
            cells=dict(values=[df["Asset"], df["Weight %"]], fill_color="white", align="left", font=dict(size=11))
        )])
        fig.update_layout(title=title, height=420)
        return fig


# =========================================================
# Main engine
# =========================================================
class ProfessionalPortfolioEngine:
    def __init__(self, config, bl_controls: Optional[Dict[str, Any]] = None, scenario_controls: Optional[Dict[str, Any]] = None):
        self.config = config
        self.bl_controls = bl_controls or {}
        self.scenario_controls = scenario_controls or {}

        self.diagnostics = RunDiagnostics()
        self.data = ProfessionalDataManager(config, self.diagnostics)
        self.analytics = AnalyticsEngine(config)
        self.visual = VisualizationEngine(config)

        self.mu = pd.Series(dtype=float)
        self.cov = pd.DataFrame()
        self.strategies: Dict[str, StrategyResult] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.metrics_df = pd.DataFrame()
        self.strategy_df = pd.DataFrame()
        self.stress_df = pd.DataFrame()
        self.stress_table = pd.DataFrame()
        self.risk_contrib_df = pd.DataFrame()
        self.rolling_beta_df = pd.DataFrame()
        self.beta_summary_df = pd.DataFrame()
        self.factor_pca_df = pd.DataFrame()
        self.charts: Dict[str, Any] = {}
        self.finquant_charts: Dict[str, Any] = {}

        self.prices = pd.DataFrame()
        self.returns = pd.DataFrame()

    def best_strategy_name(self) -> str:
        if self.metrics_df.empty:
            raise ValueError("No strategy metrics are available.")
        return str(self.metrics_df.index[0])

    def _build_chart_package(self) -> None:
        if self.metrics_df.empty:
            self.charts = {}
            return

        best_strategy = self.best_strategy_name()
        best_metrics = self.metrics[best_strategy]
        best_weights = self.strategies[best_strategy].weights

        benchmark_proxy = pd.Series(0.0, index=self.mu.index)
        if self.config.benchmark in benchmark_proxy.index:
            benchmark_proxy[self.config.benchmark] = 1.0
        elif "SPY" in benchmark_proxy.index:
            benchmark_proxy["SPY"] = 1.0
        else:
            benchmark_proxy[:] = 1.0 / len(benchmark_proxy)

        self.charts = {
            "info_hub": self.visual.info_hub_table(self.data.asset_metadata),
            "equity": self.visual.equity_curve_chart(
                best_metrics["portfolio_values"],
                best_metrics["benchmark_values"],
                best_strategy,
                self.config.initial_capital,
            ),
            "benchmark_vs_te": self.visual.benchmark_vs_tracking_error_curve(
                best_metrics["portfolio_returns"],
                best_metrics["benchmark_returns"],
                best_strategy,
            ),
            "drawdown": self.visual.drawdown_chart(best_metrics["drawdown_series"], best_strategy),
            "allocation": self.visual.allocation_chart(best_weights),
            "risk_contrib": self.visual.risk_contributions_chart(self.risk_contrib_df) if not self.risk_contrib_df.empty else None,
            "dashboard": self.visual.performance_dashboard(self.metrics_df),
            "radar": self.visual.improved_radar_chart(self.metrics_df),
            "optimization": self.visual.optimization_chart(
                self.mu,
                self.cov,
                self.strategies,
                self.config.risk_free_rate,
            ),
            "tracking_error": self.visual.tracking_error_chart(self.metrics_df),
            "relative_frontier": self.visual.relative_frontier_chart(
                self.mu,
                self.cov,
                self.strategies,
                benchmark_proxy,
            ),
            "stress": self.visual.stress_test_chart(self.stress_df),
            "absolute_var": self.visual.var_family_chart(self.metrics_df, kind="absolute"),
            "relative_var": self.visual.var_family_chart(self.metrics_df, kind="relative"),
            "rolling_beta": self.visual.rolling_beta_chart(self.rolling_beta_df),
        }

    def _build_finquant_outputs(self) -> None:
        if self.mu.empty or self.cov.empty or self.data.asset_returns.empty:
            self.finquant_charts = {}
            return

        try:
            ef_chart = self.visual.finquant_ef_chart(
                self.data.asset_returns[self.mu.index],
                self.mu,
                self.cov,
                self.config.risk_free_rate,
                mc_trials=getattr(self.config, "finquant_mc_trials", 6000),
            )

            min_vol_weights = {}
            max_sharpe_weights = {}

            if self.strategies:
                if "Minimum Volatility" in self.strategies:
                    min_vol_weights = self.strategies["Minimum Volatility"].weights
                else:
                    first = next(iter(self.strategies.values()))
                    min_vol_weights = first.weights

                if "Max Sharpe" in self.strategies:
                    max_sharpe_weights = self.strategies["Max Sharpe"].weights
                else:
                    first = next(iter(self.strategies.values()))
                    max_sharpe_weights = first.weights

            self.finquant_charts = {
                "ef_chart": ef_chart,
                "min_vol_table": self.visual.finquant_weights_table(min_vol_weights, "Reference Minimum Volatility Weights") if min_vol_weights else None,
                "max_sharpe_table": self.visual.finquant_weights_table(max_sharpe_weights, "Reference Max Sharpe Weights") if max_sharpe_weights else None,
            }
        except Exception as exc:
            self.finquant_charts = {}
            self.diagnostics.add_warning(f"FinQuant layer could not be built: {exc}")

    def run(self):
        self.data.load()

        self.prices = self.data.asset_prices.copy()
        self.returns = self.data.asset_returns.copy()

        builder = ModelInputBuilder(self.config, self.diagnostics)
        self.mu = builder.build_expected_returns(self.data.asset_returns, self.data.benchmark_returns)

        common_assets = self.mu.index.intersection(self.data.asset_prices.columns).intersection(self.data.asset_returns.columns)
        self.mu = self.mu.loc[common_assets]
        self.cov = builder.build_covariance(self.data.asset_prices[common_assets])

        optimizer = ProfessionalOptimizer(
            self.mu,
            self.cov,
            self.data.asset_returns[common_assets],
            self.data.benchmark_returns,
            self.config,
            self.diagnostics,
        )

        self.strategies = optimizer.run_all()
        logger.info(f"Optimized {len(self.strategies)} strategies")

        for name, result in self.strategies.items():
            pr = self.analytics.portfolio_returns(self.data.asset_returns[common_assets], result.weights)
            self.metrics[name] = self.analytics.calculate_all_metrics(
                pr,
                self.data.benchmark_returns,
                self.config.initial_capital,
            )
            self.metrics[name]["weights"] = pd.Series(result.weights)

        self.metrics_df = pd.DataFrame(self.metrics).T.sort_values("sharpe_ratio", ascending=False)

        self.strategy_df = pd.DataFrame(
            [
                {
                    "strategy": name,
                    "method": result.method,
                    "num_assets": len([w for w in result.weights.values() if w > 0.001]),
                    "max_weight": max(result.weights.values()),
                    "top_3_assets": ", ".join(
                        [f"{asset}" for asset, weight in sorted(result.weights.items(), key=lambda x: x[1], reverse=True)[:3]]
                    ),
                    "turnover": result.diagnostics.get("turnover", 0.0),
                    "transaction_cost": result.diagnostics.get("estimated_transaction_cost_usd", 0.0),
                    "tracking_error_target": result.diagnostics.get("tracking_error_target", np.nan),
                    "ex_ante_tracking_error": result.diagnostics.get("ex_ante_tracking_error", np.nan),
                }
                for name, result in self.strategies.items()
            ]
        )

        best_strategy = self.best_strategy_name()
        best_returns = self.metrics[best_strategy]["portfolio_returns"]
        best_benchmark = self.metrics[best_strategy]["benchmark_returns"]

        self.stress_df = self.analytics.historical_stress_tests(best_returns, best_benchmark)
        self.stress_table = self.stress_df.copy()

        best_weights = self.strategies[best_strategy].weights
        self.risk_contrib_df = self.analytics.compute_risk_contributions(best_weights, self.cov)

        self.rolling_beta_df = self.analytics.rolling_beta_table(
            self.data.asset_returns[self.mu.index],
            self.data.benchmark_returns,
            window=int(getattr(self.config, "rolling_window", 63)),
        )
        self.beta_summary_df = self.analytics.rolling_beta_summary(self.rolling_beta_df)
        self.factor_pca_df = self.analytics.factor_pca(self.data.asset_returns[self.mu.index])

        self._build_chart_package()
        self._build_finquant_outputs()

        self.diagnostics.add_info(f"Completed {len(self.stress_df)} stress tests")
        return self
