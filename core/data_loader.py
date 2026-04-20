from __future__ import annotations
                {
                    "region_type": base_meta.get("region_type", "Unknown"),
                    "ticker": ticker,
                    "identity": base_meta.get("identity", ticker),
                    "name": live_meta.get("name", ticker),
                    "exchange": live_meta.get("exchange", ""),
                    "currency": live_meta.get("currency", ""),
                    "type": live_meta.get("type", ""),
                }
            )

            if close is None:
                self.diagnostics.dropped_assets[ticker] = "No close history"
                continue

            valid_ratio = float(close.notna().mean())
            if valid_ratio < 0.80:
                self.diagnostics.dropped_assets[ticker] = f"Insufficient history ({valid_ratio:.1%})"
                continue

            price_map[ticker] = close.rename(ticker)
            quality_rows.append(
                {
                    "ticker": ticker,
                    "valid_ratio": valid_ratio,
                    "observations": int(close.notna().sum()),
                    "first_valid": close.first_valid_index(),
                    "last_valid": close.last_valid_index(),
                }
            )

        if len(price_map) < 5:
            raise ValueError("Too few assets survived data quality filtering.")

        prices = pd.concat(price_map.values(), axis=1).sort_index().ffill(limit=3).dropna()
        returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        self.asset_prices = prices.loc[returns.index]
        self.asset_returns = returns
        self.data_quality = pd.DataFrame(quality_rows).sort_values("ticker")
        self.asset_metadata = pd.DataFrame(metadata_rows).sort_values(["region_type", "ticker"])

        self._load_benchmark()
        self._align_all()

        self.diagnostics.add_info(
            f"Loaded {self.asset_returns.shape[1]} investable assets from {self.config.selected_universe}."
        )

    def _load_benchmark(self) -> None:
        try:
            benchmark_series = yf.Ticker(self.config.benchmark).history(
                start=self.config.start_date,
                end=self.config.end_date,
                auto_adjust=True,
            )["Close"]

            if getattr(benchmark_series.index, "tz", None) is not None:
                benchmark_series.index = benchmark_series.index.tz_localize(None)

            self.benchmark_prices = benchmark_series
            self.benchmark_returns = benchmark_series.pct_change().dropna()
            self.diagnostics.benchmark_used = self.config.benchmark

        except Exception:
            proxy = self.asset_returns.mean(axis=1)
            self.benchmark_returns = proxy
            self.benchmark_prices = (1 + proxy).cumprod()
            self.diagnostics.benchmark_used = "EqualWeightProxy"

    def _align_all(self) -> None:
        common = self.asset_returns.index.intersection(self.benchmark_returns.index)
        self.asset_returns = self.asset_returns.loc[common]
        self.asset_prices = self.asset_prices.loc[common]
        self.benchmark_returns = self.benchmark_returns.loc[common]
        if not self.benchmark_prices.empty:
            self.benchmark_prices = self.benchmark_prices.loc[self.benchmark_prices.index.intersection(common)]
