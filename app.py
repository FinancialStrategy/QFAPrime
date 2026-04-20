from __future__ import annotations
        ["ema_historical", "historical_mean", "capm"],
        index=0,
    )

    turnover_penalty = st.slider("Turnover Penalty", min_value=0.000, max_value=0.050, value=0.005, step=0.001)
    transaction_cost_bps = st.slider("Transaction Cost (bps)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    tracking_error_target = st.slider("Tracking Error Target", min_value=0.01, max_value=0.20, value=0.06, step=0.01)

    run_data_preview = st.button("Load Foundation Data", type="primary")


config = ProfessionalConfig(
    selected_universe=selected_universe,
    selected_region=selected_region,
    benchmark=benchmark,
    lookback_years=lookback_years,
    risk_free_rate=risk_free_rate,
    min_weight=min_weight,
    max_weight=max_weight,
    max_category_weight=max_category_weight,
    covariance_method=covariance_method,
    expected_return_method=expected_return_method,
    turnover_penalty=turnover_penalty,
    transaction_cost_bps=transaction_cost_bps,
    tracking_error_target=tracking_error_target,
)

diagnostics = RunDiagnostics()

if run_data_preview:
    try:
        data = ProfessionalDataManager(config, diagnostics)
        data.load()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Universe", config.selected_universe)
        col2.metric("Region", config.selected_region)
        col3.metric("Assets Loaded", len(data.asset_returns.columns))
        col4.metric("Benchmark", diagnostics.benchmark_used or config.benchmark)

        st.subheader("Investment Universe Metadata")
        st.dataframe(data.asset_metadata, use_container_width=True)

        st.subheader("Data Quality")
        st.dataframe(data.data_quality, use_container_width=True)

        st.subheader("Asset Price Sample")
        st.dataframe(data.asset_prices.tail(10), use_container_width=True)

        st.subheader("Asset Return Sample")
        st.dataframe(data.asset_returns.tail(10), use_container_width=True)

        if diagnostics.dropped_assets:
            st.subheader("Dropped Assets")
            dropped_df = pd.DataFrame(
                [{"ticker": k, "reason": v} for k, v in diagnostics.dropped_assets.items()]
            )
            st.dataframe(dropped_df, use_container_width=True)

    except Exception as exc:
        st.error(f"Foundation load failed: {exc}")
else:
    st.info(
        "This is the foundation stage of the Streamlit migration. Use the sidebar to choose the universe and load the data preview."
    )
