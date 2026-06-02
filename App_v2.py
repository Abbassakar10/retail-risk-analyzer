import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go

# --- UI SETUP & CUSTOM CSS ---
st.set_page_config(page_title="Institutional Risk Engine", layout="wide", page_icon="📈")

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #1E1E2E;
        border: 1px solid #333344;
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.4);
    }
    h1, h2, h3 {
        color: #E2E2E2 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Institutional Portfolio Risk Engine")

st.warning("**DISCLAIMER: This platform is strictly for educational and analytical purposes. The outputs provided do not constitute personalized financial, investment, or trading advice. Past performance is not indicative of future results.**")

# --- SIDEBAR MACRO CONTROLS ---
st.sidebar.header("🛠️ Macroeconomic Model Parameters")
st.sidebar.markdown("Adjust these variables to alter the forward-looking CAPM valuation model.")

risk_free_slider = st.sidebar.slider(
    "Risk-Free Rate (%)", 
    min_value=0.0, max_value=15.0, value=7.0, step=0.1,
    help="The baseline return of a zero-risk asset, typically backed by government bonds (e.g., Indian 10-Year G-Sec)."
)
market_return_slider = st.sidebar.slider(
    "Expected Market Return (%)", 
    min_value=5.0, max_value=25.0, value=12.0, step=0.1,
    help="The annualized forward-looking return expectation for the broad equity index (e.g., Nifty 50)."
)

# Convert sidebar percentages to decimals for the math backend
risk_free_rate = risk_free_slider / 100
expected_market_return = market_return_slider / 100
equity_risk_premium = expected_market_return - risk_free_rate

# --- MAIN PANEL: PORTFOLIO BUILDER ---
st.markdown("### **Step 1: Define Portfolio Composition**")
default_data = pd.DataFrame([
    {"Ticker": "RELIANCE.NS", "Quantity": 25},
    {"Ticker": "TCS.NS", "Quantity": 15},
    {"Ticker": "INFY.NS", "Quantity": 40},
])

edited_df = st.data_editor(
    default_data, 
    num_rows="dynamic", 
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Stock Ticker", required=True),
        "Quantity": st.column_config.NumberColumn("Quantity", min_value=1, step=1, required=True),
    }
)

timeframe_choice = st.radio(
    "Select Historical Analysis Timeframe:", 
    ["1 Year", "3 Years"], 
    horizontal=True
)

# --- THE RISK CORE ---
if st.button("Fetch Market Metrics & Execute Stress Test", type="primary"):
    
    portfolio_df = edited_df.dropna(subset=["Ticker", "Quantity"])
    portfolio_df = portfolio_df[portfolio_df["Quantity"] > 0]
    
    if portfolio_df.empty:
        st.warning("Please populate the asset grid with valid tickers and quantities.")
    else:
        with st.spinner("Accessing historical market data and calculating covariance structures..."):
            
            period_str = "1y" if timeframe_choice == "1 Year" else "3y"
            benchmarks = {'^NSEI': 'Nifty 50', '^BSESN': 'BSE Sensex'}
            tickers = portfolio_df["Ticker"].tolist()

            # 1. DOWNLOAD SYSTEM DATA
            all_tickers = list(set(tickers + list(benchmarks.keys())))
            data = yf.download(all_tickers, period=period_str, progress=False)['Close']
            
            # --- TICKER COMPLIANCE CHECK ---
            invalid_tickers = [t for t in tickers if t not in data.columns or data[t].isna().all()]
            if invalid_tickers:
                st.error(f"🚨 **Data Query Failure:** Ticker(s) **{', '.join(invalid_tickers)}** could not be resolved. Please verify formatting suffixes (.NS or .BO).")
                st.stop()
            
            # 2. DATA PROCESSING & CLEANING
            clean_data = data.ffill().dropna(how='all')
            latest_prices = clean_data.iloc[-1]
            returns = clean_data.pct_change(fill_method=None).dropna()
            
            portfolio_df['Latest Price (₹)'] = portfolio_df['Ticker'].map(latest_prices)
            portfolio_df['Total Value (₹)'] = portfolio_df['Quantity'] * portfolio_df['Latest Price (₹)']
            
            total_value = portfolio_df['Total Value (₹)'].sum()
            portfolio_df["Weight"] = portfolio_df['Total Value (₹)'] / total_value
            weights = portfolio_df["Weight"].to_numpy()

            st.success("Market price data synchronized.")
            st.subheader("Current Asset Valuation")
            st.dataframe(
                portfolio_df[['Ticker', 'Quantity', 'Latest Price (₹)', 'Total Value (₹)', 'Weight']],
                use_container_width=True, hide_index=True,
                column_config={
                    "Latest Price (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                    "Total Value (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                    "Weight": st.column_config.NumberColumn(format="%.2f")
                }
            )

            # 3. DIVERSIFICATION AND ASSET ALIGNMENT MATH
            max_weight = portfolio_df["Weight"].max()
            heaviest_asset = portfolio_df.loc[portfolio_df["Weight"].idxmax(), "Ticker"]

            st.markdown("---")
            if max_weight > 0.40:
                st.error(f"⚠️ **Concentration Alert:** Portfolio is heavily reliant on **{heaviest_asset}** ({max_weight*100:.1f}% total allocation). High concentration exposes capital to severe idiosyncratic risk shocks.")
            else:
                st.success("✅ **Concentration Check:** Asset allocation is structurally balanced across current positions (all holdings under 40%).")

            # --- STATISTICAL ESTIMATION ENGINE ---
            portfolio_returns = returns[tickers]
            daily_portfolio_returns = portfolio_returns.dot(weights)

            # Calculate Asset Correlation Matrix
            correlation_matrix = portfolio_returns.corr()

            # Volatility Profiling
            individual_volatilities = portfolio_returns.std() * np.sqrt(252) * 100
            vol_df = pd.DataFrame({
                "Ticker": individual_volatilities.index,
                "Annual Volatility (%)": individual_volatilities.values
            }).sort_values(by="Annual Volatility (%)", ascending=True)

            cov_matrix = portfolio_returns.cov() * 252
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)

            nifty_variance = returns['^NSEI'].var()
            betas = [returns[t].cov(returns['^NSEI']) / nifty_variance for t in tickers]
            portfolio_beta = np.dot(weights, betas)

            # CAPM expected return using the dynamic sidebar values
            expected_annual_return = risk_free_rate + (portfolio_beta * equity_risk_premium)
            sharpe_ratio = (expected_annual_return - risk_free_rate) / portfolio_volatility

            # Parametric VaR
            z_score = norm.ppf(1 - 0.95)
            daily_volatility = portfolio_volatility / np.sqrt(252)
            VaR_30_days = abs(z_score * daily_volatility * np.sqrt(30) * total_value)

            # --- PERFORMANCE BENCHMARKING & DRAWDOWN MATH ---
            base_investment = 100000
            cumulative_returns = pd.DataFrame(index=returns.index)
            cumulative_returns['Your Portfolio'] = (1 + daily_portfolio_returns).cumprod() * base_investment
            cumulative_returns['Nifty 50'] = (1 + returns['^NSEI']).cumprod() * base_investment
            cumulative_returns['BSE Sensex'] = (1 + returns['^BSESN']).cumprod() * base_investment
            
            # Maximum Drawdown Derivation
            # Drawdown = (Current Value - Historical Peak Value) / Historical Peak Value
            portfolio_peaks = cumulative_returns['Your Portfolio'].cummax()
            portfolio_drawdowns = (cumulative_returns['Your Portfolio'] - portfolio_peaks) / portfolio_peaks
            max_drawdown_pct = portfolio_drawdowns.min() * 100

            final_portfolio_val = cumulative_returns['Your Portfolio'].iloc[-1]
            final_nifty_val = cumulative_returns['Nifty 50'].iloc[-1]
            final_sensex_val = cumulative_returns['BSE Sensex'].iloc[-1]

            # --- DISPLAY DASHBOARD METRICS ---
            st.subheader("Performance & Risk Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Value", f"₹{total_value:,.0f}")
            with col2:
                st.metric("Expected Return (CAPM)", f"{expected_annual_return * 100:.1f}%", help="Calculated using dynamic macro inputs.")
            with col3:
                st.metric("Portfolio Volatility", f"{portfolio_volatility * 100:.1f}%")
            with col4:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            with col5:
                st.metric("Maximum Drawdown", f"{max_drawdown_pct:.1f}%", delta="Peak-to-Trough Loss", delta_color="inverse")
                
            with st.expander("💡 Metric Definitions"):
                st.markdown("""
                * **Expected Return (CAPM):** Forward-looking estimate calculated from the formula $E(R_p) = R_f + \\beta_p(E(R_m) - R_f)$ using your custom sidebar variables.
                * **Sharpe Ratio:** Measures your excess return per unit of volatility. A higher number implies superior risk-adjusted scaling.
                * **Maximum Drawdown:** The absolute worst peak-to-trough drop observed historically. It reflects the real-world tail pain an investor would have endured by entering the market at the local cyclical peak.
                """)

            # --- PERFORMANCE VISUALS ---
            st.markdown("---")
            st.subheader(f"🏃 Historical Growth vs. Benchmarks ({timeframe_choice} Horizon)")
            st.info("The visualization below models the historical performance trajectory of a base ₹1,00,000 deployment.")
            
            race_col1, race_col2, race_col3 = st.columns(3)
            with race_col1:
                st.metric("Your Custom Portfolio", f"₹{final_portfolio_val:,.0f}", f"{final_portfolio_val - base_investment:+,.0f} PnL")
            with race_col2:
                st.metric("Nifty 50 Index", f"₹{final_nifty_val:,.0f}", f"{final_nifty_val - base_investment:+,.0f} PnL")
            with race_col3:
                st.metric("BSE Sensex Index", f"₹{final_sensex_val:,.0f}", f"{final_sensex_val - base_investment:+,.0f} PnL")
            
            fig_line = px.line(
                cumulative_returns, 
                labels={'value': 'Portfolio Value (₹)', 'Date': 'Date', 'variable': 'Asset'},
                color_discrete_sequence=['#00FFA3', '#00B8FF', '#FF3366']
            )
            fig_line.update_layout(template="plotly_dark", yaxis_title='Value of ₹1,00,000 Base', xaxis_title='')
            st.plotly_chart(fig_line, use_container_width=True)

            # --- NEW INTERACTIVE MATRIX SECTION ---
            st.markdown("---")
            st.subheader("Asset Interactions & Dependency Profiles")
            
            matrix_col1, matrix_col2 = st.columns(2)
            
            with matrix_col1:
                st.markdown("##### **Asset Correlation Heatmap**")
                # Build an interactive annotated matrix plot
                fig_heat = px.imshow(
                    correlation_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r", # Red implies high correlation, Blue implies separation
                    zmin=-1.0, zmax=1.0,
                    labels=dict(color="Correlation Coefficient")
                )
                fig_heat.update_layout(template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_heat, use_container_width=True)
                st.caption("Interpret: Higher red clustering indicates co-dependent risk. Broad diversification requires mixing assets with lower or neutral (blue/white) cross-correlation values.")

            with matrix_col2:
                st.markdown("##### **Single Asset Volatility Profile**")
                fig_bar = px.bar(
                    vol_df, x='Annual Volatility (%)', y='Ticker', orientation='h',
                    color='Annual Volatility (%)', color_continuous_scale='Reds'
                )
                fig_bar.update_layout(template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")
            st.warning(f"🚨 **95% Value at Risk (VaR):** Under normal conditions, there is a statistical 95% probability that your total portfolio capital exposure will not experience a single 30-day loss exceeding **₹{VaR_30_days:,.0f}**.")
