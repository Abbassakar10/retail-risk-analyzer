import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px

# --- UI SETUP & CUSTOM CSS ---
st.set_page_config(page_title="Institutional Risk Engine", layout="wide", page_icon="📈")

# Inject Custom CSS to create "boxed" metric cards and a terminal feel
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background-color: #1E1E2E;
        border: 1px solid #333344;
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.4);
    }
    
    /* Make the main text headers pop slightly more */
    h1, h2, h3 {
        color: #E2E2E2 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Institutional Portfolio Risk Engine")

st.warning("**DISCLAIMER: This platform is strictly for educational and analytical purposes. The outputs provided do not constitute personalized financial, investment, or trading advice. Past performance is not indicative of future results.**")

st.markdown("""
**Step 1: Enter your assets below.** *Note: The engine will automatically fetch the latest market closing prices. Add **.NS** for NSE (e.g., `HDFCBANK.NS`) or **.BO** for BSE (e.g., `RELIANCE.BO`).*
""")

# --- DYNAMIC TABLE INPUT ---
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

st.markdown("---")
# --- TIMEFRAME SELECTOR ---
timeframe_choice = st.radio(
    "Select Historical Analysis Timeframe:", 
    ["1 Year", "3 Years"], 
    horizontal=True
)

# --- THE CALCULATIONS ---
if st.button("Fetch Latest Prices & Analyze Portfolio", type="primary"):
    
    portfolio_df = edited_df.dropna(subset=["Ticker", "Quantity"])
    portfolio_df = portfolio_df[portfolio_df["Quantity"] > 0]
    
    if portfolio_df.empty:
        st.warning("Please add at least one valid asset with quantity greater than 0.")
    else:
        with st.spinner("Connecting to market data and rendering risk models..."):
            
            period_str = "1y" if timeframe_choice == "1 Year" else "3y"
            benchmarks = {'^NSEI': 'Nifty 50', '^BSESN': 'BSE Sensex'}
            
            risk_free_rate = 0.07            
            expected_market_return = 0.12    
            equity_risk_premium = expected_market_return - risk_free_rate
            
            tickers = portfolio_df["Ticker"].tolist()

            # 1. FETCH MARKET DATA
            all_tickers = list(set(tickers + list(benchmarks.keys())))
            
            data = yf.download(all_tickers, period=period_str, progress=False)['Close']
            
            # --- ERROR HANDLING ---
            invalid_tickers = []
            for t in tickers:
                if t not in data.columns or data[t].isna().all():
                    invalid_tickers.append(t)
            
            if invalid_tickers:
                st.error(f"🚨 **Invalid Ticker(s) Detected:** We could not find market data for **{', '.join(invalid_tickers)}**. Please ensure you are using the correct Yahoo Finance format.")
                st.stop()
            
            # 2. ENRICH TABLE
            clean_data = data.ffill().dropna(how='all')
            latest_prices = clean_data.iloc[-1]
            returns = clean_data.pct_change(fill_method=None).dropna()
            
            portfolio_df['Latest Price (₹)'] = portfolio_df['Ticker'].map(latest_prices)
            portfolio_df['Total Value (₹)'] = portfolio_df['Quantity'] * portfolio_df['Latest Price (₹)']
            
            total_value = portfolio_df['Total Value (₹)'].sum()
            portfolio_df["Weight"] = portfolio_df['Total Value (₹)'] / total_value
            weights = portfolio_df["Weight"].to_numpy()

            st.success("Latest market prices fetched successfully!")
            st.subheader("Current Portfolio Valuation")
            
            st.dataframe(
                portfolio_df[['Ticker', 'Quantity', 'Latest Price (₹)', 'Total Value (₹)']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Latest Price (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                    "Total Value (₹)": st.column_config.NumberColumn(format="₹%.2f")
                }
            )

            max_weight = portfolio_df["Weight"].max()
            heaviest_asset = portfolio_df.loc[portfolio_df["Weight"].idxmax(), "Ticker"]

            st.markdown("---")
            if max_weight > 0.40:
                st.error(f"⚠️ **Concentration Alert:** Portfolio heavily reliant on **{heaviest_asset}** ({max_weight*100:.1f}% weight). Consider diversifying into broad Index Funds to mitigate single-asset risk.")
            else:
                st.success("✅ **Diversification Check:** Assets are reasonably balanced (No single position > 40%).")

            # --- MATH ENGINE ---
            portfolio_returns = returns[tickers]
            daily_portfolio_returns = portfolio_returns.dot(weights)

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

            expected_annual_return = risk_free_rate + (portfolio_beta * equity_risk_premium)
            sharpe_ratio = (expected_annual_return - risk_free_rate) / portfolio_volatility

            z_score = norm.ppf(1 - 0.95)
            daily_volatility = portfolio_volatility / np.sqrt(252)
            VaR_30_days = abs(z_score * daily_volatility * np.sqrt(30) * total_value)

            base_investment = 100000
            cumulative_returns = pd.DataFrame()
            cumulative_returns['Your Portfolio'] = (1 + daily_portfolio_returns).cumprod() * base_investment
            cumulative_returns['Nifty 50'] = (1 + returns['^NSEI']).cumprod() * base_investment
            cumulative_returns['BSE Sensex'] = (1 + returns['^BSESN']).cumprod() * base_investment
            
            final_portfolio_val = cumulative_returns['Your Portfolio'].iloc[-1]
            final_nifty_val = cumulative_returns['Nifty 50'].iloc[-1]
            final_sensex_val = cumulative_returns['BSE Sensex'].iloc[-1]

            # --- DISPLAY METRICS ---
            st.subheader("Institutional Risk Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Investment", f"₹{total_value:,.0f}")
            with col2:
                ret_color = "normal" if expected_annual_return > risk_free_rate else "inverse"
                st.metric("Expected Return (CAPM)", f"{expected_annual_return * 100:.1f}%", delta="Forward-Looking", delta_color="off")
            with col3:
                st.metric("Annual Volatility", f"{portfolio_volatility * 100:.1f}%")
            with col4:
                st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
            with col5:
                sharpe_color = "normal" if sharpe_ratio > 1 else "inverse"
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", delta="Risk Adjusted", delta_color=sharpe_color)
                
            with st.expander("💡 Metric Definitions"):
                st.markdown("""
                * **Expected Return (CAPM):** Forward-looking estimate based on risk-free rate (7%) and portfolio risk level (Beta).
                * **Sharpe Ratio:** Measures risk-adjusted return. >1.0 is healthy; <1.0 indicates excessive risk for the reward.
                * **Portfolio Beta:** Market sensitivity. Beta > 1.0 indicates higher volatility than the overall market.
                """)

            # --- DARK MODE VISUALS ---
            st.markdown("---")
            st.subheader(f"🏃 Performance Benchmarking: {timeframe_choice}")
            
            race_col1, race_col2, race_col3 = st.columns(3)
            with race_col1:
                st.metric("Your Custom Portfolio", f"₹{final_portfolio_val:,.0f}", f"{final_portfolio_val - base_investment:+,.0f} PnL")
            with race_col2:
                st.metric("Nifty 50 Index", f"₹{final_nifty_val:,.0f}", f"{final_nifty_val - base_investment:+,.0f} PnL")
            with race_col3:
                st.metric("BSE Sensex Index", f"₹{final_sensex_val:,.0f}", f"{final_sensex_val - base_investment:+,.0f} PnL")
            
            # High Contrast Line Chart
            fig_line = px.line(
                cumulative_returns, 
                labels={'value': 'Portfolio Value (₹)', 'Date': 'Date', 'variable': 'Asset'},
                color_discrete_sequence=['#00FFA3', '#00B8FF', '#FF3366'] # Terminal Green, Cyan, Magenta
            )
            fig_line.update_layout(template="plotly_dark", yaxis_title='Value of ₹1,00,000 Base', xaxis_title='')
            st.plotly_chart(fig_line, use_container_width=True)

            st.markdown("---")
            st.subheader("Risk & Allocation Breakdown")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                fig_pie = px.pie(portfolio_df, values='Total Value (₹)', names='Ticker', title='Capital Allocation', hole=0.5, color_discrete_sequence=px.colors.sequential.Teal)
                fig_pie.update_layout(template="plotly_dark")
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                fig_bar = px.bar(vol_df, x='Annual Volatility (%)', y='Ticker', orientation='h', title='Single Asset Volatility Profile', color='Annual Volatility (%)', color_continuous_scale='Reds')
                fig_bar.update_layout(template="plotly_dark")
                st.plotly_chart(fig_bar, use_container_width=True)
                
            st.warning(f"🚨 **95% Value at Risk (VaR):** Under normal market conditions, there is a 95% probability this portfolio will not lose more than **₹{VaR_30_days:,.0f}** in a single 30-day period.")
