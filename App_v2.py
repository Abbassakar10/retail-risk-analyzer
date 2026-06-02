import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px

# --- UI SETUP & RETAIL VISUAL STYLING ---
st.set_page_config(page_title="Simple Portfolio Risk Analyzer", layout="wide", page_icon="📈")

# Inject Custom CSS for clear, clean dashboard cards
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

st.title("📊 Intuitive Portfolio Risk & Performance Analyzer")

# COMPLIANCE DISCLAIMER
st.warning("**DISCLAIMER: This platform is strictly for educational purposes. It does not provide financial, investment, or tax advice. All calculations are estimates based on past market data. Please consult a financial advisor before investing.**")

st.markdown("""
### **Step 1: Add Your Investments**
Tell us which stocks you own and how many shares you have. The system will look up the latest market prices for you.
*To find Indian stocks, add **.NS** for NSE (e.g., `HDFCBANK.NS`, `INFY.NS`) or **.BO** for BSE (e.g., `RELIANCE.BO`).*
""")

# --- PORTFOLIO INPUT TABLE ---
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
        "Ticker": st.column_config.TextColumn("Stock Ticker Symbol", required=True),
        "Quantity": st.column_config.NumberColumn("Number of Shares Owned", min_value=1, step=1, required=True),
    }
)

st.markdown("---")
# --- SIDEBAR: SIMPLIFIED ECONOMIC ADJUSTMENTS ---
st.sidebar.header("⚙️ Market Settings")
st.sidebar.markdown("We use standard economic estimates to calculate your expected returns. You can tweak them below:")

risk_free_slider = st.sidebar.slider(
    "Safe Return Rate (Govt Bonds %)", 
    min_value=0.0, max_value=15.0, value=7.0, step=0.1,
    help="The guaranteed return you can get from zero-risk fixed deposits or government bonds."
)
market_return_slider = st.sidebar.slider(
    "Expected Stock Market Growth (%)", 
    min_value=5.0, max_value=25.0, value=12.0, step=0.1,
    help="The long-term average growth estimate for the overall stock market index (like the Nifty 50)."
)

risk_free_rate = risk_free_slider / 100
expected_market_return = market_return_slider / 100
equity_risk_premium = expected_market_return - risk_free_rate

timeframe_choice = st.radio(
    "Lookback Period for Analysis:", 
    ["1 Year", "3 Years"], 
    horizontal=True,
    help="Should we look at the last 1 year of price data or the last 3 years to analyze the patterns?"
)

# --- ENGINE START ---
if st.button("Check My Portfolio Risk", type="primary"):
    
    portfolio_df = edited_df.dropna(subset=["Ticker", "Quantity"])
    portfolio_df = portfolio_df[portfolio_df["Quantity"] > 0]
    
    if portfolio_df.empty:
        st.warning("Please add at least one stock ticker and enter the number of shares you own.")
    else:
        with st.spinner("Connecting to market databases and analyzing your stocks..."):
            
            period_str = "1y" if timeframe_choice == "1 Year" else "3y"
            benchmarks = {'^NSEI': 'Nifty 50', '^BSESN': 'BSE Sensex'}
            tickers = portfolio_df["Ticker"].tolist()

            # Download data
            all_tickers = list(set(tickers + list(benchmarks.keys())))
            data = yf.download(all_tickers, period=period_str, progress=False)['Close']
            
            # Ticker validation checks
            invalid_tickers = [t for t in tickers if t not in data.columns or data[t].isna().all()]
            if invalid_tickers:
                st.error(f"🔍 **Ticker Error:** We couldn't find data for **{', '.join(invalid_tickers)}**. Please verify that the name matches Yahoo Finance formatting.")
                st.stop()
            
            # Data cleaning
            clean_data = data.ffill().dropna(how='all')
            latest_prices = clean_data.iloc[-1]
            returns = clean_data.pct_change(fill_method=None).dropna()
            
            portfolio_df['Latest Price (₹)'] = portfolio_df['Ticker'].map(latest_prices)
            portfolio_df['Total Investment Value (₹)'] = portfolio_df['Quantity'] * portfolio_df['Latest Price (₹)']
            
            total_value = portfolio_df['Total Investment Value (₹)'].sum()
            portfolio_df["Weight (%)"] = (portfolio_df['Total Investment Value (₹)'] / total_value) * 100
            weights = (portfolio_df["Weight (%)"] / 100).to_numpy()

            st.success("Successfully fetched current prices!")
            st.subheader("Current Portfolio Snapshot")
            st.dataframe(
                portfolio_df[['Ticker', 'Quantity', 'Latest Price (₹)', 'Total Investment Value (₹)', 'Weight (%)']],
                use_container_width=True, hide_index=True,
                column_config={
                    "Latest Price (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                    "Total Investment Value (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                    "Weight (%)": st.column_config.NumberColumn(format="%.1f%%")
                }
            )

            # Concentration Risk Framework
            max_weight = weights.max()
            heaviest_asset = portfolio_df.loc[portfolio_df["Weight (%)"].idxmax(), "Ticker"]

            st.markdown("---")
            if max_weight > 0.40:
                st.error(f"⚠️ **Too Many Eggs in One Basket:** **{heaviest_asset}** makes up **{max_weight*100:.1f}%** of your entire portfolio. If something bad happens to this single company, your whole savings could take a major hit. Consider spreading your money into index funds or other sectors.")
            else:
                st.success("✅ **Good Separation:** Your investments are nicely balanced. No single stock dominates more than 40% of your total portfolio value.")

            # Math engine computations
            portfolio_returns = returns[tickers]
            daily_portfolio_returns = portfolio_returns.dot(weights)
            correlation_matrix = portfolio_returns.corr()

            individual_volatilities = portfolio_returns.std() * np.sqrt(252) * 100
            vol_df = pd.DataFrame({
                "Ticker": individual_volatilities.index,
                "Price Swings (Volatility %)": individual_volatilities.values
            }).sort_values(by="Price Swings (Volatility %)", ascending=True)

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

            # Benchmark simulation math
            base_investment = 100000
            cumulative_returns = pd.DataFrame(index=returns.index)
            cumulative_returns['Your Portfolio'] = (1 + daily_portfolio_returns).cumprod() * base_investment
            cumulative_returns['Nifty 50 (NSE)'] = (1 + returns['^NSEI']).cumprod() * base_investment
            cumulative_returns['Sensex (BSE)'] = (1 + returns['^BSESN']).cumprod() * base_investment
            
            # Maximum Drawdown
            portfolio_peaks = cumulative_returns['Your Portfolio'].cummax()
            portfolio_drawdowns = (cumulative_returns['Your Portfolio'] - portfolio_peaks) / portfolio_peaks
            max_drawdown_pct = portfolio_drawdowns.min() * 100

            final_portfolio_val = cumulative_returns['Your Portfolio'].iloc[-1]
            final_nifty_val = cumulative_returns['Nifty 50 (NSE)'].iloc[-1]
            final_sensex_val = cumulative_returns['Sensex (BSE)'].iloc[-1]

            # --- DISPLAY USER-FRIENDLY DASHBOARD METRICS ---
            st.subheader("Your Simple Risk Scorecard")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Account Value", f"₹{total_value:,.0f}")
            with col2:
                st.metric("Estimated Next-Year Return", f"{expected_annual_return * 100:.1f}%")
            with col3:
                st.metric("Price Swings (Volatility)", f"{portfolio_volatility * 100:.1f}%")
            with col4:
                st.metric("Market Sensitivity (Beta)", f"{portfolio_beta:.2f}")
            with col5:
                st.metric("Worst Historic Crash", f"{abs(max_drawdown_pct):.1f}%", delta="Peak-to-Bottom Loss", delta_color="inverse")
                
            # Plain-English Explainer Collapse Panel
            with st.expander("💡 What do these scores actually mean? (Plain English Guide)"):
                st.markdown(f"""
                * **Estimated Next-Year Return:** Based on your settings, if the broad market grows at {market_return_slider}%, your specific combination of stocks is estimated to grow around **{expected_annual_return * 100:.1f}%** over a normal year.
                * **Price Swings (Volatility):** This tells you how bumpy the ride is. A score of **{portfolio_volatility * 100:.1f}%** means your account value can safely fluctuate up or down by this percentage in a typical year. Higher means a wilder rollercoaster.
                * **Market Sensitivity (Beta):** A score of **{portfolio_beta:.2f}** means your portfolio is **{'more sensitive' if portfolio_beta > 1 else 'less sensitive'}** than the general stock market. If the Nifty index moves up or down by 10%, your portfolio is expected to move by roughly **{portfolio_beta * 10:.1f}%**.
                * **Worst Historic Crash:** Over the lookback period selected, if you had bought your portfolio at its absolute highest point right before a major drop, your account would have temporary declined by **{abs(max_drawdown_pct):.1f}%** before recovering.
                """)

            # --- VISUAL BENCHMARK RACE ---
            st.markdown("---")
            st.subheader(f"🏃 The ₹1 Lakh Growth Race ({timeframe_choice} Lookback)")
            st.info("If you had put exactly ₹1,00,000 into your portfolio, the Nifty index, and the Sensex index back then, here is exactly how much money you would have today:")
            
            race_col1, race_col2, race_col3 = st.columns(3)
            with race_col1:
                st.metric("Your Portfolio Result", f"₹{final_portfolio_val:,.0f}", f"{final_portfolio_val - base_investment:+,.0f} Growth")
            with race_col2:
                st.metric("Nifty 50 Tracker", f"₹{final_nifty_val:,.0f}", f"{final_nifty_val - base_investment:+,.0f} Growth")
            with race_col3:
                st.metric("Sensex Tracker", f"₹{final_sensex_val:,.0f}", f"{final_sensex_val - base_investment:+,.0f} Growth")
            
            fig_line = px.line(
                cumulative_returns, 
                labels={'value': 'Account Value (₹)', 'Date': 'Date', 'variable': 'Investment Choice'},
                color_discrete_sequence=['#00FFA3', '#00B8FF', '#FF3366']
            )
            fig_line.update_layout(template="plotly_dark", yaxis_title='Growth of ₹1,00,000 Base', xaxis_title='')
            st.plotly_chart(fig_line, use_container_width=True)

            # --- CORRELATION AND INDIVIDUAL RISK VISUALS ---
            st.markdown("---")
            st.subheader("Under the Hood: Stock Interactions & Standalone Risk")
            
            matrix_col1, matrix_col2 = st.columns(2)
            
            with matrix_col1:
                st.markdown("##### **Do Your Stocks Move Together? (Co-Movement Table)**")
                fig_heat = px.imshow(
                    correlation_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1.0, zmax=1.0,
                    labels=dict(color="Co-Movement Strength")
                )
                fig_heat.update_layout(template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_heat, use_container_width=True)
                st.caption("How to read: Deep RED square blocks mean the two stocks move in lockstep (high concentration danger). Neutral or BLUE blocks mean they move independently, providing true diversification protection.")

            with matrix_col2:
                st.markdown("##### **Individual Stock Bumps (Which asset swings hardest?)**")
                fig_bar = px.bar(
                    vol_df, x='Price Swings (Volatility %)', y='Ticker', orientation='h',
                    color='Price Swings (Volatility %)', color_continuous_scale='Reds'
                )
                fig_bar.update_layout(template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")
            st.info(f"🛡️ **Safety Threshold Check:** Based on normal market cycles, there is a 95% mathematical probability that your portfolio value will not drop more than **₹{VaR_30_days:,.0f}** in a single month.")
