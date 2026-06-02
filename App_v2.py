import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px

# --- UI SETUP & RETAIL VISUAL STYLING ---
st.set_page_config(page_title="Simple Portfolio Risk Analyzer", layout="wide", page_icon="📈")

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #1E1E2E;
        border: 1px solid #333344;
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.4);
    }
    h1, h2, h3 { color: #E2E2E2 !important; }
    /* Style the tabs to make them larger and more clickable */
    button[data-baseweb="tab"] { font-size: 18px !important; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Intuitive Portfolio Risk & Performance Analyzer")

st.warning("**DISCLAIMER: This platform is strictly for educational purposes. It does not provide financial, investment, or tax advice. All calculations are estimates based on past market data.**")

st.markdown("""
### **Step 1: Add Your Investments**
Tell us which stocks you own and how many shares you have. We'll pull the latest market prices for you.
*To find Indian stocks, add **.NS** for NSE (e.g., `HDFCBANK.NS`) or **.BO** for BSE (e.g., `RELIANCE.BO`).*
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

risk_free_slider = st.sidebar.slider("Safe Return Rate (Govt Bonds %)", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
market_return_slider = st.sidebar.slider("Expected Stock Market Growth (%)", min_value=5.0, max_value=25.0, value=12.0, step=0.1)

risk_free_rate = risk_free_slider / 100
expected_market_return = market_return_slider / 100
equity_risk_premium = expected_market_return - risk_free_rate

timeframe_choice = st.radio("Lookback Period for Analysis:", ["1 Year", "3 Years"], horizontal=True)

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

            all_tickers = list(set(tickers + list(benchmarks.keys())))
            data = yf.download(all_tickers, period=period_str, progress=False)['Close']
            
            invalid_tickers = [t for t in tickers if t not in data.columns or data[t].isna().all()]
            if invalid_tickers:
                st.error(f"🔍 **Ticker Error:** We couldn't find data for **{', '.join(invalid_tickers)}**. Please verify that the name matches Yahoo Finance formatting.")
                st.stop()
            
            clean_data = data.ffill().dropna(how='all')
            latest_prices = clean_data.iloc[-1]
            returns = clean_data.pct_change(fill_method=None).dropna()
            
            portfolio_df['Latest Price (₹)'] = portfolio_df['Ticker'].map(latest_prices)
            portfolio_df['Total Investment Value (₹)'] = portfolio_df['Quantity'] * portfolio_df['Latest Price (₹)']
            
            total_value = portfolio_df['Total Investment Value (₹)'].sum()
            portfolio_df["Weight (%)"] = (portfolio_df['Total Investment Value (₹)'] / total_value) * 100
            weights = (portfolio_df["Weight (%)"] / 100).to_numpy()

            max_weight = weights.max()
            heaviest_asset = portfolio_df.loc[portfolio_df["Weight (%)"].idxmax(), "Ticker"]

            portfolio_returns = returns[tickers]
            daily_portfolio_returns = portfolio_returns.dot(weights)

            individual_volatilities = portfolio_returns.std() * np.sqrt(252) * 100
            vol_df = pd.DataFrame({
                "Ticker": individual_volatilities.index,
                "Price Swings (Volatility %)": individual_volatilities.values
            }).sort_values(by="Price Swings (Volatility %)", ascending=True)

            cov_matrix = portfolio_returns.cov() * 252
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = portfolio_variance ** 0.5
            vol_pct = portfolio_volatility * 100

            nifty_variance = returns['^NSEI'].var()
            betas = [returns[t].cov(returns['^NSEI']) / nifty_variance for t in tickers]
            portfolio_beta = np.dot(weights, betas)

            expected_annual_return = risk_free_rate + (portfolio_beta * equity_risk_premium)
            sharpe_ratio = (expected_annual_return - risk_free_rate) / portfolio_volatility

            z_score = norm.ppf(1 - 0.95)
            daily_volatility = portfolio_volatility / np.sqrt(252)
            VaR_30_days = abs(z_score * daily_volatility * np.sqrt(30) * total_value)

            base_investment = 100000
            cumulative_returns = pd.DataFrame(index=returns.index)
            cumulative_returns['Your Portfolio'] = (1 + daily_portfolio_returns).cumprod() * base_investment
            cumulative_returns['Nifty 50 (NSE)'] = (1 + returns['^NSEI']).cumprod() * base_investment
            cumulative_returns['Sensex (BSE)'] = (1 + returns['^BSESN']).cumprod() * base_investment
            
            portfolio_peaks = cumulative_returns['Your Portfolio'].cummax()
            portfolio_drawdowns = (cumulative_returns['Your Portfolio'] - portfolio_peaks) / portfolio_peaks
            max_drawdown_pct = portfolio_drawdowns.min() * 100

            final_portfolio_val = cumulative_returns['Your Portfolio'].iloc[-1]
            final_nifty_val = cumulative_returns['Nifty 50 (NSE)'].iloc[-1]
            final_sensex_val = cumulative_returns['Sensex (BSE)'].iloc[-1]

            # --- TRAFFIC LIGHT LOGIC ---
            if vol_pct < 12:
                risk_status = "🟢 Low Risk"
            elif vol_pct < 20:
                risk_status = "🟡 Moderate"
            else:
                risk_status = "🔴 High Risk"

            st.markdown("---")
            
            # ==========================================
            # UX: TABS SYSTEM START
            # ==========================================
            tab1, tab2 = st.tabs(["🚦 Quick Health Summary", "🔬 Deep Dive Breakdown"])
            
            # --- TAB 1: THE EVERYDAY RETAIL INVESTOR ---
            with tab1:
                st.subheader("Your Portfolio Health")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Account Value", f"₹{total_value:,.0f}")
                with col2:
                    st.metric("Estimated Next-Year Return", f"{expected_annual_return * 100:.1f}%")
                with col3:
                    st.metric("Bumpy Ride Factor", risk_status)
                with col4:
                    st.metric("Worst Historic Crash", f"{abs(max_drawdown_pct):.1f}%", delta="Peak-to-Bottom Loss", delta_color="inverse")
                    
                with st.expander("💡 What do these scores actually mean? (Plain English Guide)"):
                    st.markdown(f"""
                    * **Estimated Next-Year Return:** Based on your settings, your specific combination of stocks is estimated to grow around **{expected_annual_return * 100:.1f}%** over a normal year.
                    * **Bumpy Ride Factor:** We graded your portfolio's volatility. A **{risk_status}** score means you can expect your account balance to jump around {'a little bit' if vol_pct < 12 else 'a fair amount' if vol_pct < 20 else 'wildly'} from month to month.
                    * **Worst Historic Crash:** If you had the worst luck possible and bought your portfolio right before a major drop, your account would have temporarily declined by **{abs(max_drawdown_pct):.1f}%** before eventually recovering.
                    """)

                st.markdown("---")
                
                # Action-Oriented Concentration Warning
                if max_weight > 0.40:
                    st.error(f"⚠️ **Action Needed:** **{heaviest_asset}** makes up **{max_weight*100:.1f}%** of your entire portfolio. To lower your overall risk grade, consider selling a portion of this stock and buying broad market Index Funds.")
                else:
                    st.success("✅ **Good Separation:** Your investments are nicely balanced. No single stock dominates more than 40% of your total portfolio value.")

                st.info(f"🛡️ **Safety Threshold Check:** Based on normal market cycles, there is a 95% mathematical probability that your portfolio value will not drop more than **₹{VaR_30_days:,.0f}** in a single month.")

                st.subheader(f"🏃 The ₹1 Lakh Growth Race ({timeframe_choice} Lookback)")
                
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

            # --- TAB 2: THE TECHNICAL POWER USER ---
            with tab2:
                st.subheader("Under the Hood: Technical Risk")
                st.markdown("For investors who want to see the exact statistical math driving their portfolio.")
                
                tech_col1, tech_col2, tech_col3 = st.columns(3)
                with tech_col1:
                    st.metric("Exact Annual Volatility", f"{vol_pct:.2f}%")
                with tech_col2:
                    st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
                with tech_col3:
                    sharpe_color = "normal" if sharpe_ratio > 1 else "inverse"
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", delta="Risk Adjusted", delta_color=sharpe_color)
                
                st.markdown("---")
                st.markdown("##### **Individual Stock Volatility Breakdowns**")
                
                fig_bar = px.bar(
                    vol_df, x='Price Swings (Volatility %)', y='Ticker', orientation='h',
                    color='Price Swings (Volatility %)', color_continuous_scale='Reds'
                )
                fig_bar.update_layout(template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_bar, use_container_width=True)
                
                st.subheader("Exact Capital Allocation")
                st.dataframe(
                    portfolio_df[['Ticker', 'Quantity', 'Latest Price (₹)', 'Total Investment Value (₹)', 'Weight (%)']],
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Latest Price (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                        "Total Investment Value (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                        "Weight (%)": st.column_config.NumberColumn(format="%.1f%%")
                    }
                )
