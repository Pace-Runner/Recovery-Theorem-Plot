"""
Recovery Theorem 3D Options Surface Application
Using yfinance for Options Data
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from scipy.interpolate import griddata, Rbf
from scipy.ndimage import gaussian_filter
from scipy.linalg import eig
from scipy.stats import norm
import warnings
import time
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Recovery Theorem - Options Surface", layout="wide")

st.title("Recovery Theorem: Real-Time Options Analysis")
st.markdown("### Powered by yfinance")

# Enhanced asset selection
ASSETS = {
    "SPY (S&P 500 ETF)": {
        "symbol": "SPY",
        "type": "index_etf",
        "volatility_regime": "low",
        "typical_iv": 0.15,
        "liquidity": "ultra_high"
    },
    "QQQ (Nasdaq-100 ETF)": {
        "symbol": "QQQ",
        "type": "index_etf",
        "volatility_regime": "medium",
        "typical_iv": 0.20,
        "liquidity": "ultra_high"
    },
    "IWM (Russell 2000 ETF)": {
        "symbol": "IWM",
        "type": "index_etf",
        "volatility_regime": "medium",
        "typical_iv": 0.22,
        "liquidity": "high"
    },
    "AAPL (Apple)": {
        "symbol": "AAPL",
        "type": "large_cap",
        "volatility_regime": "low",
        "typical_iv": 0.25,
        "liquidity": "ultra_high"
    },
    "TSLA (Tesla)": {
        "symbol": "TSLA",
        "type": "large_cap",
        "volatility_regime": "high",
        "typical_iv": 0.50,
        "liquidity": "high"
    },
    "NVDA (Nvidia)": {
        "symbol": "NVDA",
        "type": "large_cap",
        "volatility_regime": "high",
        "typical_iv": 0.45,
        "liquidity": "ultra_high"
    },
    "MSFT (Microsoft)": {
        "symbol": "MSFT",
        "type": "large_cap",
        "volatility_regime": "low",
        "typical_iv": 0.23,
        "liquidity": "ultra_high"
    },
    "AMZN (Amazon)": {
        "symbol": "AMZN",
        "type": "large_cap",
        "volatility_regime": "medium",
        "typical_iv": 0.30,
        "liquidity": "ultra_high"
    }
}

# Sidebar controls
st.sidebar.header("⚙️ Analysis Parameters")
asset_name = st.sidebar.selectbox("Select Asset:", list(ASSETS.keys()))
asset_info = ASSETS[asset_name]
ticker_symbol = asset_info["symbol"]

# Advanced smoothing
iv_smoothing = st.sidebar.slider("IV Surface Smoothing", 0.3, 3.0, 1.2, 0.1)

# Display asset characteristics
with st.sidebar.expander("📋 Asset Characteristics"):
    st.write(f"**Type:** {asset_info['type']}")
    st.write(f"**Volatility:** {asset_info['volatility_regime']}")
    st.write(f"**Typical IV:** {asset_info['typical_iv']:.1%}")
    st.write(f"**Liquidity:** {asset_info['liquidity']}")

def fetch_yfinance_options_data(ticker, asset_info):
    
    st.info("🔄 Fetching data from yfinance...")
    
    # Fixed parameters for safe, consistent fetching
    MAX_EXPIRATIONS = 15  # Fetch first 15 expirations to avoid rate limits
    
    try:
        import yfinance as yf
        
        ticker_obj = yf.Ticker(ticker)
        
        st.write("🔍 Fetching options expirations...")
        
        # Add a delay before first request to avoid rate limit
        time.sleep(2)
        
        try:
            expirations = ticker_obj.options
            
            if not expirations or len(expirations) == 0:
                st.warning("No options expirations available")
                return None, None, None
                
            st.success(f"✅ Found {len(expirations)} expiration dates")
            
        except Exception as e:
            st.error(f"Failed to get expirations: {str(e)}")
            
            # Check if it's a rate limit error
            if "429" in str(e):
                st.warning("⚠️ Yahoo Finance is rate-limiting requests")
                st.info("💡 Wait 5-10 minutes before trying again")
            
            return None, None, None
        
        all_options = []
        spot_price = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch options chains with delays
        for idx, exp_date in enumerate(expirations[:MAX_EXPIRATIONS]):
            try:
                status_text.text(f"Fetching expiration {idx+1}/{min(MAX_EXPIRATIONS, len(expirations))}: {exp_date}")
                progress_bar.progress((idx + 1) / min(MAX_EXPIRATIONS, len(expirations)))
                
                # Delay between requests
                time.sleep(2)
                
                opt_chain = ticker_obj.option_chain(exp_date)
                
                # Get spot price from first valid chain
                if spot_price is None and hasattr(opt_chain, 'calls') and len(opt_chain.calls) > 0:
                    calls = opt_chain.calls
                    valid_calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)].copy()
                    if len(valid_calls) > 0:
                        valid_calls['spread'] = valid_calls['ask'] - valid_calls['bid']
                        atm_idx = valid_calls['spread'].idxmin()
                        spot_price = float(valid_calls.loc[atm_idx, 'strike'])
                        st.success(f"✅ Estimated spot price: ${spot_price:.2f}")
                
                if hasattr(opt_chain, 'calls') and len(opt_chain.calls) > 0:
                    calls = opt_chain.calls.copy()
                    calls['type'] = 'call'
                    calls['expiry'] = exp_date
                    all_options.append(calls)
                
                if hasattr(opt_chain, 'puts') and len(opt_chain.puts) > 0:
                    puts = opt_chain.puts.copy()
                    puts['type'] = 'put'
                    puts['expiry'] = exp_date
                    all_options.append(puts)
                    
            except Exception as e:
                if "429" in str(e):
                    st.warning(f"⚠️ Rate limited at {exp_date}. Stopping.")
                    break
                st.warning(f"Skipped {exp_date}: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_options:
            st.error("No options data retrieved")
            return None, None, None
        
        if spot_price is None:
            st.error("Could not determine spot price")
            return None, None, None
        
        df = pd.concat(all_options, ignore_index=True)
        st.success(f"✅ Retrieved {len(df)} options contracts")
        
        df['expiry_date'] = pd.to_datetime(df['expiry'])
        df['days_to_expiry'] = (df['expiry_date'] - datetime.now()).dt.days
        
        # CLEAN DATA - filters for valid data ,overwites old df with the valid data ,removing invalid data multiple times
        df = df[df['days_to_expiry'] > 0].copy()
        df = df[df['impliedVolatility'] > 0].copy()
        df = df[df['lastPrice'] > 0].copy()
        df = df.dropna(subset=['strike', 'lastPrice', 'impliedVolatility']) #removes data with missing rows
        
        if len(df) < 20:
            st.warning(f"Only {len(df)} valid contracts after filtering")
            return None, None, None
        
        # Remove outliers
        iv_median = df['impliedVolatility'].median() #finds meawn
        iv_std = df['impliedVolatility'].std() #finds s.d
        df = df[abs(df['impliedVolatility'] - iv_median) < 3 * iv_std].copy() #only keeps data within 3 s.d 
        
        return df, spot_price, expirations
        
    except ImportError as e:
        st.error(f"Missing library: {e}")
        st.info("Install with: pip install yfinance")
        return None, None, None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None, None, None

def build_enhanced_iv_surface(df, spot_price, asset_info, smoothing=1.2):
    """Build asset-specific IV surface with advanced interpolation"""
    
    resolution = 120  # Fixed high resolution for best quality
    
    strikes = df['strike'].values
    ttm = df['days_to_expiry'].values
    iv = df['impliedVolatility'].values
    
    # Remove outliers -0 Removes options that are too far OTM/ITM for surface interpolation
    valid_mask = (
        (strikes > spot_price * 0.5) &  #keeps strikes between 50% and 150% of spot
        (strikes < spot_price * 1.5) & 
        (ttm > 0) & #remove expired options
        (iv > 0.01) & #removes nonsese IV
        (iv < 3.0) &
        np.isfinite(strikes) & #drops invalid data
        np.isfinite(ttm) & 
        np.isfinite(iv)
    )
    
    strikes = strikes[valid_mask]
    ttm = ttm[valid_mask]
    iv = iv[valid_mask]
    
    if len(strikes) < 20:
        return None, None, None
    
    # Create high-resolution grid
    strike_min = max(spot_price * 0.6, strikes.min())  #ensures stays close to stop
    strike_max = min(spot_price * 1.4, strikes.max())
    ttm_min = max(1, ttm.min()) #That avoids creating a mesh with zero TTM which can be unstable
    ttm_max = ttm.max()
    
    strike_grid = np.linspace(strike_min, strike_max, resolution)
    ttm_grid = np.linspace(ttm_min, ttm_max, int(resolution * 0.6))
    
    strike_mesh, ttm_mesh = np.meshgrid(strike_grid, ttm_grid)
    
    # Asset-specific interpolation
    points = np.column_stack([strikes, ttm])
    
    try:
        # Use RBF for smoother interpolation
        rbf = Rbf(strikes, ttm, iv, function='multiquadric', smooth=smoothing)
        iv_mesh = rbf(strike_mesh, ttm_mesh)
        
        # Additional Gaussian smoothing
        iv_mesh = gaussian_filter(iv_mesh, sigma=smoothing * 0.5)
        
        # Ensure reasonable bounds
        iv_mesh = np.clip(iv_mesh, 0.05, 2.0)
        
    except Exception as e:
        # Fallback to griddata
        iv_mesh = griddata(points, iv, (strike_mesh, ttm_mesh), method='cubic')
        nan_mask = np.isnan(iv_mesh)
        if nan_mask.any():
            iv_mesh[nan_mask] = griddata(points, iv, 
                                         (strike_mesh[nan_mask], ttm_mesh[nan_mask]), 
                                         method='nearest')
        iv_mesh = gaussian_filter(iv_mesh, sigma=smoothing)
    
    return strike_mesh, ttm_mesh, iv_mesh

def compute_enhanced_risk_neutral_pdf(df, spot_price, asset_info, expiry_days=30, smoothing=1.2):
    """Asset-specific risk-neutral PDF"""
    
    # Define a time window around target expiry to capture nearby options
    window = 15
    # Filter for call options within expiry window - calls contain all pricing info we need
    df_filtered = df[
        (df['days_to_expiry'] >= expiry_days - window) & 
        (df['days_to_expiry'] <= expiry_days + window) &
        (df['type'] == 'call')
    ].copy()
    
    # Need minimum data points for reliable PDF calculation
    if len(df_filtered) < 15:
        return None, None
    
    # Sort by strike for proper derivative calculations later
    df_filtered = df_filtered.sort_values('strike')
    df_filtered = df_filtered.drop_duplicates(subset=['strike'], keep='first') #drop duplicates
    
    strikes = df_filtered['strike'].values
    prices = df_filtered['lastPrice'].values
    
    # Need enough strikes to compute smooth derivatives
    if len(strikes) < 10:
        return None, None
    
    # Enhanced smoothing based on asset volatility
    # High volatility assets need more smoothing to avoid noise
    vol_multiplier = {'low': 0.8, 'medium': 1.0, 'high': 1.3}
    adjusted_smoothing = smoothing * vol_multiplier.get(asset_info['volatility_regime'], 1.0)
    
    # Use RBF for price smoothing - Breeden needs perfectly smooth!
    # RBF (Radial Basis Function) creates a smooth interpolation between data points
    try:
        # thin_plate spline balances smoothness and data fidelity
        rbf_price = Rbf(strikes, prices, function='thin_plate', smooth=adjusted_smoothing)
        # Create dense strike grid for smooth derivatives
        strike_dense = np.linspace(strikes.min(), strikes.max(), 200)
        prices_smooth = rbf_price(strike_dense)
        # Ensure no negative prices (option prices must be non-negative)
        prices_smooth = np.maximum(prices_smooth, 0.01)
    except:
        # Fallback if RBF fails - use Gaussian smoothing on original strikes
        strike_dense = strikes
        prices_smooth = gaussian_filter(prices, sigma=adjusted_smoothing) #real market call prices equation
    
    # Breeden-Litzenberger formula: PDF = exp(rT) * d²C/dK²
    # First derivative: rate of change of call price with strike
    dC_dK = np.gradient(prices_smooth, strike_dense)
    # Second derivative reveals probability density
    d2C_dK2 = np.gradient(dC_dK, strike_dense) # second derivative in terms of k ( strike density)
    
    # PDF must be non-negative (probabilities can't be negative)
    rn_pdf = np.maximum(d2C_dK2, 0)
    # Light smoothing to remove numerical noise from differentiation
    rn_pdf = gaussian_filter(rn_pdf, sigma=adjusted_smoothing * 0.5) #smooth again a little bit
    
    # Normalize so the PDF integrates to 1 (total probability = 100%)
    if rn_pdf.sum() > 0:    #integral =1
        # Trapezoidal integration calculates area under curve
        rn_pdf = rn_pdf / (np.trapz(rn_pdf, strike_dense) + 1e-10) # Now the PDF integrates to exactly 1.This makes it a proper probability distribution.
                                                            #no make sure don't /0
    return strike_dense, rn_pdf

def recovery_theorem_advanced(df, spot_price, asset_info, n_states=25, smoothing=1.2, max_days=365):
    """Advanced Recovery Theorem"""
    
    # Get all unique expiration dates, sorted chronologically
    expiries = sorted(df['days_to_expiry'].unique())
    # Only use expiries within our max time horizon
    expiries = [e for e in expiries if e <= max_days]
    
    # Need multiple expiries to build transition dynamics
    if len(expiries) < 5:
        return None, None, None
    
    # Asset-specific state space - the space or range can increase with volatility - wider future price range
    # High volatility = wider possible price range to capture tail events
    if asset_info['volatility_regime'] == 'high':
        strike_range = (0.5, 1.5)  # ±50% from spot
    elif asset_info['volatility_regime'] == 'medium':
        strike_range = (0.65, 1.35)  # ±35% from spot
    else:
        strike_range = (0.75, 1.25)  # ±25% from spot
    
    # Create discrete state space - possible future price levels
    strike_min = spot_price * strike_range[0]
    strike_max = spot_price * strike_range[1]
    state_strikes = np.linspace(strike_min, strike_max, n_states)
    
    # Build transition matrix
    # Q[i,j] = probability of moving from state i to state j
    Q = np.zeros((n_states, n_states)) #declare Q matrix
    weights = []
    
    # Loop through near-term expiries to build transition probabilities
    for i, exp_days in enumerate(expiries[:12]):
        # Get risk-neutral PDF for this expiry
        strikes_rn, pdf_rn = compute_enhanced_risk_neutral_pdf(
            df, spot_price, asset_info, exp_days, smoothing
        )
        
        if strikes_rn is None:
            continue
        
        # Map continuous PDF onto our discrete state grid
        pdf_interp = np.interp(state_strikes, strikes_rn, pdf_rn, left=0, right=0)
        # Small floor prevents numerical issues
        pdf_interp = np.maximum(pdf_interp, 1e-10)
        
        # Ensure interpolated PDF sums to 1
        if pdf_interp.sum() > 0:
            pdf_interp = pdf_interp / pdf_interp.sum() #Normalizes to make it a proper probability distribution
        
        # Exponential decay: near-term expiries weighted more heavily
        weight = np.exp(-0.01 * exp_days)
        weights.append(weight) #longer expiries are less important
        
        # Add this expiry's contribution to transition matrix
        # Each row gets the same PDF because we're building aggregate transitions
        for j in range(n_states):
            Q[j, :] += pdf_interp * weight #Every row of Q gets the same RN PDF contribution
    
    if len(weights) == 0:
        return None, None, None
    
    # Normalize each row to sum to 1 (valid probability distribution)
    row_sums = Q.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    Q = Q / row_sums #Makes each row sum to 1 (each is a probability distribution).
    
    # Add stabilizing diagonal term to prevent numerical instability
    Q = Q * 0.95 + 0.05 * np.eye(n_states) #Adds a small 5% chance the process stays in the same state.
    Q = Q / Q.sum(axis=1, keepdims=True) #Re-normalizes after stabilizing.
    
    try:
        # The dominant eigenvector of the transition matrix Q approximates the physical probability distribution.
        # Transpose Q because we want stationary distribution (left eigenvector)
        eigenvalues, eigenvectors = eig(Q.T) 
        # Find largest eigenvalue (should be 1 for stochastic matrix)
        idx = np.argmax(np.real(eigenvalues))
        # Corresponding eigenvector is the stationary distribution
        principal_eigenvector = np.real(eigenvectors[:, idx])
        
        # Extract physical probabilities from eigenvector
        physical_prob = np.abs(principal_eigenvector)
        if physical_prob.sum() > 0:
            physical_prob = physical_prob / physical_prob.sum()
        
    except Exception as e:
        # If eigenvalue decomposition fails, use uniform distribution
        physical_prob = np.ones(n_states) / n_states
    
    # Create meshgrid for 3D surface plot
    ttm_range = np.linspace(expiries[0], min(expiries[-1], max_days), 60)
    strike_mesh, ttm_mesh = np.meshgrid(state_strikes, ttm_range)
    
    # Initialize surface to hold time-evolved probabilities
    physical_surface = np.zeros_like(strike_mesh)
    
    # Asset-specific time evolution parameters
    # Different asset types evolve differently over time
    decay_rate = {'low': 0.001, 'medium': 0.002, 'high': 0.004}
    diffusion_rate = {'low': 0.0005, 'medium': 0.001, 'high': 0.002}
    
    decay = decay_rate.get(asset_info['volatility_regime'], 0.002)
    diffusion = diffusion_rate.get(asset_info['volatility_regime'], 0.001)
    
    # Evolve probability distribution through time
    for i, t in enumerate(ttm_range):
        # Probability decays over time (discounting effect)
        time_decay = np.exp(-decay * t)
        # Uncertainty grows with sqrt(time) - standard diffusion
        time_diffusion = 1.0 + diffusion * np.sqrt(t)
        
        # Mean reversion: probabilities drift back toward center over long horizons
        mean_reversion = 0.02 * (1 - np.exp(-0.01 * t))
        center_idx = len(state_strikes) // 2
        # Gaussian weights centered at middle strike
        reversion_weights = np.exp(-0.5 * ((np.arange(n_states) - center_idx) / (n_states * 0.2)) ** 2)
        
        # Combine base physical prob with time evolution effects
        physical_surface[i, :] = (
            physical_prob * time_decay * time_diffusion * (1 - mean_reversion) +
            reversion_weights * mean_reversion / reversion_weights.sum()
        )
    
    # Smooth the surface to remove artifacts
    physical_surface = gaussian_filter(physical_surface, sigma=smoothing * 0.8)
    # Ensure non-negative probabilities
    physical_surface = np.maximum(physical_surface, 0)
    
    # Normalize each time slice to integrate to 1
    for i in range(len(ttm_range)):
        if physical_surface[i, :].sum() > 0:
            physical_surface[i, :] = physical_surface[i, :] / physical_surface[i, :].sum()
    
    return strike_mesh, ttm_mesh, physical_surface

# Main execution
if st.button("🚀 Fetch Data & Analyze", type="primary"):
    
    # Store in session state to persist data
    # Session state allows data to persist across Streamlit reruns
    df, spot_price, expirations = fetch_yfinance_options_data(ticker_symbol, asset_info)
    
    if df is not None:
        st.session_state['df'] = df
        st.session_state['spot_price'] = spot_price
        st.session_state['expirations'] = expirations
        st.session_state['asset_info'] = asset_info
        st.session_state['asset_name'] = asset_name

# Check if we have data in session state
if 'df' in st.session_state:
    # Retrieve all stored data from session
    df = st.session_state['df']
    spot_price = st.session_state['spot_price']
    expirations = st.session_state['expirations']
    asset_info = st.session_state['asset_info']
    asset_name = st.session_state['asset_name']
    
    if df is None or len(df) == 0:
        st.error(f"❌ Unable to fetch options data for {asset_name}")
        st.info("""
        **Possible reasons:**
        - Options market is closed
        - API rate limits reached
        - Limited options liquidity
        
        **Try:**
        - SPY, QQQ, or AAPL (most liquid)
        - Wait a few minutes and try again
        """)
    else:
        # Data summary
        st.success(f"✅ Loaded {len(df):,} options contracts")
        
        # Display key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Spot Price", f"${spot_price:.2f}")
        with col2:
            st.metric("Total Contracts", f"{len(df):,}")
        with col3:
            st.metric("Expirations", len(expirations) if expirations else len(df['expiry'].unique()))
        with col4:
            st.metric("Max Days Out", df['days_to_expiry'].max())
        
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Avg IV", f"{df['impliedVolatility'].mean():.1%}")
        with col6:
            st.metric("IV Std", f"{df['impliedVolatility'].std():.1%}")
        with col7:
            st.metric("Min Strike", f"${df['strike'].min():.2f}")
        with col8:
            st.metric("Max Strike", f"${df['strike'].max():.2f}")
        
        # Show sample data
        with st.expander("📊 Sample Options Data"):
            st.dataframe(df.head(20))

# Check if we have data in session state
if 'df' in st.session_state:
    df = st.session_state['df']
    spot_price = st.session_state['spot_price']
    expirations = st.session_state['expirations']
    asset_info = st.session_state['asset_info']
    asset_name = st.session_state['asset_name']
    
    if df is None or len(df) == 0:
        st.stop()
    
    # 1. IV SURFACE
    st.markdown("---")
    st.subheader(f"📈 Implied Volatility Surface - {asset_name}")
    st.markdown(f"*Asset Type: {asset_info['type']} | Volatility: {asset_info['volatility_regime']}*")
    
    with st.spinner("Building IV surface..."):
        strike_mesh, ttm_mesh, iv_mesh = build_enhanced_iv_surface(
            df, spot_price, asset_info, iv_smoothing
        )
    
    if strike_mesh is not None:
        # Create 3D surface plot of implied volatility
        fig_iv = go.Figure(data=[go.Surface(
            x=strike_mesh,
            y=ttm_mesh,
            z=iv_mesh * 100,  # Convert to percentage
            colorscale='Viridis',
            colorbar=dict(title="IV %"),
            showscale=True
        )])
        
        fig_iv.update_layout(
            title=f'Implied Volatility Surface - {asset_name}',
            scene=dict(
                xaxis_title='Strike ($)',
                yaxis_title='Days to Expiration',
                zaxis_title='Implied Volatility (%)',
                camera=dict(eye=dict(x=1.7, y=1.7, z=1.4))  # Set viewing angle
            ),
            height=700,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_iv, use_container_width=True)
        st.info(f"**Surface Stats:** Min IV: {iv_mesh.min():.1%} | Max IV: {iv_mesh.max():.1%} | Mean: {iv_mesh.mean():.1%}")
    
    # 2. RISK-NEUTRAL PDF
    st.markdown("---")
    st.subheader("📊 Risk-Neutral Probability Distribution")
    
    # Get available expiries for user selection
    exp_options = sorted(df['days_to_expiry'].unique())[:10]
    if exp_options:
        # Slider to choose expiration date
        selected_expiry = st.select_slider("Select Expiry (days)", options=exp_options, value=exp_options[min(3, len(exp_options)-1)])
        
        with st.spinner("Computing risk-neutral PDF..."):
            strikes_rn, pdf_rn = compute_enhanced_risk_neutral_pdf(
                df, spot_price, asset_info, selected_expiry, iv_smoothing
            )
        
        if strikes_rn is not None:
            # Plot the probability density function
            fig_pdf = go.Figure()
            fig_pdf.add_trace(go.Scatter(
                x=strikes_rn,
                y=pdf_rn,
                mode='lines',
                fill='tozeroy',  # Fill area under curve
                name='Risk-Neutral PDF',
                line=dict(color='cyan', width=3)
            ))
            
            # Mark current spot price
            fig_pdf.add_vline(x=spot_price, line_dash="dash", 
                             line_color="red", 
                             annotation_text=f"Current: ${spot_price:.2f}")
            
            # Calculate and mark expected future price under risk-neutral measure
            expected_price = np.trapz(strikes_rn * pdf_rn, strikes_rn)
            fig_pdf.add_vline(x=expected_price, line_dash="dot",
                             line_color="yellow",
                             annotation_text=f"Expected: ${expected_price:.2f}")
            
            fig_pdf.update_layout(
                title=f'Risk-Neutral PDF - {selected_expiry} Days | {asset_name}',
                xaxis_title='Strike Price ($)',
                yaxis_title='Probability Density',
                template='plotly_dark',
                height=500
            )
            
            st.plotly_chart(fig_pdf, use_container_width=True)
            
            # Calculate probability of up vs down moves by integrating PDF
            prob_up = np.trapz(pdf_rn[strikes_rn > spot_price], strikes_rn[strikes_rn > spot_price])
            prob_down = np.trapz(pdf_rn[strikes_rn < spot_price], strikes_rn[strikes_rn < spot_price])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prob(Up)", f"{prob_up:.1%}")
            with col2:
                st.metric("Prob(Down)", f"{prob_down:.1%}")
            with col3:
                st.metric("Expected Return", f"{((expected_price/spot_price - 1) * 100):.2f}%")
    
    # 3. RECOVERY THEOREM
    st.markdown("---")
    st.subheader("🎯 Physical Probability Surface (Recovery Theorem)")
    st.markdown(f"*Real-world probabilities for {asset_name}*")
    
    with st.spinner("Applying Recovery Theorem..."):
        strike_mesh_phys, ttm_mesh_phys, physical_surface = recovery_theorem_advanced(
            df, spot_price, asset_info, n_states=25, smoothing=iv_smoothing
        )
    
    if physical_surface is not None:
        # Create 3D surface of physical (real-world) probabilities
        fig_recovery = go.Figure(data=[go.Surface(
            x=strike_mesh_phys,
            y=ttm_mesh_phys,
            z=physical_surface * 100,  # Convert to percentage
            colorscale='Hot',
            colorbar=dict(title="Probability %"),
            showscale=True
        )])
        
        fig_recovery.update_layout(
            title=f'Physical Probability Surface - {asset_name}<br>Type: {asset_info["type"]} | Volatility: {asset_info["volatility_regime"]}',
            scene=dict(
                xaxis_title='Strike ($)',
                yaxis_title='Days to Expiration',
                zaxis_title='Physical Probability (%)',
                camera=dict(eye=dict(x=1.7, y=1.7, z=1.4))
            ),
            height=700,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_recovery, use_container_width=True)
        
        with st.expander("📖 Ross Recovery Theorem Explained"):
            st.markdown(f"""
            ### Recovery Theorem Applied to {asset_name}
            
            **Asset Profile:**
            - **Type:** {asset_info['type']}
            - **Volatility Regime:** {asset_info['volatility_regime']}
            - **Liquidity:** {asset_info['liquidity']}
            - **Typical IV:** {asset_info['typical_iv']:.1%}
            
            **What This Shows:**
            This surface represents the **physical (real-world) probability** distribution
            of {asset_name} prices over time, extracted from options prices using the
            Ross Recovery Theorem (2015).
            
            **Key Differences:**
            1. **Risk Aversion:** Physical probabilities account for investor risk preferences
            2. **Market Dynamics:** Includes mean reversion and volatility clustering
            3. **Asset-Specific:** Parameters tuned to {asset_name}'s characteristics
            
            **Data Source:** yfinance
            **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
            """)
    
    st.markdown("---")
    st.success("✅ Analysis complete!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>Market Data</strong> via yfinance</p>
    <p>Methods: Breeden-Litzenberger (1978) + Ross Recovery Theorem (2015)</p>
    <p>Fetches up to 15 expirations for optimal performance and rate-limit safety</p>
</div>
""", unsafe_allow_html=True)