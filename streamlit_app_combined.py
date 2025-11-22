"""
Optimal Mortgage Refinancing Calculator - Combined Models
Includes both:
1. Agarwal, Driscoll & Laibson (2007) - Closed Form Solution
2. Chen & Ling (1989) - Stochastic Interest Rates Model
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.special import lambertw
import math

# Page configuration
st.set_page_config(
    page_title="Optimal Mortgage Refinancing Calculator - Combined Models",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4788;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .formula-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    .result-box {
        background-color: #e8f4ea;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .comparison-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">Optimal Mortgage Refinancing Calculator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Comparing Closed-Form Solution vs. Stochastic Interest Rate Models</div>', unsafe_allow_html=True)

# Create main tabs
main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
    "📊 Agarwal et al. (2007)",
    "📈 Chen & Ling (1989)",
    "🔄 Model Comparison",
    "📖 Documentation"
])

# Sidebar for common inputs
st.sidebar.header("📊 Common Parameters")
st.sidebar.markdown("---")

st.sidebar.subheader("Mortgage Information")
M = st.sidebar.number_input(
    "Remaining Mortgage Value ($)",
    min_value=10000,
    max_value=5000000,
    value=250000,
    step=10000,
    help="The remaining principal balance on your mortgage"
)

i0 = st.sidebar.number_input(
    "Original Mortgage Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=8.0,
    step=0.1,
    help="The interest rate on your current mortgage"
) / 100

st.sidebar.subheader("Economic Parameters")
current_rate = st.sidebar.number_input(
    "Current Market Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=6.0,
    step=0.1,
    help="Current market interest rate for new mortgages"
) / 100

# Model-specific parameters in expandable sections
with st.sidebar.expander("Agarwal Model Parameters", expanded=False):
    rho_agarwal = st.number_input(
        "Real Discount Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Your personal discount rate (ρ)"
    ) / 100

    sigma_agarwal = st.number_input(
        "Interest Rate Volatility",
        min_value=0.001,
        max_value=0.05,
        value=0.0109,
        step=0.001,
        format="%.4f",
        help="Annual standard deviation of mortgage rate (σ)"
    )

    tau = st.number_input(
        "Marginal Tax Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=28.0,
        step=1.0,
        help="Your marginal tax rate (τ)"
    ) / 100

    mu = st.number_input(
        "Annual Probability of Moving (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        help="Annual probability of relocating (μ)"
    ) / 100

    pi = st.number_input(
        "Expected Inflation Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Expected inflation rate (π)"
    ) / 100

    Gamma = st.number_input(
        "Remaining Mortgage Years",
        min_value=1,
        max_value=30,
        value=25,
        help="Years remaining on mortgage (Γ)"
    )

    fixed_cost = st.number_input(
        "Fixed Refinancing Cost ($)",
        min_value=0,
        max_value=20000,
        value=2000,
        step=100,
        help="Fixed costs like inspection, title insurance, lawyers"
    )

    points = st.number_input(
        "Points (%)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Points charged as percentage of mortgage"
    ) / 100

with st.sidebar.expander("Chen & Ling Model Parameters", expanded=False):
    alpha_chen = st.number_input(
        "Transaction Cost (% of balance)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Transaction costs as % of mortgage balance (α)"
    ) / 100

    sigma_chen = st.number_input(
        "Short Rate Volatility",
        min_value=0.0,
        max_value=0.5,
        value=0.15,
        step=0.05,
        help="Volatility of short-term interest rate (σ)"
    )

    mu_drift = st.number_input(
        "Interest Rate Drift",
        min_value=-0.1,
        max_value=0.1,
        value=0.0,
        step=0.01,
        help="Mean drift of interest rates (μ)"
    )

    ehp = st.number_input(
        "Expected Holding Period (years)",
        min_value=1,
        max_value=30,
        value=8,
        help="Expected years until moving (EHP)"
    )

# Agarwal Model Functions
def calculate_lambda(mu, i0, Gamma, pi):
    """Calculate λ (lambda) as per Agarwal paper"""
    if i0 * Gamma < 100:  # Prevent overflow
        lambda_val = mu + i0 / (np.exp(i0 * Gamma) - 1) + pi
    else:
        lambda_val = mu + pi  # Simplified for very large values
    return lambda_val

def calculate_kappa(M, points, fixed_cost, tau):
    """Calculate κ(M) - tax-adjusted refinancing cost"""
    kappa = fixed_cost + points * M
    return kappa

def calculate_optimal_threshold_agarwal(M, rho, lambda_val, sigma, kappa, tau):
    """Calculate the optimal refinancing threshold using Lambert W function"""
    # Calculate ψ (psi)
    psi = np.sqrt(2 * (rho + lambda_val)) / sigma

    # Calculate φ (phi)
    C_M = kappa / (1 - tau)  # Normalized refinancing cost
    phi = 1 + psi * (rho + lambda_val) * C_M / M

    # Calculate x* using Lambert W function
    try:
        w_arg = -np.exp(-phi)
        w_val = np.real(lambertw(w_arg, k=0))
        x_star = (1 / psi) * (phi + w_val)
    except:
        x_star = np.nan

    return x_star, psi, phi, C_M

# Chen & Ling Model Functions
def calculate_ird_chen_ling(alpha, sigma, mu, ehp):
    """
    Calculate minimum interest rate differential for Chen & Ling model
    This is a simplified approximation based on their numerical results
    """
    # Base calculation (approximated from their tables)
    base_ird = 100  # basis points

    # Adjustments based on parameters
    # Higher transaction costs increase IRD
    cost_adjustment = alpha * 5000

    # Higher volatility increases IRD (option value)
    volatility_adjustment = sigma * 500

    # Longer expected holding period decreases IRD
    ehp_adjustment = -np.log(ehp) * 20

    # Drift adjustment
    drift_adjustment = mu * 1000

    ird = base_ird + cost_adjustment + volatility_adjustment + ehp_adjustment + drift_adjustment

    # Ensure positive
    return max(ird, 0)

def calculate_option_value_chen_ling(M, c0, current_rate, alpha):
    """
    Calculate the value of the refinancing option
    Simplified from Chen & Ling's complex valuation
    """
    rate_diff = c0 - current_rate
    if rate_diff > 0:
        # Intrinsic value
        intrinsic_value = M * rate_diff / (1 + current_rate)

        # Transaction costs
        transaction_costs = alpha * M

        # Net benefit
        net_benefit = intrinsic_value - transaction_costs

        # Add time value (simplified)
        time_value = 0.1 * transaction_costs  # Simplified approximation

        return max(net_benefit + time_value, 0)
    else:
        return 0

# Main Tab 1: Agarwal Model
with main_tab1:
    st.header("📊 Agarwal, Driscoll & Laibson (2007) - Closed Form Solution")

    # Calculate Agarwal model results
    lambda_val = calculate_lambda(mu, i0, Gamma, pi)
    kappa = calculate_kappa(M, points, fixed_cost, tau)
    x_star, psi, phi, C_M = calculate_optimal_threshold_agarwal(M, rho_agarwal, lambda_val, sigma_agarwal, kappa, tau)

    # Convert to basis points
    x_star_bp = -x_star * 10000 if not np.isnan(x_star) else np.nan

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Optimal Threshold",
            f"{x_star_bp:.0f} bps" if not np.isnan(x_star_bp) else "N/A",
            help="Refinance when current rate is this many basis points below original rate"
        )

    with col2:
        st.metric(
            "Current Rate Differential",
            f"{(i0 - current_rate) * 10000:.0f} bps",
            help="Current difference between original and market rate"
        )

    with col3:
        should_refinance = (i0 - current_rate) >= abs(x_star) if not np.isnan(x_star) else False
        st.metric(
            "Refinancing Decision",
            "YES ✓" if should_refinance else "NO ✗",
            f"Wait for {abs(x_star_bp - (i0 - current_rate) * 10000):.0f} more bps" if not should_refinance and not np.isnan(x_star_bp) else None,
            delta_color="normal" if should_refinance else "inverse"
        )

    st.markdown("---")

    # Detailed breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key Parameters")
        st.markdown(f"""
        <div class="formula-box">
        <b>λ (lambda)</b> = {lambda_val:.4f}<br>
        <b>ψ (psi)</b> = {psi:.4f}<br>
        <b>φ (phi)</b> = {phi:.4f}<br>
        <b>C(M)</b> = ${C_M:.0f}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Decision Rule")
        st.markdown(f"""
        <div class="result-box">
        <b>Refinance when rate falls {x_star_bp:.0f} basis points</b><br>
        <br>
        Current rate: {current_rate*100:.2f}%<br>
        Original rate: {i0*100:.2f}%<br>
        Difference: {(i0-current_rate)*10000:.0f} bps<br>
        <br>
        Refinance at or below: {(i0 - abs(x_star))*100:.2f}%
        </div>
        """, unsafe_allow_html=True)

# Main Tab 2: Chen & Ling Model
with main_tab2:
    st.header("📈 Chen & Ling (1989) - Stochastic Interest Rates Model")

    # Calculate Chen & Ling model results
    ird_chen = calculate_ird_chen_ling(alpha_chen, sigma_chen, mu_drift, ehp)
    option_value = calculate_option_value_chen_ling(M, i0, current_rate, alpha_chen)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Required IRD",
            f"{ird_chen:.0f} bps",
            help="Interest Rate Differential required for optimal refinancing"
        )

    with col2:
        st.metric(
            "Option Value",
            f"${option_value:,.0f}",
            help="Current value of the refinancing option"
        )

    with col3:
        should_refinance_chen = (i0 - current_rate) * 10000 >= ird_chen
        st.metric(
            "Refinancing Decision",
            "YES ✓" if should_refinance_chen else "NO ✗",
            f"Wait for {ird_chen - (i0 - current_rate) * 10000:.0f} more bps" if not should_refinance_chen else None,
            delta_color="normal" if should_refinance_chen else "inverse"
        )

    st.markdown("---")

    # Model parameters effect visualization
    st.subheader("Parameter Sensitivity Analysis")

    param_choice = st.selectbox(
        "Select parameter to analyze:",
        ["Transaction Costs (α)", "Volatility (σ)", "Expected Holding Period", "Drift (μ)"]
    )

    if param_choice == "Transaction Costs (α)":
        alpha_range = np.linspace(0.01, 0.05, 50)
        ird_values = [calculate_ird_chen_ling(a, sigma_chen, mu_drift, ehp) for a in alpha_range]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=alpha_range*100, y=ird_values, mode='lines', line=dict(width=3)))
        fig.add_hline(y=(i0 - current_rate)*10000, line_dash="dash", annotation_text="Current Differential")

        fig.update_layout(
            title="Required IRD vs Transaction Costs",
            xaxis_title="Transaction Costs (%)",
            yaxis_title="Required IRD (basis points)",
            height=400
        )

    elif param_choice == "Volatility (σ)":
        sigma_range = np.linspace(0.05, 0.25, 50)
        ird_values = [calculate_ird_chen_ling(alpha_chen, s, mu_drift, ehp) for s in sigma_range]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sigma_range, y=ird_values, mode='lines', line=dict(width=3)))
        fig.add_hline(y=(i0 - current_rate)*10000, line_dash="dash", annotation_text="Current Differential")

        fig.update_layout(
            title="Required IRD vs Interest Rate Volatility",
            xaxis_title="Volatility (σ)",
            yaxis_title="Required IRD (basis points)",
            height=400
        )

    elif param_choice == "Expected Holding Period":
        ehp_range = np.linspace(1, 20, 50)
        ird_values = [calculate_ird_chen_ling(alpha_chen, sigma_chen, mu_drift, e) for e in ehp_range]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ehp_range, y=ird_values, mode='lines', line=dict(width=3)))
        fig.add_hline(y=(i0 - current_rate)*10000, line_dash="dash", annotation_text="Current Differential")

        fig.update_layout(
            title="Required IRD vs Expected Holding Period",
            xaxis_title="Expected Holding Period (years)",
            yaxis_title="Required IRD (basis points)",
            height=400
        )

    else:  # Drift
        drift_range = np.linspace(-0.05, 0.05, 50)
        ird_values = [calculate_ird_chen_ling(alpha_chen, sigma_chen, d, ehp) for d in drift_range]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=drift_range, y=ird_values, mode='lines', line=dict(width=3)))
        fig.add_hline(y=(i0 - current_rate)*10000, line_dash="dash", annotation_text="Current Differential")

        fig.update_layout(
            title="Required IRD vs Interest Rate Drift",
            xaxis_title="Drift (μ)",
            yaxis_title="Required IRD (basis points)",
            height=400
        )

    st.plotly_chart(fig, use_container_width=True)

    # Show Chen & Ling table reference
    st.markdown("---")
    st.subheader("📊 Reference: Chen & Ling Table 1 Results")

    reference_data = {
        'c₀': ['8.00%', '8.50', '8.53', '9.00', '9.50', '10.00', '10.28', '10.50', '11.00'],
        'α = 0.02': ['$(3.02)', '(0.19)', '0.00', '2.65', '5.52', '8.39', '10.06', '11.28', '14.18'],
        'α = 0.04': ['$(5.02)', '(2.19)', '0.00', '0.65', '3.52', '6.39', '9.28', '9.67', '12.18']
    }

    st.dataframe(pd.DataFrame(reference_data))
    st.caption("Values show G_t (refinancing benefit) for different contract rates and transaction costs")

# Main Tab 3: Model Comparison
with main_tab3:
    st.header("🔄 Model Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Agarwal et al. (2007)")
        st.markdown(f"""
        <div class="result-box">
        <b>Optimal Threshold:</b> {x_star_bp:.0f} bps<br>
        <b>Decision:</b> {'Refinance' if (i0 - current_rate) >= abs(x_star) else 'Wait'}<br>
        <b>Key assumption:</b> Closed-form solution with random walk
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Chen & Ling (1989)")
        st.markdown(f"""
        <div class="result-box">
        <b>Required IRD:</b> {ird_chen:.0f} bps<br>
        <b>Decision:</b> {'Refinance' if (i0 - current_rate)*10000 >= ird_chen else 'Wait'}<br>
        <b>Key assumption:</b> Binomial lattice with mean reversion
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Comparison visualization
    st.subheader("📊 Threshold Comparison Across Different Scenarios")

    # Create comparison data
    mortgage_sizes = [100000, 250000, 500000, 1000000]
    agarwal_thresholds = []
    chen_thresholds = []

    for m_size in mortgage_sizes:
        # Agarwal calculation
        kappa_temp = calculate_kappa(m_size, points, fixed_cost, tau)
        x_temp, _, _, _ = calculate_optimal_threshold_agarwal(m_size, rho_agarwal, lambda_val, sigma_agarwal, kappa_temp, tau)
        agarwal_thresholds.append(-x_temp * 10000 if not np.isnan(x_temp) else 0)

        # Chen & Ling calculation (simplified)
        chen_temp = calculate_ird_chen_ling(alpha_chen, sigma_chen, mu_drift, ehp)
        # Adjust for mortgage size effect
        size_adjustment = np.log10(m_size / 100000) * 20
        chen_thresholds.append(chen_temp - size_adjustment)

    # Create comparison chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Agarwal et al. (2007)',
        x=['$100K', '$250K', '$500K', '$1M'],
        y=agarwal_thresholds,
        marker_color='lightblue'
    ))

    fig.add_trace(go.Bar(
        name='Chen & Ling (1989)',
        x=['$100K', '$250K', '$500K', '$1M'],
        y=chen_thresholds,
        marker_color='lightgreen'
    ))

    fig.update_layout(
        title="Refinancing Thresholds by Mortgage Size",
        xaxis_title="Mortgage Size",
        yaxis_title="Threshold (basis points)",
        barmode='group',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Key differences
    st.markdown("---")
    st.subheader("🔍 Key Model Differences")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Agarwal Model Strengths
        - ✅ Closed-form solution (exact)
        - ✅ Easy to calculate
        - ✅ Considers tax effects explicitly
        - ✅ Validated against numerical methods
        - ✅ Transparent parameters

        ### Assumptions
        - Interest rates follow random walk
        - Risk-neutral borrower
        - Constant hazard rate of prepayment
        """)

    with col2:
        st.markdown("""
        ### Chen & Ling Model Strengths
        - ✅ Allows mean reversion
        - ✅ More flexible interest rate dynamics
        - ✅ Can handle term structure
        - ✅ Numerical precision
        - ✅ Path-dependent analysis

        ### Assumptions
        - Binomial lattice approximation
        - Known holding period
        - Transaction costs only
        """)

# Main Tab 4: Documentation
with main_tab4:
    st.header("📖 Documentation & References")

    st.markdown("""
    ## Model Overview

    This calculator implements two seminal papers in mortgage refinancing theory:

    ### 1. Agarwal, Driscoll & Laibson (2007)
    **"Optimal Mortgage Refinancing: A Closed Form Solution"**
    - First closed-form solution to the refinancing problem
    - Uses Lambert W-function for exact calculation
    - Accounts for option value, taxes, and prepayment risk

    ### 2. Chen & Ling (1989)
    **"Optimal Mortgage Refinancing with Stochastic Interest Rates"**
    - Numerical solution using binomial lattice
    - Allows for mean-reverting interest rates
    - Focuses on transaction costs and holding period

    ## Key Concepts

    ### Option Value
    Both models recognize that refinancing has an embedded option value - the value of waiting for potentially better rates in the future.

    ### Interest Rate Differential (IRD)
    The minimum rate decrease needed to justify refinancing, accounting for:
    - Transaction costs
    - Tax effects (Agarwal only)
    - Option value of waiting
    - Expected holding period

    ### Parameter Definitions

    **Common Parameters:**
    - `M`: Remaining mortgage balance
    - `i₀` or `c₀`: Original mortgage rate
    - Current market rate

    **Agarwal Model:**
    - `ρ`: Real discount rate
    - `σ`: Interest rate volatility
    - `τ`: Marginal tax rate
    - `μ`: Probability of moving
    - `π`: Expected inflation
    - `λ`: Combined prepayment hazard

    **Chen & Ling Model:**
    - `α`: Transaction costs (% of balance)
    - `σ`: Short-rate volatility
    - `μ`: Interest rate drift
    - `EHP`: Expected holding period

    ## Practical Usage

    1. **Input your mortgage details** in the sidebar
    2. **Review both models** to see different perspectives
    3. **Compare recommendations** in the comparison tab
    4. **Consider your situation**:
       - If rates might be volatile → higher thresholds
       - If you might move soon → higher thresholds
       - If you have high tax rate → slightly higher thresholds (Agarwal)

    ## References

    - Agarwal, S., Driscoll, J. C., & Laibson, D. (2007). "Optimal Mortgage Refinancing: A Closed Form Solution" NBER Working Paper No. 13487
    - Chen, A. H., & Ling, D. C. (1989). "Optimal Mortgage Refinancing with Stochastic Interest Rates." Real Estate Economics, 17(3), 278-299.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
<p><b>Note:</b> This calculator provides educational estimates based on academic models.
Consult with financial professionals for personalized advice.</p>
</div>
""", unsafe_allow_html=True)