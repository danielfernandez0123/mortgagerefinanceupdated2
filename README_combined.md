# Optimal Mortgage Refinancing Calculator - Combined Models

This Streamlit dashboard implements and compares two influential academic models for optimal mortgage refinancing decisions:

## Models Included

### 1. Agarwal, Driscoll & Laibson (2007) - Closed Form Solution
- **Key Innovation**: First closed-form solution using Lambert W-function
- **Main Formula**: x* = (1/ψ)[φ + W(-exp(-φ))]
- **Accounts for**:
  - Option value of waiting
  - Tax deductibility of mortgage interest
  - Probability of moving
  - Interest rate volatility

### 2. Chen & Ling (1989) - Stochastic Interest Rates
- **Key Innovation**: Binomial lattice model with stochastic rates
- **Approach**: Numerical solution via dynamic programming
- **Accounts for**:
  - Mean-reverting interest rates
  - Transaction costs
  - Expected holding period
  - Interest rate volatility

## Installation

```bash
# Clone or download the repository
cd mortgagerefinance

# Install dependencies
pip install -r requirements_combined.txt

# Run the application
streamlit run streamlit_app_combined.py
```

## Features

### 📊 Individual Model Tabs
- **Agarwal Model**: Calculate exact optimal refinancing threshold
- **Chen & Ling Model**: Determine required interest rate differential (IRD)

### 🔄 Model Comparison Tab
- Side-by-side comparison of recommendations
- Visual comparison across different mortgage sizes
- Key differences and assumptions

### 📈 Sensitivity Analysis
- Both models include parameter sensitivity charts
- Understand how changing conditions affect decisions

### 📖 Documentation Tab
- Detailed explanation of both models
- Parameter definitions
- Practical usage guidance

## Input Parameters

### Common Inputs
- Remaining mortgage balance
- Original mortgage rate
- Current market rate

### Agarwal Model Specific
- Real discount rate (ρ)
- Interest rate volatility (σ)
- Marginal tax rate (τ)
- Probability of moving (μ)
- Expected inflation (π)
- Refinancing costs (fixed + points)

### Chen & Ling Model Specific
- Transaction costs (α)
- Short-rate volatility (σ)
- Interest rate drift (μ)
- Expected holding period (EHP)

## Key Insights

1. **Option Value**: Both models show that simple NPV rules significantly underestimate optimal thresholds
2. **Typical Thresholds**: Range from 100-250 basis points depending on parameters
3. **Model Agreement**: Despite different approaches, both models often give similar recommendations
4. **Parameter Sensitivity**:
   - Higher volatility → wait longer
   - Higher transaction costs → wait longer
   - Longer expected holding period → can refinance sooner

## Academic References

1. Agarwal, S., Driscoll, J. C., & Laibson, D. (2007). "Optimal Mortgage Refinancing: A Closed Form Solution." NBER Working Paper No. 13487.

2. Chen, A. H., & Ling, D. C. (1989). "Optimal Mortgage Refinancing with Stochastic Interest Rates." Real Estate Economics, 17(3), 278-299.

## Disclaimer

This calculator is for educational purposes based on academic models. For actual refinancing decisions, consult with qualified financial professionals who can consider your complete financial situation.

## Model Comparison Summary

| Feature | Agarwal et al. (2007) | Chen & Ling (1989) |
|---------|----------------------|-------------------|
| Solution Type | Closed-form (exact) | Numerical |
| Interest Rate Model | Random walk | Binomial with drift |
| Tax Effects | Explicit | Implicit |
| Transaction Costs | Fixed + percentage | Percentage only |
| Computation | Instant | Requires iteration |
| Mean Reversion | No | Yes |

## Running the Dashboard

After installation, the dashboard will be available at `http://localhost:8501` with four main tabs:

1. **Agarwal Model**: Implementation of the closed-form solution
2. **Chen & Ling Model**: Stochastic interest rate model
3. **Model Comparison**: Side-by-side analysis
4. **Documentation**: Detailed explanations and references