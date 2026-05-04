# Kalman State-Space Regime Filter for High-Frequency Crypto Data

A quantitative backtesting framework that applies control theory (2D discrete-time Kalman Filtering) to 1-minute cryptocurrency market data. 

This project explores an alternative to lagging technical indicators by modeling the market as a dynamic system. It isolates true price action from microstructure noise and dynamically detects trending vs. ranging market regimes to optimize mean-reversion strategies.

## 🧠 Core Architecture

The framework relies on a two-variable state vector to estimate market conditions:
1.  **Position ($x_1$):** The estimated true underlying asset price, filtered from bid-ask bounce and transient liquidity gaps.
2.  **Velocity ($x_2$):** The rate of price change (momentum).

By estimating the hidden "Velocity" state ($x_2$), the model introduces a dynamic regime filter to prevent the execution of mean-reversion trades during strong directional trends.

### Execution Logic
1.  **Noise Filtering:** The Kalman Filter continuously updates its price estimation based on incoming 1m OHLCV data.
2.  **Statistical Divergence (Z-Score):** A rolling 60-minute Z-score calculates the residual between the raw exchange price and the Kalman-estimated price. Extreme deviations ($\pm 3.0 \sigma$) act as base signals.
3.  **Regime Detection:** The absolute value of the Kalman Velocity ($x_2$) is tracked over a rolling 24-hour window. If the current velocity exceeds $Mean + 1\sigma$, the market is classified as "Trending", and all mean-reversion signals are ignored.

## 📊 Backtest Results & Optimization

Tested on 1-minute BTC/USD data (April 2026) with simulated institutional transaction fees (0.05% per trade).

*   **Baseline Mean Reversion:** Operating without the regime filter resulted in severe over-trading (1,423 trades) during a bullish macro trend, leading to a **-53.10% Maximum Drawdown**.
*   **Regime Filter Activated:** Introducing the $x_2$ velocity state as a trend-blocker drastically improved risk metrics. 
    *   **Trade volume reduced by 60%** (576 trades), minimizing transaction fee decay.
    *   **Maximum Drawdown reduced to -28.31%**.

*Conclusion:* The application of control theory effectively mitigated catastrophic drawdowns typically associated with mean-reversion strategies in trending environments.

## 💻 Tech Stack
*   **Python 3.10+**
*   **NumPy:** Matrix operations and state-space mathematics.
*   **Pandas:** Vectorized time-series analysis.
*   **Matplotlib:** Performance and signal visualization.

## 🚀 How to Run
```bash
git clone https://github.com/Zsoli28/kalman-regime-filter.git
cd kalman-regime-filter
pip install pandas numpy matplotlib
python backtest.py
