import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA INGESTION
# ==========================================
print("Loading high-frequency data...")
file_path = "BTCUSD-1m-2026-04\BTCUSD-1m-2026-04.csv"
col_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'num_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
df = pd.read_csv(file_path, header=None, names=col_names)
df['timestamp'] = pd.to_datetime(df['open_time'], unit='us')
df.set_index('timestamp', inplace=True)

# ==========================================
# 2. STATE-SPACE MODELING (KALMAN FILTER)
# ==========================================
print("Applying 2D Kalman Filter State Estimation...")
def run_kalman_filter(prices, Q_variance, R_variance):
    n = len(prices)
    x = np.zeros((2, 1))
    x[0, 0] = prices[0]
    x[1, 0] = 0.0
    
    F = np.array([[1, 1], [0, 1]])
    H = np.array([[1, 0]])
    Q = np.array([[Q_variance, 0], [0, Q_variance * 0.1]])
    R = np.array([[R_variance]])
    P = np.eye(2) * 1000 
    
    filtered_prices = np.zeros(n)
    velocities = np.zeros(n) 
    
    for i in range(n):
        x_pred = np.dot(F, x)
        P_pred = np.dot(F, np.dot(P, F.T)) + Q
        z = np.array([[prices[i]]])
        S = np.dot(H, np.dot(P_pred, H.T)) + R
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(S)))
        x = x_pred + np.dot(K, (z - np.dot(H, x_pred)))
        P = P_pred - np.dot(K, np.dot(H, P_pred))
        
        filtered_prices[i] = x[0, 0]
        velocities[i] = x[1, 0]
        
    return filtered_prices, velocities

df['kalman_price'], df['kalman_velocity'] = run_kalman_filter(df['close'].values, Q_variance=1e-4, R_variance=50.0)

# ==========================================
# 3. STATISTICAL ARBITRAGE & REGIME FILTER
# ==========================================
print("Calculating Statistical Divergence and Market Regimes...")
df['residual'] = df['close'] - df['kalman_price']
df['rolling_mean'] = df['residual'].rolling(window=60).mean()
df['rolling_std'] = df['residual'].rolling(window=60).std()
df['z_score'] = (df['residual'] - df['rolling_mean']) / df['rolling_std']

df['abs_velocity'] = df['kalman_velocity'].abs()
df['vel_mean_24h'] = df['abs_velocity'].rolling(1440).mean()
df['vel_std_24h'] = df['abs_velocity'].rolling(1440).std()
df['trend_threshold'] = df['vel_mean_24h'] + 1.0 * df['vel_std_24h']
df['is_ranging'] = df['abs_velocity'] < df['trend_threshold'] 

# ==========================================
# 4. TRADING LOGIC & EXECUTION
# ==========================================
print("Executing Trading Strategy...")
df['position'] = np.nan
df.loc[(df['z_score'] < -3.0) & df['is_ranging'], 'position'] = 1   
df.loc[(df['z_score'] > 3.0) & df['is_ranging'], 'position'] = -1   
df.loc[df['z_score'].between(-0.5, 0.5), 'position'] = 0            
df['position'] = df['position'].ffill().fillna(0)

# ==========================================
# 5. PERFORMANCE METRICS (PnL & RISK)
# ==========================================
df['market_returns'] = df['close'].pct_change()
df['strategy_returns'] = df['position'].shift(1) * df['market_returns']

fee_rate = 0.0005
df['trade'] = df['position'].diff().fillna(0) != 0
df.loc[df['trade'], 'strategy_returns'] -= fee_rate

initial_capital = 10000
df['strategy_returns'] = df['strategy_returns'].fillna(0)
df['equity_curve'] = initial_capital * (1 + df['strategy_returns']).cumprod()
df['bh_equity_curve'] = initial_capital * (1 + df['market_returns'].fillna(0)).cumprod()

max_drawdown = ((df['equity_curve'] - df['equity_curve'].cummax()) / df['equity_curve'].cummax()).min()
num_trades = df['trade'].sum()

print(f"--- BACKTEST COMPLETE ---")
print(f"Total Trades: {num_trades}")
print(f"Max Drawdown: {max_drawdown*100:.2f}%")

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['equity_curve'], label='Kalman State-Space Filter', color='green', linewidth=2)
plt.plot(df.index, df['bh_equity_curve'], label='Buy & Hold Benchmark', color='gray', alpha=0.5)
plt.title(f'Kalman Regime Filter Backtest | Trades: {num_trades} | Max DD: {max_drawdown*100:.2f}%')
plt.ylabel('Portfolio Value (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('performance_plot.png')
print("Performance plot saved as 'performance_plot.png'")
