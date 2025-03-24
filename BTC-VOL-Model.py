import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from scipy.optimize import minimize
from scipy.stats import norm

# Load the Data
df = pd.read_csv("btc_ohlcv_data.csv", parse_dates=['timestamp'], index_col='timestamp')

# Compute log returns
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df = df.dropna()

# GJR-GARCH (1,1) Model
@jit(nopython=True)
def gjr_garch_filter(omega, alpha, gamma, beta, eps):
    T = len(eps)
    sigma2 = np.zeros(T)
    s = np.zeros(T)
    
    sigma2[0] = omega / (1 - alpha - 0.5 * gamma - beta)
    s[eps < 0] = 1
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t-1]**2 + gamma * s[t-1] * eps[t-1]**2 + beta * sigma2[t-1]
    
    return sigma2

@jit(nopython=True)
def gjr_garch_loglike(params, returns):
    mu, omega, alpha, gamma, beta = params
    eps = returns - mu
    sigma2 = gjr_garch_filter(omega, alpha, gamma, beta, eps)
    loglike = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps**2 / sigma2)
    return -loglike

# Optimize GJR-GARCH parameters
def estimate_gjr_garch(returns):
    initial_params = [0.0, 1e-5, 0.1, 0.1, 0.8]
    bounds = [(None, None), (1e-8, None), (0, 1), (0, 1), (0, 1)]
    
    result = minimize(gjr_garch_loglike, initial_params, args=(returns,),
                      method='L-BFGS-B', bounds=bounds)
    
    return result.x

# Compute BTC CDF based on predicted volatility
def btc_cdf(price_levels, last_price, forecast_vol):
    standardized_prices = (price_levels - last_price) / (forecast_vol * last_price)
    return norm.cdf(standardized_prices)

# Function to run the model with a specified window size
def run_gjr_garch(window_size):
    df_filtered = df.iloc[-window_size:]
    returns = df_filtered['log_return'].values

    if len(returns) < 10:
        print("Not enough data for GARCH estimation. Try a larger window size.")
        return

    # Estimate GJR-GARCH parameters
    mu, omega, alpha, gamma, beta = estimate_gjr_garch(returns)

    # Volatility Calculations
    eps = returns - mu
    sigma2 = gjr_garch_filter(omega, alpha, gamma, beta, eps)
    volatility = np.sqrt(sigma2)

    # Computing the CDF
    last_price = df_filtered['close'].iloc[-1]
    forecast_vol = volatility[-1]
    price_range = np.linspace(df_filtered['close'].min(), df_filtered['close'].max(), 100)
    cdf_values = btc_cdf(price_range, last_price, forecast_vol)

    # Deriving the PDF from the CDF
    pdf_values = np.diff(cdf_values, prepend=0)

    # EV computations
    expected_value = np.sum(pdf_values * price_range)
    print(f"Expected BTC Price (E[X]) for the next period: {expected_value:.2f}")

    # Plotting the CDF
    plt.figure(figsize=(10, 5))
    plt.plot(price_range, cdf_values, label=f'CDF of BTC Price (Last {window_size} min)')
    plt.xlabel("BTC Price")
    plt.ylabel("Probability")
    plt.title(f"Cumulative Distribution Function for BTC Prices\n(Last {window_size} minutes)")
    plt.legend()
    plt.grid()
    plt.show()

# You can change the window size (in minutes, 1440 = 1 day)
run_gjr_garch(window_size=2880)
