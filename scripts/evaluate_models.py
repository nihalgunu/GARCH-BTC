import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_squared_error

# Load data
btc_data = pd.read_csv("data/btc_prices.csv", index_col=0, parse_dates=True)

# Fit multiple models
models = {
    "GARCH(1,1)": arch_model(btc_data['Returns'] * 100, vol='Garch', p=1, q=1),
    "EGARCH": arch_model(btc_data['Returns'] * 100, vol='EGarch', p=1, q=1),
    "GJR-GARCH": arch_model(btc_data['Returns'] * 100, vol='GJR-Garch', p=1, q=1)
}

results = {}
for name, model in models.items():
    res = model.fit(disp="off")
    mse = mean_squared_error(btc_data['Returns'][1:] * 100, res.conditional_volatility[1:])
    results[name] = mse
    print(f"{name} - MSE: {mse}")

# Save results
with open("results/model_performance.txt", "w") as f:
    for name, mse in results.items():
        f.write(f"{name}: MSE = {mse}\n")

# Plot comparison
plt.figure(figsize=(10,5))
for name, model in models.items():
    btc_data[name] = model.fit(disp="off").conditional_volatility
    plt.plot(btc_data.index, btc_data[name], label=name)

plt.title("Comparison of GARCH Models for BTC Volatility")
plt.legend()
plt.savefig("results/model_comparison.png")
plt.show()