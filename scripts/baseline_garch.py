import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# Load data
btc_data = pd.read_csv("data/btc_prices.csv", index_col=0, parse_dates=True)

# Fit a GARCH(1,1) model
model = arch_model(btc_data['Returns'] * 100, vol='Garch', p=1, q=1)
result = model.fit(disp="off")

# Print summary
print(result.summary())

# Plot volatility
btc_data['Volatility'] = result.conditional_volatility
plt.figure(figsize=(10,5))
plt.plot(btc_data.index, btc_data['Volatility'], label="GARCH(1,1) Volatility", color="red")
plt.title("Bitcoin Volatility Over Time")
plt.legend()
plt.savefig("results/volatility_plot.png")
plt.show()