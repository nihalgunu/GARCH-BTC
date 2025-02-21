import os

print("Fetching BTC price data...")
os.system("python scripts/collect_data.py")

print("Running baseline GARCH model...")
os.system("python scripts/baseline_garch.py")

print("Optimizing GARCH model...")
os.system("python scripts/optimize_garch.py")

print("Evaluating models...")
os.system("python scripts/evaluate_models.py")

print("Backtesting trading strategy...")
os.system("python scripts/backtest_strategy.py")