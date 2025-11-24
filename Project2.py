import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Variables
tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "JPM"]
start_date = "2020-01-01"
end_date = "2025-01-01"
risk_free_rate = 0.04  # 4% annual risk-free rate

# Data download
data = yf.download(tickers, start=start_date, end=end_date)["Close"]
returns = data.pct_change().dropna()

# Statistics
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

weight_list = []

# Random portfolios
num_portfolios = 10000
results = np.zeros((3, num_portfolios))  # [return, volatility, sharpe]

for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)  # Weight vector normalization

    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = sharpe_ratio
    weight_list.append(weights)

# Optimal portfolio
max_sharpe_idx = np.argmax(results[2])
max_sharpe_return = results[0, max_sharpe_idx]
max_sharpe_volatility = results[1, max_sharpe_idx]
max_sharpe_weights = weight_list[max_sharpe_idx]

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap="viridis", s=10)
plt.colorbar(label="Sharpe Ratio")
plt.scatter(
    max_sharpe_volatility,
    max_sharpe_return,
    color="red",
    s=80,
    label="Max Sharpe Ratio",
)
plt.title("Efficient Frontier (Random Portfolios)")
plt.xlabel("Volatility (Risk)")
plt.ylabel("Expected Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print results
print("--------------------------------")
print("Optimal Portfolio (Max Sharpe Ratio):")
print(f"Expected Return: {max_sharpe_return:.2%}")
print(f"Volatility: {max_sharpe_volatility:.2%}")
print(f"Sharpe Ratio: {results[2, max_sharpe_idx]:.2f}")

print("--------------------------------")
print("Optimal Portfolio Weights:")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: {max_sharpe_weights[i]:.2%}")
