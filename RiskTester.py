import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# Download Historical Data
# =========================================
tickers = ['AAPL', 'MSFT', 'TSLA', 'AMZN']

# Pull Close prices (rows=dates, cols=tickers)
data = yf.download(tickers, start='2022-01-01', end='2025-01-01')['Close']

# Compute daily returns
returns = data.pct_change().dropna()

print("Data loaded successfully!\n")
print(returns.head())  # Shows first 5 rows of the returns dataframe

# =========================================
# Compute Expected Return & Volatility
# =========================================
# Annualize stats (252 trading days/year)
mean_returns = returns.mean() * 252
volatility = returns.std() * np.sqrt(252)

summary = pd.DataFrame({
    'Expected Return': mean_returns,
    'Volatility': volatility
})
print("\nSummary statistics:")
print(summary)

# Bar plot of expected return
summary['Expected Return'].plot(kind='bar', title='Expected Annual Returns')
plt.tight_layout()
plt.show()

# Bar plot of expected volatility
summary['Volatility'].plot(kind='bar', title='Expected Annual Volatility')
plt.tight_layout()
plt.show()

# =========================================
# Correlation
# =========================================
corr = returns.corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Asset Correlations")
plt.tight_layout()
plt.show()
