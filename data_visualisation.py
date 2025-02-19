import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

# Download stock data
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = '2020-01-01'
end_date = '2023-12-31'

data = yf.download(stocks, start=start_date, end=end_date)
if 'Adj Close' in data.columns:
    data = data['Adj Close']
else:
    data = data['Close']
returns = data.pct_change().dropna()

# 1. Stock Price Trends
plt.figure(figsize=(12,6))
for stock in stocks:
    plt.plot(data.index, data[stock], label=stock)
plt.title('Stock Price Trends (2020-2023)')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Stock Returns')
plt.show()

# 3. Distribution of Stock Returns
plt.figure(figsize=(10,6))
for stock in stocks:
    sns.kdeplot(returns[stock], label=stock)
plt.title('Stock Return Distributions')
plt.xlabel('Daily Return')
plt.ylabel('Density')
plt.legend()
plt.show()

# 4. PCA for Dimensionality Reduction
scaler = StandardScaler()
scaled_returns = scaler.fit_transform(returns.dropna())
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_returns)

plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], alpha=0.7)
plt.title('PCA: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 5. K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(pca_result)
labels = kmeans.labels_

plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=labels, palette='Set1')
plt.title('K-Means Clustering of Stocks')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 6. Feature Importance (Random Forest)
X = returns.iloc[:-1, :]
y = returns.shift(-1).iloc[:-1, :]['AAPL']  # Predict Apple stock
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
importance.plot(kind='bar')
plt.title('Feature Importance for Stock Prediction (Random Forest)')
plt.xlabel('Stock Features')
plt.ylabel('Importance')
plt.show()
