import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR  # Changed SVC to SVR for regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Download stock data
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = '2020-01-01'
end_date = '2024-12-31'
data = yf.download(stocks, start=start_date, end=end_date)
if 'Adj Close' in data.columns:
    data = data['Adj Close']
else:
    data = data['Close']
returns = data.pct_change().dropna()

# Feature Engineering
X = returns.iloc[:-1, :]
y = returns.shift(-1).iloc[:-1, :]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Support Vector Regression (SVR) Model
svm_model = SVR(kernel='rbf', C=15, gamma=0.01, epsilon=0.001)  # Changed to SVR for regression
svm_model.fit(X_train_scaled, y_train.mean(axis=1))
svm_preds = svm_model.predict(X_test_scaled)
svm_rmse = np.sqrt(mean_squared_error(y_test.mean(axis=1), svm_preds))
print(f"SVR RMSE: {svm_rmse:.4f}")



# 2. Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=2, random_state=42)
rf_model.fit(X_train_scaled, y_train.mean(axis=1))
rf_preds = rf_model.predict(X_test_scaled)
rf_rmse = np.sqrt(mean_squared_error(y_test.mean(axis=1), rf_preds))
print(f"Random Forest RMSE: {rf_rmse:.4f}")

# 3. Long Short-Term Memory (LSTM) Model
X_lstm = np.expand_dims(X_train_scaled, axis=-1)
X_lstm_test = np.expand_dims(X_test_scaled, axis=-1)

lstm_model = Sequential([
    LSTM(200, activation='tanh', return_sequences=True, input_shape=(X_lstm.shape[1], 1)),
    LSTM(200, activation='tanh', return_sequences=True),
    LSTM(200, activation='tanh'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_lstm, y_train.mean(axis=1), epochs=50, batch_size=4, verbose=1)



lstm_preds = lstm_model.predict(X_lstm_test)
lstm_rmse = np.sqrt(mean_squared_error(y_test.mean(axis=1), lstm_preds))
print(f"LSTM RMSE: {lstm_rmse:.4f}")

# Predict 2024 data

test_start_date = '2024-01-01'
test_end_date = '2024-12-31'
test_data = yf.download(stocks, start=test_start_date, end=test_end_date)
if 'Adj Close' in test_data.columns:
    test_data = test_data['Adj Close']
else:
    test_data = test_data['Close']
test_returns = test_data.pct_change().dropna()
X_2024 = test_returns.iloc[:-1, :]
y_actual_2024 = test_returns.shift(-1).iloc[:-1, :]
X_2024_scaled = scaler.transform(X_2024)

# Predict with SVR
svm_preds_2024 = svm_model.predict(X_2024_scaled)
svm_rmse_2024 = np.sqrt(mean_squared_error(y_actual_2024.mean(axis=1), svm_preds_2024))
print(f"SVR RMSE for 2024: {svm_rmse_2024:.4f}")

# Predict with Random Forest
rf_preds_2024 = rf_model.predict(X_2024_scaled)
rf_rmse_2024 = np.sqrt(mean_squared_error(y_actual_2024.mean(axis=1), rf_preds_2024))
print(f"Random Forest RMSE for 2024: {rf_rmse_2024:.4f}")

# Predict with LSTM
lstm_preds_2024 = lstm_model.predict(np.expand_dims(X_2024_scaled, axis=-1))
lstm_rmse_2024 = np.sqrt(mean_squared_error(y_actual_2024.mean(axis=1), lstm_preds_2024))
print(f"LSTM RMSE for 2024: {lstm_rmse_2024:.4f}")

# Visualization

plt.figure(figsize=(10, 6))
plt.plot(y_actual_2024.mean(axis=1).values, label='Actual 2024 Returns', linestyle='dotted')
plt.plot(svm_preds_2024, label='SVR Prediction 2024')
plt.plot(rf_preds_2024, label='Random Forest Prediction 2024')
plt.plot(lstm_preds_2024, label='LSTM Prediction 2024')
plt.legend()
plt.title('Stock Return Predictions for 2024')
plt.xticks(ticks=np.linspace(0, len(y_actual_2024), num=6), labels=['Jan 2024', 'Mar 2024', 'May 2024', 'Jul 2024', 'Sep 2024', 'Nov 2024'])
plt.ylabel('Return')
plt.show()

# Existing visualization
plt.figure(figsize=(10, 6))
plt.plot(y_test.mean(axis=1).values, label='Actual Returns', linestyle='dotted')
plt.plot(svm_preds, label="SVR Prediction")
plt.plot(rf_preds, label="Random Forest Prediction")
plt.plot(lstm_preds, label="LSTM Prediction")
plt.legend()
plt.title('Stock Return Predictions')
plt.xticks(ticks=np.linspace(0, len(y_test), num=8), labels=['Jan 2020', 'Jul 2020', 'Jan 2021', 'Jul 2021', 'Jan 2022', 'Jul 2022', 'Jan 2023', 'Jul 2023'])
plt.ylabel('Return')
plt.show()

#run3