import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout


# Download stock data
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Add more as needed
start_date = '2020-01-01'
end_date = '2024-12-31'
data = yf.download(stocks, start=start_date, end=end_date)
if 'Adj Close' in data.columns:
    data = data['Adj Close']
else:
    data = data['Close']
returns = data.pct_change().dropna()

# Feature Engineering
X = returns.iloc[:-1, :]  # Features (all but last day)
y = returns.shift(-1).iloc[:-1, :]  # Target (next day's return)

# Split data into training and testing sets

# Prepare 2024 data for prediction
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
scaler = StandardScaler()
X_2024_scaled = scaler.fit_transform(X_2024)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Support Vector Machine (SVM) Classification
svm_model = SVC(kernel='rbf', C=5, gamma=0.1)
svm_model.fit(X_train_scaled, (y_train.mean(axis=1) > 0).astype(int))  # Classify profitable vs non-profitable
svm_preds = svm_model.predict(X_test_scaled)
svm_preds = (svm_preds > 0.5).astype(int)  # Ensure binary classification output for accuracy_score
  # Ensure binary classification output
svm_preds = (svm_preds > 0.5).astype(int)  # Ensure binary classification output
  # Scale predictions to match return values
svm_accuracy = accuracy_score((y_test.mean(axis=1) > 0).astype(int), svm_preds)
print(f"SVM Accuracy: {svm_accuracy:.2f}")

# 2. Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=1000, max_depth=30, min_samples_split=2, min_samples_leaf=2, random_state=42)
rf_model.fit(X_train_scaled, y_train.mean(axis=1))

rf_preds = rf_model.predict(X_test_scaled)
rf_rmse = np.sqrt(mean_squared_error(y_test.mean(axis=1), rf_preds))
print(f"Random Forest RMSE: {rf_rmse:.4f}")

# 3. Long Short-Term Memory (LSTM) Model
X_lstm = np.expand_dims(X_train_scaled, axis=-1)  # Reshape for LSTM input
X_lstm_test = np.expand_dims(X_test_scaled, axis=-1)

lstm_model = Sequential([
    LSTM(300, activation='tanh', return_sequences=True, input_shape=(X_lstm.shape[1], 1)),
    Dropout(0.2),
    LSTM(300, activation='tanh', return_sequences=True),
    Dropout(0.2),
    LSTM(300, activation='tanh'),
    Dense(1)
])

lstm_model.compile(optimizer='adamax', loss='mse')
lstm_model.fit(X_lstm, y_train.mean(axis=1), epochs=50, batch_size=2, verbose=1)


lstm_preds = lstm_model.predict(X_lstm_test)
lstm_rmse = np.sqrt(mean_squared_error(y_test.mean(axis=1), lstm_preds))
print(f"LSTM RMSE: {lstm_rmse:.4f}")

# Predict 2024 data
svm_preds_2024 = svm_model.predict(X_2024_scaled)
svm_preds_2024 = (svm_preds_2024 > 0.5).astype(int)  # Ensure binary classification output
rf_preds_2024 = rf_model.predict(X_2024_scaled)
lstm_preds_2024 = lstm_model.predict(np.expand_dims(X_2024_scaled, axis=-1))
lstm_rmse_2024 = np.sqrt(mean_squared_error(y_actual_2024.mean(axis=1), lstm_preds_2024))
rf_rmse_2024 = np.sqrt(mean_squared_error(y_actual_2024.mean(axis=1), rf_preds_2024))
print(f"LSTM RMSE for 2024: {lstm_rmse_2024:.4f}")
print(f"Random Forest RMSE for 2024: {rf_rmse_2024:.4f}")

# Visualizing 2024 Predictions
plt.figure(figsize=(10, 6))
plt.plot(y_actual_2024.mean(axis=1).values, label='Actual 2024 Returns', linestyle='dotted')
#plt.plot(range(len(svm_preds_2024)), svm_preds_2024, label="SVM Prediction 2024", linestyle="dashed", alpha=0.6)
plt.plot(rf_preds_2024, label="Random Forest Prediction 2024")
plt.plot(lstm_preds_2024, label="LSTM Prediction 2024")
plt.legend()
plt.title('Stock Return Predictions for 2024')
plt.xticks(ticks=np.linspace(0, len(y_actual_2024), num=6), labels=['Jan 2024', 'Mar 2024', 'May 2024', 'Jul 2024', 'Sep 2024', 'Nov 2024'])
plt.ylabel('Return')
plt.show()



# Visualizing 2024 Predictions Per Stock
# Visualizing 2024 Predictions Per Stock

for stock_idx, stock in enumerate(stocks):
    plt.figure(figsize=(10, 6))
    plt.plot(y_actual_2024.iloc[:, stock_idx].values, label=f'Actual {stock} Returns 2024', linestyle='dotted')
    plt.plot(rf_preds_2024, label=f'Random Forest {stock} Prediction 2024')
    plt.plot(lstm_preds_2024, label=f'LSTM {stock} Prediction 2024')
    plt.legend()
    plt.title(f'{stock} Return Predictions for 2024')
    plt.xticks(ticks=np.linspace(0, len(y_actual_2024), num=6), labels=['Jan 2024', 'Mar 2024', 'May 2024', 'Jul 2024', 'Sep 2024', 'Nov 2024'])
    plt.ylabel('Return')
    plt.show()

# Visualizing Results
plt.figure(figsize=(10, 6))
plt.plot(svm_preds, label="SVM Prediction", linestyle="dashed", alpha=0.6)
plt.plot(rf_preds, label="Random Forest Prediction")
plt.plot(lstm_preds, label="LSTM Prediction")
plt.legend()
plt.title('Stock Return Predictions')
plt.xticks(ticks=np.linspace(0, len(y_test), num=8), labels=['Jan 2020', 'Jul 2020', 'Jan 2021', 'Jul 2021', 'Jan 2022', 'Jul 2022', 'Jan 2023', 'Jul 2023'])
plt.ylabel('Return')
plt.show()
