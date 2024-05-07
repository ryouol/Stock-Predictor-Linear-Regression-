import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf  # We use yfinance to fetch stock data

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Date'] = data.index
    return data

def prepare_data(data):
    data['Day'] = data['Date'].dt.dayofyear
    features = ['Day']
    target = 'Close'
    return data[features], data[target]

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def plot_predictions(X, y, model):
    plt.figure(figsize=(10, 5))
    plt.scatter(X['Day'], y, color='black', label='Actual Price')
    plt.plot(X['Day'], model.predict(X), color='blue', linewidth=3, label='Predicted Price')
    plt.xlabel('Day of Year')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

# Main program
ticker = 'AAPL'
start_date = '2021-01-01'
end_date = '2024-01-01'

data = fetch_stock_data(ticker, start_date, end_date)
X, y = prepare_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = train_model(X_train, y_train)
print(f'Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%')
plot_predictions(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), model)
