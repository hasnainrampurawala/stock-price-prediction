import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go

# loading data
with open("semiconductor_ai_stocks_5yr.json") as f:
    all_stocks = json.load(f)

# convert json to panda series
def json_to_series(stock_json):
    data = {k: float(v["5. adjusted close"]) for k, v in stock_json.items()}
    series = pd.Series(data)
    series.index = pd.to_datetime(series.index)
    return series.sort_index()

# create windows for training
def make_windows(series, window=12):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series.iloc[i:i+window].values)
        y.append(series.iloc[i+window])
    return np.array(X), np.array(y)

# linear ar model
class ARModel(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.linear = nn.Linear(window, 1)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)

# predicting stock prices
def predict_stock(stock_data, window=12):
    # converting prices to returns
    prices = json_to_series(stock_data)
    returns = np.log(prices / prices.shift(1)).dropna()
    
    # train model
    X, y = make_windows(returns, window)
    model = ARModel(window)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    X_t, y_t = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    for _ in range(1000):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(X_t), y_t)
        loss.backward()
        optimizer.step()
    
    # 12-month rolling forecast
    model.eval()
    current_window = returns.iloc[-window:].values.copy()
    future_returns = []
    
    with torch.no_grad():
        for _ in range(12):
            next_ret = model(torch.tensor(current_window, dtype=torch.float32).unsqueeze(0)).item()
            future_returns.append(next_ret)
            current_window = np.append(current_window[1:], next_ret)
    
    # convert returns to prices
    last_price = prices.iloc[-1]
    future_prices = [last_price := last_price * np.exp(r) for r in future_returns]
    future_dates = pd.date_range(prices.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq="M")
    
    return prices, returns, future_dates, future_prices

# get user choice
ticker = input("Enter stock ticker (NVDA, AMD, INTC): ").upper()
if ticker not in all_stocks:
    print(f"Stock {ticker} not found. Defaulting to NVDA.")
    ticker = "NVDA"

print(f"Processing {ticker}")

# run prediction
prices, returns, forecast_dates, forecast_prices = predict_stock(all_stocks[ticker])

# create dashboard
fig = go.Figure()

# historical prices
fig.add_trace(
    go.Scatter(x=prices.index, y=prices.values, name='Historical')
)

# predicted prices
fig.add_trace(
    go.Scatter(x=forecast_dates, y=forecast_prices, name='Predicted',
               line=dict(dash='dash'))
)

# layout
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price ($)",
    title=f"{ticker} Stock Price Prediction"
)

# print summary
print(f"\n{'-'*80}")
print(f"{ticker} - 12 Month Price Forecast Summary")
print(f"\n")
print(f"Current Price: ${prices.iloc[-1]:.2f}")
print(f"Predicted Price (12 months): ${forecast_prices[-1]:.2f}")
print(f"Expected Return: {((forecast_prices[-1] / prices.iloc[-1]) - 1) * 100:.2f}%")

# show dashboard
fig.show()