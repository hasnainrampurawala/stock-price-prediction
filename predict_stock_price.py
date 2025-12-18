import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go

with open("semiconductor_ai_stocks_5yr.json") as f:
    all_stocks = json.load(f)

# convert json to panda series
def json_to_series(stock_json):
    data = {k: float(v["5. adjusted close"]) for k, v in stock_json.items()}
    series = pd.Series(data)
    series.index = pd.to_datetime(series.index)
    return series.sort_index()

# Sliding window approach: each window predicts the next return
def make_windows(series, window=12):
    X = []
    y = []  
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
    prices = json_to_series(stock_data)
    returns = np.log(prices / prices.shift(1)).dropna()
    X, y = make_windows(returns, window)
    model = ARModel(window)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    for _ in range(1000):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    
    # Generate predictions iteratively, feeding each prediction back in
    model.eval()
    current_window = returns.iloc[-window:].values.copy()
    future_returns = []
    
    with torch.no_grad():
        for _ in range(12):
            next_ret = model(torch.tensor(current_window, dtype=torch.float32).unsqueeze(0)).item()
            future_returns.append(next_ret)
            current_window = np.append(current_window[1:], next_ret)
    
    # convert returns to prices 
    future_prices = []
    current_price = prices.iloc[-1]
    for r in future_returns:
        current_price = current_price * np.exp(r)
        future_prices.append(current_price)
    
    future_dates = pd.date_range(prices.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq="M")
    
    return prices, returns, future_dates, future_prices

# prompting user to select between NVDA, AMD and INTC
ticker = input("Enter stock ticker (NVDA, AMD, INTC): ").upper()
if ticker not in all_stocks:
    print(f"Stock {ticker} not found. Defaulting to NVDA.")
    ticker = "NVDA"

print(f"Processing {ticker}")

prices, returns, forecast_dates, forecast_prices = predict_stock(all_stocks[ticker])

# interactive dashboard
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=prices.index, y=prices.values, name='Historical')
)

fig.add_trace(
    go.Scatter(x=forecast_dates, y=forecast_prices, name='Predicted',
               line=dict(dash='dash'))
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price ($)",
    title=f"{ticker} Stock Price Prediction"
)

fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

print(f"\n{'-'*80}")
print(f"{ticker} - 12 Month Price Forecast Summary")
print(f"\n")
print(f"Current Price: ${prices.iloc[-1]:.2f}")
print(f"Predicted Price (12 months): ${forecast_prices[-1]:.2f}")
print(f"Expected Return: {((forecast_prices[-1] / prices.iloc[-1]) - 1) * 100:.2f}%")

fig.show()
