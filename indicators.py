import datetime as dt
import pandas as pd
from util import get_data
import matplotlib.pyplot as plt
import numpy as np

def author():
    return "fadam6"

def run(symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31)
        ):
    prices = get_data([symbol], pd.date_range(start=sd, end=ed)).drop(columns='SPY')
    prices_low = get_data([symbol], pd.date_range(start=sd, end=ed), colname="Low").drop(columns='SPY')
    prices_high = get_data([symbol], pd.date_range(start=sd, end=ed), colname="High").drop(columns='SPY')
    # BB and BB%
    rolling_mean, upper_band, lower_band, bb_value = calculate_bollinger_bands(prices, window=20)
    plot_bollinger_bands(prices, rolling_mean, upper_band, lower_band, bb_value)
    plot_bollinger_band_percentage(prices, bb_value)
    # Momentum
    momentum = calculate_momentum(prices, period=8)
    plot_momentum(prices, momentum)
    # EMA
    ema = calculate_ema(prices, span=12)
    plot_ema(prices, ema)
    #RSI
    rsi = calculate_rsi(prices, period=14)
    plot_rsi(prices, rsi)
    # CCI

    cci = calculate_cci(prices_high, prices_low, prices, period=20)
    plot_cci(prices, cci)

def calculate_bollinger_bands(prices, window=20):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    bb_value = (prices - lower_band) / (upper_band - lower_band) * 100
    return rolling_mean, upper_band, lower_band, bb_value

def plot_bollinger_bands(prices, rolling_mean, upper_band, lower_band, bb_value, show=False):
    plt.figure(figsize=(10, 6))
    plt.plot(prices.index, prices, label='Prices')
    plt.plot(rolling_mean.index, rolling_mean, label='Rolling Mean')
    plt.plot(upper_band.index, upper_band, label='Upper Band')
    plt.plot(lower_band.index, lower_band, label='Lower Band')
    plt.title('Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.ylim(10, 50)
    plt.legend()
    plt.savefig('Chart_BB.png')
    if show:
        plt.show()

def plot_bollinger_band_percentage(prices, bb_value, show=False):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color=color)
    ax1.plot(prices.index, prices, label='Prices', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(5, 60)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('BB% Value', color=color)
    ax2.plot(bb_value.index, bb_value, label='BB% Value', color=color, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-20, 120)  # Ensure the BB% scale is fixed from 0 to 100
    plt.axhline(y=80, color='red', linestyle='--', label='Overbought', alpha=0.5)
    plt.axhline(y=20, color='green', linestyle='--', label='Oversold', alpha=0.5)
    plt.title('Price and Bollinger Band Percentage')
    fig.tight_layout()
    plt.savefig('Chart_BB%.png')
    if show:
        plt.show()

def calculate_momentum(prices, period=8):
    momentum = (prices / prices.shift(period) - 1) * 100  # Convert to percentage
    return momentum

def plot_momentum(prices, momentum, show=False):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color=color)
    ax1.plot(prices.index, prices, label='Prices', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Momentum (%)', color=color)  # we already handled the x-label with ax1
    ax2.plot(momentum.index, momentum, label='Momentum', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Price and Momentum')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('Chart_Momentum.png')
    if show:
        plt.show()

def calculate_ema(prices, span=12):
    ema = prices.ewm(span=span, adjust=False).mean()
    return ema

def plot_ema(prices, ema, show=False):
    plt.figure(figsize=(10, 6))
    plt.plot(prices.index, prices, label='Prices')
    plt.plot(ema.index, ema, label='EMA', linestyle='--')
    plt.title('Exponential Moving Average (EMA)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('Chart_EMA.png')
    if show:
        plt.show()

def calculate_rsi(prices, period=14):
    delta = prices.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def plot_rsi(prices, rsi, show=False):
    plt.figure(figsize=(10, 6))
    plt.plot(rsi.index, rsi, label='RSI')
    plt.plot(prices.index, prices, label='Prices')
    plt.axhline(y=70, color='red', linestyle='--', label='Overbought')
    plt.axhline(y=30, color='green', linestyle='--', label='Oversold')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.savefig('Chart_RSI.png')
    if show:
        plt.show()

def calculate_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(window=period).mean()
    md = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
    cci = (tp - ma) / (0.015 * md)
    return cci

def plot_cci(prices, cci, show=False):
    plt.figure(figsize=(10, 6))
    plt.plot(cci.index, cci, label='CCI')
    plt.plot(prices.index, prices, label='Prices')
    plt.axhline(y=100, color='red', linestyle='--', label='Overbought')
    plt.axhline(y=-100, color='green', linestyle='--', label='Oversold')
    plt.title('Commodity Channel Index (CCI)')
    plt.xlabel('Date')
    plt.ylabel('CCI')
    plt.legend()
    plt.savefig('Chart_CCI.png')
    if show:
        plt.show()