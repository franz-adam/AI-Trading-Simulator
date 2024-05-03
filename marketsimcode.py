import numpy as np
import pandas as pd
from util import get_data, plot_data

def author():
    return "fadam6"

def compute_portvals(
        df_trades,
        symbol="JPM",
        start_val=100000,
):
    dates = pd.date_range(start=df_trades.index[0], end=df_trades.index[-1])
    symbol_count, profit = 0, 0
    prices = get_data([symbol], dates)  # automatically adds SPY
    prices = prices.drop(columns='SPY')  # only portfolio symbols
    portfolio_values = pd.DataFrame(index=df_trades.index, columns=['Value'], data=np.nan)

    for trade_day in df_trades.index:
        share_price = prices.at[trade_day, symbol]
        symbol_count = symbol_count + df_trades.at[trade_day, 'Trades']
        profit = profit + share_price * (-1) * df_trades.at[trade_day, 'Trades']
        port_val = start_val + profit + symbol_count * share_price
        portfolio_values.loc[trade_day, 'Value'] = port_val

    return portfolio_values

def compute_portvals_norm(
        df_trades,
        symbol="JPM",
        start_val=100000,
):
    values = compute_portvals(df_trades, symbol, start_val)
    normalizer = values.iloc[0,0]

    for i in range(len(values)):
        values.iloc[i] = values.iloc[i] / normalizer

    return values
def compute_benchmark_values(
        df_trades,
        symbol="JPM",
        start_val=100000,
        commission=0.0,
        impact=0.0
):
    dates = pd.date_range(start=df_trades.index[0], end=df_trades.index[-1])
    symbol_count, profit = 0, 0
    prices = get_data([symbol], dates)  # automatically adds SPY
    prices = prices.drop(columns='SPY')  # only portfolio symbols
    portfolio_values_bench = pd.DataFrame(index=df_trades.index, columns=['Value'], data=np.nan)

    for trade_day in df_trades.index:
        if trade_day == df_trades.index[0]:
            share_price = prices.at[trade_day, symbol] + prices.at[trade_day, symbol] * impact
            portfolio_values_bench.loc[trade_day, 'Value'] = start_val
            initial_transaction_cost = share_price * 1000 * (-1) + commission
            symbol_count = 1000
        else:
            share_price = prices.at[trade_day, symbol]
            port_val = start_val + initial_transaction_cost + symbol_count * share_price
            portfolio_values_bench.loc[trade_day, 'Value'] = port_val

    return portfolio_values_bench

def compute_benchmark_values_norm(
        df_trades,
        symbol="JPM",
        start_val=100000
):
    values = compute_benchmark_values(df_trades, symbol, start_val)
    normalizer = values.iloc[0,0]

    for i in range(len(values)):
        values.iloc[i] = values.iloc[i] / normalizer

    return values