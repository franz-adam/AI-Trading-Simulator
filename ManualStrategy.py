import pandas as pd
import datetime as dt
from datetime import timedelta
import numpy as np
from util import get_data
import matplotlib.pyplot as plt
import concurrent.futures
import itertools
import marketsimcode as m_sim


def transform_trades_vectorized(df):
    # Shift the trades column to compare with the previous day
    prev_trades = df['Trades'].shift(1).fillna(0)  # fill NaN with 0 for the first day
    # Where the trade is the same as the previous day, replace with 0
    df['Trades'] = np.where(df['Trades'] == prev_trades, 0, df['Trades'])

    last_trade = df['Trades'].iloc[0]
    for i in range(1, len(df)):
        current_trade = df['Trades'].iloc[i]
        if current_trade == 0:
            # No change, carry on
            continue
        elif current_trade == last_trade:
            # Same trade as last, set to 0
            df['Trades'].iloc[i] = 0
        else:
            # New trade, update last_trade
            last_trade = current_trade

    first_trade_index = df['Trades'].ne(0).idxmax()
    df.loc[first_trade_index, 'Trades'] *= 0.5

    return df


class ManualStrategy:
    """
    This is a Manual Strategy object.

    verbose (bool) – If “verbose” is True, code prints debugging information. If False, no printing.
    impact (float) – The market impact of each transaction, defaults to 0.0
    commission (float) – The commission amount charged, defaults to 0.0

    """

    def __init__(
            self,
            verbose=False,
            impact=0.005,
            commission=9.95,
    ):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def testPolicy(
            self,
            symbol="JPM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 12, 31),
            sv=100000):
        """
        Returns
            A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        Return type:
            pandas.DataFrame
        """
        prices = get_symbol_prices(symbol=symbol, sd=sd, ed=ed, additional_lookback=60)
        df_trades_1 = self.strategy_1(symbol, prices, sd, ed)
        port_vals_1 = self.compute_port_vals(df_trades_1, prices, symbol=symbol)
        self.plot_charts()

        #ema = calculate_ema(prices, span=12, sd=sd)

        return df_trades_1

    def plot_charts(self, symbol="JPM", sv=100000):
        sd_in_sample, ed_in_sample = dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)
        sd_out_sample, ed_out_sample = dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31)

        prices_in_samp = get_symbol_prices(symbol=symbol, sd=sd_in_sample, ed=ed_in_sample, additional_lookback=60)
        prices_out_samp = get_symbol_prices(symbol=symbol, sd=sd_out_sample, ed=ed_out_sample, additional_lookback=60)

        df_in_sample = self.strategy_1(symbol, prices_in_samp, sd_in_sample, ed_in_sample)
        df_out_sample = self.strategy_1(symbol, prices_out_samp, sd_out_sample, ed_out_sample)

        port_vals_in = self.compute_port_vals(df_in_sample, prices_in_samp)
        port_vals_out = self.compute_port_vals(df_out_sample, prices_out_samp)

        bench_vals_in = m_sim.compute_benchmark_values(df_in_sample, symbol=symbol, start_val=sv,
                                                       commission=self.commission, impact=self.impact)
        bench_vals_out = m_sim.compute_benchmark_values(df_out_sample, symbol=symbol, start_val=sv,
                                                       commission=self.commission, impact=self.impact)

        plot_bench_vs_manual_norm(bench_vals_in, port_vals_in, df_in_sample, show=False)
        plot_bench_vs_manual_norm(bench_vals_out, port_vals_out, df_out_sample, show=False, in_sample=False)

        return port_vals_in, port_vals_out, bench_vals_in, bench_vals_out


    def strategy_1(self, symbol, prices, sd, ed, bb_lookback=30, bb_low=10, bb_up=95, mom_lookback=8, mom_low=-10,
                   mom_up=15, rsi_lookback=4, rsi_up=110, rsi_low=10):
        bb = calculate_bollinger_bands(prices, window=bb_lookback, sd=sd)
        mtm = calculate_momentum(prices, period=mom_lookback, sd=sd)
        rsi = calculate_rsi(prices, period=rsi_lookback, sd=sd)

        df_trades = pd.DataFrame(index=bb.index, columns=['Trades'], data=0)
        # Buy signals: BB % below bb_low, momentum positive, RSI above rsi_buy_low
        buy_signals = (bb < bb_low) & (mtm > mom_low) & (rsi < rsi_up)
        # Sell (short) signals: BB % above bb_up, momentum negative, RSI below rsi_short_low
        sell_signals = (bb > bb_up) & (mtm < mom_up) & (rsi > rsi_low)

        # Assign actions to df_trades: 1 for buy, -1 for short, 0 for out
        df_trades['Trades'] = np.where(buy_signals, 1, np.where(sell_signals, -1, 0))

        # Transform trades dataframe to adhere to portvals function API
        df_trades = transform_trades_vectorized(df_trades.copy())

        return df_trades

    def compute_port_vals(
            self,
            df_trades,
            prices,
            symbol="JPM",
            start_val=100000,
    ):
        symbol_count, profit = 0, 0
        portfolio_values = pd.DataFrame(index=df_trades.index, columns=['Value'], data=np.nan)

        for trade_day in df_trades.index:
            share_price = prices.at[trade_day, symbol]

            # Calculate Commission and Impact
            if df_trades.at[trade_day, 'Trades'] != 0.0:
                profit = profit - self.commission  # Commission
                if df_trades.at[trade_day, 'Trades'] > 0:
                    share_price += prices.at[trade_day, symbol] * self.impact  # Impact
                else:
                    share_price -= prices.at[trade_day, symbol] * self.impact

            symbol_count = symbol_count + df_trades.at[trade_day, 'Trades'] * 2000
            profit = profit + share_price * (-2000) * df_trades.at[trade_day, 'Trades']
            port_val = start_val + profit + symbol_count * share_price
            portfolio_values.loc[trade_day, 'Value'] = port_val

        return portfolio_values

    def add_evidence(
            self,
            symbol="JPM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=100000):
        pass

    def author(self):
        return 'fadam6'


def get_symbol_prices(symbol="JPM",
                      sd=dt.datetime(2010, 1, 1),
                      ed=dt.datetime(2011, 12, 31),
                      additional_lookback=60
                      ):
    adjusted_sd = sd - timedelta(days=additional_lookback)
    prices = get_data([symbol], pd.date_range(start=adjusted_sd, end=ed)).drop(columns='SPY')
    return prices


def get_bb_mtm_rsi(prices, bb_lookb, mtm_lookb, rsi_lookb, sd=dt.datetime(2010, 1, 1)):
    bb = calculate_bollinger_bands(prices, window=bb_lookb, sd=sd)
    mtm = calculate_momentum(prices, period=mtm_lookb, sd=sd)
    rsi = calculate_rsi(prices, period=rsi_lookb, sd=sd)
    return bb, mtm, rsi


def calculate_bollinger_bands(prices, window=20, sd=dt.datetime(2010, 1, 1)):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)

    bb_value = (prices - lower_band) / (upper_band - lower_band) * 100
    bb_value = bb_value.loc[sd:]  # Start at start date

    #print(bb_value.iloc[0]["JPM"])
    return bb_value


def calculate_momentum(prices, period=8, sd=dt.datetime(2010, 1, 1)):
    momentum = (prices / prices.shift(period) - 1) * 100  # Convert to percentage
    momentum = momentum.loc[sd:]
    return momentum


def calculate_ema(prices, span=12, sd=dt.datetime(2010, 1, 1)):
    ema = prices.ewm(span=span, adjust=False).mean()
    ema = ema.loc[sd:]
    return ema


def calculate_rsi(prices, period=14, sd=dt.datetime(2010, 1, 1)):
    delta = prices.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.loc[sd:]
    return rsi


def compute_port_vals(
        df_trades,
        prices,
        symbol="JPM",
        start_val=100000,
):
    symbol_count, profit = 0, 0
    portfolio_values = pd.DataFrame(index=df_trades.index, columns=['Value'], data=np.nan)

    for trade_day in df_trades.index:
        share_price = prices.at[trade_day, symbol]

        # Calcualte Commission and Impact
        if df_trades.at[trade_day, 'Trades'] != 0:
            profit = profit - 9.95  # Commission
            if df_trades.at[trade_day, 'Trades'] > 0:
                share_price += prices.at[trade_day, symbol] * 0.005  # Impact
            else:
                share_price -= prices.at[trade_day, symbol] * 0.005

        symbol_count = symbol_count + df_trades.at[trade_day, 'Trades'] * 2000
        profit = profit + share_price * (-2000) * df_trades.at[trade_day, 'Trades']
        port_val = start_val + profit + symbol_count * share_price
        portfolio_values.loc[trade_day, 'Value'] = port_val

    return portfolio_values


def main():
    bb_lookback_options = [30, 25, 20, 15]
    bb_low_options = [5, 10, 15, 20]
    bb_up_options = [85, 90, 95, 100]
    mom_lookback_options = [4, 8, 12]
    mom_low_options = [-10, -5, 0]
    mom_up_options = [0, 5, 10, 15]
    rsi_lookback_options = [4, 8, 12]
    rsi_buy_up_options = [110, 105, 100, 90, 85]
    rsi_buy_low_options = [10, 20, 30, 40]

    params_product = list(itertools.product(
        bb_lookback_options, bb_low_options, bb_up_options,
        mom_lookback_options, mom_low_options, mom_up_options, rsi_lookback_options,
        rsi_buy_up_options, rsi_buy_low_options
    ))

    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    # Run the optimization
    best_params, best_portfolio_value = optimize_strategy_parallel(
        symbol, sd, ed, get_symbol_prices, compute_port_vals, params_product
    )
    print("Best Parameters:", best_params)
    print("Best Portfolio Value:", best_portfolio_value)


def process_params(params, prices, sd=dt.datetime(2008, 1, 1)):
    bb_lookback, bb_low, bb_up, mom_lookback, mom_low, mom_up, rsi_lookback, rsi_buy_up, rsi_buy_low = params

    # Assume strategy_1 function is defined correctly
    df_trades = strategy_1(prices, sd=sd, bb_lookback=bb_lookback, bb_low=bb_low, bb_up=bb_up,
                           mom_lookback=mom_lookback, mom_low=mom_low, mom_up=mom_up,
                           rsi_lookback=rsi_lookback, rsi_up=rsi_buy_up, rsi_low=rsi_buy_low)

    # Assume compute_port_vals function is defined to calculate the portfolio values
    port_vals = compute_port_vals(df_trades, prices)
    return port_vals["Value"].iloc[-1], params


def optimize_strategy_parallel(symbol, sd, ed, get_symbol_prices, compute_port_vals, params_product):
    prices = get_symbol_prices(symbol=symbol, sd=sd, ed=ed)

    best_portfolio_value = -float('inf')
    best_params = None

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_params, params, prices) for params in params_product]
        for future in concurrent.futures.as_completed(futures):
            portfolio_value, params = future.result()
            if portfolio_value > best_portfolio_value:
                best_portfolio_value = portfolio_value
                best_params = params

    return best_params, best_portfolio_value


def strategy_1(prices, sd, bb_lookback=20, bb_low=30, bb_up=70, mom_lookback=8, mom_low=0,
               mom_up=0, rsi_lookback=14, rsi_up=70, rsi_low=30):
    bb = calculate_bollinger_bands(prices, window=bb_lookback, sd=sd)
    mtm = calculate_momentum(prices, period=mom_lookback, sd=sd)
    rsi = calculate_rsi(prices, period=rsi_lookback, sd=sd)

    df_trades = pd.DataFrame(index=bb.index, columns=['Trades'], data=0)
    # Buy signals: BB % below bb_low, momentum positive, RSI above rsi_buy_low
    buy_signals = (bb < bb_low) & (mtm > mom_low) & (rsi < rsi_up)
    # Sell (short) signals: BB % above bb_up, momentum negative, RSI below rsi_short_low
    sell_signals = (bb > bb_up) & (mtm < mom_up) & (rsi > rsi_low)

    # Assign actions to df_trades: 1 for buy, -1 for short, 0 for out
    df_trades['Trades'] = np.where(buy_signals, 1, np.where(sell_signals, -1, 0))

    # Transform trades dataframe to adhere to portvals function API
    df_trades = transform_trades_vectorized(df_trades.copy())

    return df_trades


def plot_bench_vs_manual_norm(df_bench, df_manual, trades, show=False, in_sample=True):
    plt.rcParams.update({'font.size': 20})
    df_manual = df_manual / df_manual.iloc[0]
    df_bench = df_bench / df_bench.iloc[0]
    calculate_statistics(values_norm=df_manual, values_bench_norm=df_bench, show=False, in_sample=in_sample)

    plt.figure(figsize=(24, 14))
    plt.plot(df_bench.index, df_bench['Value'], label='Benchmark', color='purple')
    plt.plot(df_manual.index, df_manual['Value'], label='Manual', color='red')

    trade_count = (trades != 0).sum().sum()

    for date, value in trades.iterrows():
        if value["Trades"] > 0:  # Change 0 to the column name if different
            plt.axvline(x=date, color='blue', linestyle='-', linewidth=0.5, alpha=0.7)
        elif value["Trades"] < 0:  # Change 0 to the column name if different
            plt.axvline(x=date, color='black', linestyle='-', linewidth=0.5, alpha=0.7)

    if in_sample:
        plt.title('Benchmark vs Manual (In Sample)')
    else:
        plt.title('Benchmark vs Manual (Out of Sample)')

    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend(title=f'Total # of Trades: {trade_count}')
    plt.grid(False)
    if in_sample:
        plt.savefig('BenchVsManual-InSample.png')
    else:
        plt.savefig('BenchVsManual-OutOfSample.png')
    if show:
        plt.show()

def calculate_statistics(values_norm, values_bench_norm, show=False, in_sample=True):
    cum_ret_values = round(((values_norm.iloc[-1, 0] - values_norm.iloc[0, 0]) / values_norm.iloc[0, 0]) * 100,6)
    cum_ret_values_bench = round(((values_bench_norm.iloc[-1, 0] - values_bench_norm.iloc[0, 0]) / values_bench_norm.iloc[
        -1, 0]) * 100,6)
    daily_returns = round(values_norm.pct_change().dropna(),6) * 100
    daily_returns_bench = round(values_bench_norm.pct_change().dropna(),6) * 100
    daily_returns_std = float(round(np.std(daily_returns),6))
    daily_returns_bench_std = float(round(np.std(daily_returns_bench),6))
    daily_returns_mean = float(round(np.mean(daily_returns),6))
    daily_returns_bench_mean = float(round(np.mean(daily_returns_bench),6))

    rows = ["Cumulative Return", "Daily Return Std", "Daily Return Mean"]

    if in_sample:
        columns = ["Benchmark (In-Sample)", "Manual Strategy (In-Sample)"]
    else:
        columns = ["Benchmark (Out-Sample)", "Manual Strategy (Out-Sample)"]

    cell_text = [
        [cum_ret_values_bench, cum_ret_values],
        [daily_returns_bench_std, daily_returns_std],
        [daily_returns_bench_mean, daily_returns_mean]
    ]

    plt.figure(figsize=(18, 8))
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.subplots_adjust(left=0.2)
    if in_sample:
        plt.savefig('Stat_Table_in_sample.png')
    else:
        plt.savefig('Stat_Table_out_sample.png')
    if show:
        plt.show()


if __name__ == "__main__":
    #print("Hello, this is main.")
    ml = ManualStrategy(commission=9.95, impact=0.005)
    vals = ml.testPolicy(symbol="JPM")
    #print(vals["Trades"].value_counts())
    #main()
