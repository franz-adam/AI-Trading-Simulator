""""""
"""  		  	   		 	   			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Student Name: Franz Adam (replace with your name)  		  	   		 	   			  		 			     			  	 
GT User ID: fadam6 (replace with your User ID)  		  	   		 	   			  		 			     			  	 
GT ID: 903950687 (replace with your GT ID)  		  	   		 	   			  		 			     			  	 
"""

import datetime as dt
import random
import pandas as pd
from datetime import timedelta
import numpy as np
import ManualStrategy as ms
import QLearner as ql
from util import get_data
import matplotlib.pyplot as plt
import itertools
import marketsimcode as m_sim


def update_holdings(initial_holdings, action):
    if action == 0 or action == initial_holdings:
        return initial_holdings
    else:
        return action


def transform_holdings_into_count(holdings):
    if holdings == 0:
        return 0
    elif holdings == 1:
        return -1000
    else:
        return 1000


def transform_action(action):
    if action == 0:
        return 0
    elif action == 1:
        return -1.0
    else:
        return 1.0


def calculate_port_value(symbol, action, prices, holdings, day, profit, commission, impact, start_val, trade_count):
    share_price = prices.iloc[day][symbol]
    trade = transform_action(action)

    if action == 0 or action == holdings:
        pass  # If no trade action happened or the action matches our holdings, don't trade
    else:
        holdings = action
        trade = trade / 2 if trade_count == 1 else trade  # Account for very first trade
        profit = profit - commission  # Commission
        if action == 1:
            share_price -= share_price * impact
        else:
            share_price += share_price * impact
        profit = profit + share_price * (-2000) * trade

    symbol_count = transform_holdings_into_count(holdings)
    port_val = start_val + profit + symbol_count * share_price

    return port_val, profit, holdings


def calculate_port_value_print(symbol, action, prices, holdings, day, profit, commission, impact, start_val, trade_count):
    share_price = prices.iloc[day][symbol]
    trade = transform_action(action)
    print("Share Price: ", share_price, " Holdings before: ", holdings, " Action: ", action, " Profit before: ", profit)

    if action == 0 or action == holdings:
        print("NO - Trade")
        pass  # If no trade action happened or the action matches our holdings, don't trade
    else:
        print("YES - Trade")
        holdings = action
        print("Updated holdings: ", holdings)
        trade = trade / 2 if trade_count == 1 else trade  # Account for very first trade
        print("Trade: ", trade)
        profit = profit - commission  # Commission
        if action == 1:
            share_price -= share_price * impact
        else:
            share_price += share_price * impact
        profit = profit + share_price * (-2000) * trade
        print("Updated Profit: ", profit)

    symbol_count = transform_holdings_into_count(holdings)
    print("Symbol count: ", symbol_count)
    port_val = start_val + profit + symbol_count * share_price
    print("PortValue: ", port_val)
    return port_val, profit, holdings

class StrategyLearner(object):
    """  		  	   		 	   			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 	   			  		 			     			  	 
    :type verbose: bool  		  	   		 	   			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	   			  		 			     			  	 
    :type impact: float  		  	   		 	   			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	   			  		 			     			  	 
    :type commission: float  		  	   		 	   			  		 			     			  	 
    """

    # constructor  		  	   		 	   			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """  		  	   		 	   			  		 			     			  	 
        Constructor method  		  	   		 	   			  		 			     			  	 
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.q_learner = ql.QLearner()
        self.port_vals = np.zeros(10000)

        # this method should create a QLearner, and train it for trading

    def add_evidence(
            self,
            symbol="JPM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 12, 31),
            sv=100000
    ):
        """  		  	   		 	   			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		 	   			  		 			     			  	 
        :type symbol: str  		  	   		 	   			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
        :type sd: datetime  		  	   		 	   			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
        :type ed: datetime  		  	   		 	   			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
        :type sv: int  		  	   		 	   			  		 			     			  	 
        """
        episodes = 100
        results = []

        # Get price data and indicator values, create Q Learner object
        prices = get_symbol_prices(symbol=symbol, sd=sd, ed=ed, additional_lookback=60)
        bb, mtm, rsi = ms.get_bb_mtm_rsi(prices, bb_lookb=20, mtm_lookb=8, rsi_lookb=4, sd=sd)
        prices = prices.loc[sd:]  # Start at start date
        self.q_learner = ql.QLearner()

        for epoch in range(1, episodes + 1):
            # Set initial state (first trading day, random holdings)
            port_vals, holdings, trade_count, profit, action_counter = np.zeros(len(prices)), 0, 0, 0, 0
            state_0 = discretize(bb.iloc[0][symbol], mtm.iloc[0][symbol], rsi.iloc[0][symbol])
            action = self.q_learner.querysetstate(state_0)
            trade_count += 1 if action != 0 else trade_count

            port_vals[0], profit, holdings = calculate_port_value(symbol=symbol, action=action, prices=prices,
                                                                  holdings=holdings,
                                                                  day=0,
                                                                  profit=profit, commission=self.commission,
                                                                  impact=self.impact,
                                                                  start_val=sv, trade_count=trade_count)

            total_value = 0

            for day in range(1, len(prices)):
                port_vals[day], profit, holdings = calculate_port_value(symbol=symbol, action=action, prices=prices,
                                                                        holdings=holdings, day=day, profit=profit,
                                                                        commission=self.commission, impact=self.impact,
                                                                        start_val=sv, trade_count=trade_count)

                reward = 10 * ((port_vals[day] - port_vals[day - 1]) / port_vals[day - 1])
                #reward += 0.0001 if action != 0 else reward

                next_state = discretize(bb.iloc[day][symbol], mtm.iloc[day][symbol], rsi.iloc[day][symbol])
                action = self.q_learner.query(s_prime=next_state, r=reward)
                trade_count += 1 if action != 0 else trade_count

        return None

    def testPolicy(
            self,
            symbol="IBM",
            sd=dt.datetime(2010, 1, 1),
            ed=dt.datetime(2011, 12, 31),
            sv=100000,
    ):
        """  		  	   		 	   			  		 			     			  	 
        Tests your learner using data outside the training data
  		  	   		 	   			  		 			     			  	 
        :param symbol: The stock symbol that you trained on
        :type symbol: str  		  	   		 	   			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
        :type sd: datetime  		  	   		 	   			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
        :type ed: datetime  		  	   		 	   			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
        :type sv: int  		  	   		 	   			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	   			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	   			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	   			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	   			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 	   			  		 			     			  	 
        """

        prices = get_symbol_prices(symbol=symbol, sd=sd, ed=ed, additional_lookback=60)
        bb, mtm, rsi = ms.get_bb_mtm_rsi(prices, bb_lookb=20, mtm_lookb=8, rsi_lookb=4, sd=sd)
        prices = prices.loc[sd:]  # Start at start date
        assert(len(prices) == len(bb) == len(mtm) == len(rsi))

        df_trades = pd.DataFrame(index=prices.index, columns=['Trades'], data=np.nan)
        df_trades_test = pd.DataFrame(index=prices.index, columns=['Trades'], data=np.nan)

        holdings, profit, trade_count, self.port_vals = 0, 0, 0, np.zeros(len(prices))

        for day in range(len(prices)):
            q_index = discretize(bb.iloc[day][symbol], mtm.iloc[day][symbol], rsi.iloc[day][symbol])
            action = np.argmax(self.q_learner.q_table[q_index])
            trade_count += 1 if action != 0 else trade_count
            self.port_vals[day], profit, holdings = calculate_port_value(symbol=symbol, action=action, prices=prices,
                                                                  holdings=holdings,
                                                                  day=day,
                                                                  profit=profit, commission=self.commission,
                                                                  impact=self.impact,
                                                                  start_val=sv, trade_count=trade_count)
            trade = transform_holdings_into_count(holdings)
            trade_test = transform_holdings_into_count(action)
            df_trades_test.iloc[day]["Trades"] = trade_test
            df_trades.iloc[day]["Trades"] = trade

        df_trades = transform_trades_vectorized(df_trades)
        df_trades_test = transform_trades_vectorized(df_trades_test)
        df_trades['Trades'] *= 2
        df_trades_test['Trades'] *= 2

        df_trades['Trades'] = df_trades['Trades'].astype(int)
        df_trades_test['Trades'] = df_trades_test['Trades'].astype(int)

        return df_trades_test

    def get_portfolio_values(self):
        return self.port_vals

    def author(self):
        return "fadam6"


    def plot_charts(self, sl_pvs_in, sl_pvs_out, symbol="JPM", sv=100000):
        manual_learner = ms.ManualStrategy()
        ms_pvs_in, ms_pvs_out, bench_pvs_in, bench_pvs_out = manual_learner.plot_charts()

        sd_in_sample, ed_in_sample = dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)
        sd_out_sample, ed_out_sample = dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31)
        prices_in = ms.get_symbol_prices(symbol=symbol, sd=sd_in_sample, ed=ed_in_sample, additional_lookback=60)
        prices_out = ms.get_symbol_prices(symbol=symbol, sd=sd_out_sample, ed=ed_out_sample, additional_lookback=60)

        plot_bench_vs_manual_vs_learn(bench_pvs_in, ms_pvs_in, sl_pvs_in, show=False)
        plot_bench_vs_manual_vs_learn(bench_pvs_out, ms_pvs_out, sl_pvs_out, show=False, in_sample=False)

def plot_bench_vs_manual_vs_learn(df_bench, df_manual, df_learning, show=False, in_sample=True):
    plt.rcParams.update({'font.size': 20})

    # Convert to pd dataframe
    df_learning = pd.DataFrame(df_learning, index=df_bench.index, columns=['Value'])

    # Normalize the values for all strategies by dividing by their initial values
    df_manual = df_manual / df_manual.iloc[0]
    df_bench = df_bench / df_bench.iloc[0]
    df_learning = df_learning / 100000  # Normalize the learning strategy the same way

    plt.figure(figsize=(24, 14))
    plt.plot(df_bench.index, df_bench['Value'], label='Benchmark', color='purple')
    plt.plot(df_manual.index, df_manual['Value'], label='Manual', color='red')
    plt.plot(df_learning.index, df_learning['Value'], label='Learning Strategy',
             color='blue')  # Add the learning strategy plot

    # Adjust the title to reflect all strategies
    if in_sample:
        plt.title('Benchmark vs Manual vs Learning (In Sample)')
    else:
        plt.title('Benchmark vs Manual vs Learning (Out of Sample)')

    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()  # Add a legend to distinguish between the plots
    plt.grid(False)

    # Adjust the save file name to reflect the inclusion of the learning strategy
    if in_sample:
        plt.savefig('BenchVsManualVsLearning-InSample.png')
    else:
        plt.savefig('BenchVsManualVsLearning-OutOfSample.png')

    if show:
        plt.show()


def get_symbol_prices(symbol="JPM",
                      sd=dt.datetime(2010, 1, 1),
                      ed=dt.datetime(2011, 12, 31),
                      additional_lookback=60
                      ):
    adjusted_sd = sd - timedelta(days=additional_lookback)
    prices = get_data([symbol], pd.date_range(start=adjusted_sd, end=ed)).drop(columns='SPY')
    return prices


def discretize(bb, mtm, rsi):
    # Convert holdings and binned indicator values to return discrete Q Table index
    bb_bin, mtm_bin, rsi_bin = indicator_binning(bb, mtm, rsi)
    return bb_bin * 12 ** 2 + mtm_bin * 12 ** 1 + rsi_bin


def indicator_binning(bb, mtm, rsi):
    # Bollinger Bands binning
    if bb < 0:
        bb_bin = 0
    elif bb >= 100:
        bb_bin = 11
    else:
        bb_bin = int((bb - 0) / 10) + 1  # Dividing the range 0-100 into 10 equal bins

        # Momentum binning
    if mtm < -30:
        mtm_bin = 0
    elif mtm >= 30:
        mtm_bin = 11
    else:
        mtm_bin = int((mtm + 30) / 6) + 1  # Dividing the range -30 to 30 into 10 equal bins

        # Relative Strength Index binning
    if rsi < 10:
        rsi_bin = 0
    elif rsi >= 90:
        rsi_bin = 11
    else:
        rsi_bin = int((rsi - 10) / 8) + 1  # Dividing the range 10-90 into 10 equal bins

    return bb_bin, mtm_bin, rsi_bin


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


if __name__ == "__main__":
    print("One does not simply think up a strategy")
