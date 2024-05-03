import random
import numpy as np
import StrategyLearner as sl
import datetime as dt
import pandas as pd
import ManualStrategy as ms
import matplotlib.pyplot as plt

def author():
  return 'fadam6'


def main():
    random.seed(903950687)
    sd_in = sd=dt.datetime(2008, 1, 1)
    ed_in = ed=dt.datetime(2009, 12, 31)
    sd_out = sd=dt.datetime(2010, 1, 1)
    ed_out = ed=dt.datetime(2011, 12, 31)

    impacts = [0.001, 0.0025, 0.005, 0.01]
    comission = 0
    trades_count, sharpe_ratios, cum_ret = [], [], []

    for impact in impacts:
        learner = sl.StrategyLearner(impact=impact, commission=comission)
        learner.add_evidence(symbol="JPM", sd=sd_in, ed=ed_in, sv=100000)

        df_in_trades = learner.testPolicy(symbol="JPM", sd=sd_in, ed=ed_in, sv=100000)
        ls_pvs_in = learner.get_portfolio_values()

        # Get metrics
        daily_returns = np.diff(ls_pvs_in) / ls_pvs_in[:-1]
        # Calculate cumulative return
        cum_ret.append((ls_pvs_in[-1] / ls_pvs_in[0]) - 1.0)
        # Calculate Sharpe Ratio (Assuming 252 trading days)
        sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        sharpe_ratios.append(sharpe_ratio)

    # Create a figure and a set of subplots
    plt.figure(figsize=(18, 12))
    fig, ax1 = plt.subplots()
    # Plotting the cumulative return on the left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Impact')
    ax1.set_ylabel('Cumulative Return', color=color)
    ax1.plot(impacts, cum_ret, color=color, marker='o', label='Cumulative Return')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the Sharpe Ratio
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Sharpe Ratio', color=color)
    ax2.plot(impacts, sharpe_ratios, color=color, marker='o', linestyle='--', label='Sharpe Ratio')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and legend
    plt.title('Impact vs Cumulative Return and Sharpe Ratio')
    fig.tight_layout()  # Adjust layout to prevent clipping of ylabel

    # Show the plot
    plt.savefig('Exp2_Impact_Sharpe.png')

if __name__ == '__main__':
    main()