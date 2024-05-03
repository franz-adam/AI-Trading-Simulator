import StrategyLearner as sl
import datetime as dt
import pandas as pd
import ManualStrategy as ms
import marketsimcode as mk_sim
import random

def author():
  return 'fadam6'

def main():
    random.seed(903950687)
    sd_in = sd = dt.datetime(2008, 1, 1)
    ed_in = ed = dt.datetime(2009, 12, 31)
    sd_out = sd = dt.datetime(2010, 1, 1)
    ed_out = ed = dt.datetime(2011, 12, 31)

    learner = sl.StrategyLearner(impact=0.005, commission=9.95)
    learner.add_evidence(symbol="ML4T-220", sd=sd_in, ed=ed_in, sv=100000)

    df_in_trades = learner.testPolicy(symbol="ML4T-220", sd=sd_in, ed=ed_in, sv=100000)
    ls_pvs_in = learner.get_portfolio_values()
    df_out_trades = learner.testPolicy(symbol="ML4T-220", sd=sd_out, ed=ed_out, sv=100000)
    ls_pvs_out = learner.get_portfolio_values()

    learner.plot_charts(ls_pvs_in, ls_pvs_out)

if __name__ == '__main__':
    main()
