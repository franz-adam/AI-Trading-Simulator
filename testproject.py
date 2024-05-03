import StrategyLearner as sl
import datetime as dt
import pandas as pd
import ManualStrategy as ms
import experiment1 as ex1
import experiment2 as ex2
import random

def author():
  return 'fadam6'

if __name__ == '__main__':
    random.seed(903950687)
    sd_in = sd=dt.datetime(2008, 1, 1)
    ed_in = ed=dt.datetime(2009, 12, 31)
    sd_out = sd=dt.datetime(2010, 1, 1)
    ed_out = ed=dt.datetime(2011, 12, 31)

    symbol = "JPM"

    # Run Manual Learner
    manual_learn = ms.ManualStrategy()
    ms_trades = manual_learn.testPolicy(symbol=symbol)

    # Run Strategy Learner
    learner = sl.StrategyLearner()
    learner.add_evidence(symbol=symbol, sd=sd_in, ed=ed_in)
    sl_trades = learner.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out)

    # Run experiments 1 and 2
    ex1.main()
    ex2.main()

