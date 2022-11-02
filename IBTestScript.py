from IBAlgo import Strategy, Account 
from ib_insync import *



ib = IB()
account = Account(USD_balance=1000, BTC_balance=0)
S = Strategy(account, ib, 1, paper_trading=True)
S.RunStrategy()