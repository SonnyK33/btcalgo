import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, clear_output
from math import sqrt
import numpy as np
from ib_insync import *
from collections import deque
from enum import Enum
import datetime
import logging
import math
import time
import pytz


from collections import deque
import pandas as pd


Sentiment = Enum('Sentiment','BEAR NEUTRAL BULL')
Signal = Enum('Signal','BUY SELL')

today = datetime.datetime.today().date()
logging.basicConfig(level = logging.INFO,
                   filename = 'algo_log_'+str(today)+'_.log',
                   filemode='w')

strategy_dict = dict()

def AddStrategy(strategy_fn):
    strategy_dict[strategy_fn.__name__] = strategy_fn
    return strategy_fn
  
def LogFunctionCall(fn):
    logging.info(fn.__name__+' called')
    return fn

def PrintFunction(fn):
    print(fn)
#     print(fn.__code__.co_varnames)
    return fn
    

class Strategy():
    def __init__(self, ib, client_id: int, contract: Contract,
                 paper_trading: bool = True):
        self.client_id = client_id
        self.ib = ib     
        self.contract = contract
        self.paper_trading=paper_trading
        self.sentiment = dict()
        self.signal = dict()
        self.trades = list()
        self.signal_dict = dict()
        self.ConnectIB()     
        self.QualifyContracts()
          

    def ConnectIB(self):
#         self.ib.connect('127.0.0.1', 7497, self.client_id)
        self.ib.connect()
        time.sleep(1)
        
    def QualifyContracts(self):        
        self.ib.qualifyContracts(contract)
        
    def SubscribeData(self):
        print('subscribing to data')
#         for contract in self.contract_lst:
        self.ib.reqMktData(contract)
            
    def GetBars(self, duration, barSize, endDate=""):
        print('getting bars')
        print(self.contract)
        bars = self.ib.reqHistoricalData(
            contract = self.contract,
            endDateTime=endDate,
            durationStr=duration,
            barSizeSetting=barSize,
            whatToShow='MIDPOINT',
            useRTH=False,
            formatDate=2,
            keepUpToDate=True)
        
        return bars       

        
    
    def onBarUpdate(self, bars, hasNewBar):        
        if hasNewBar:
            df = pd.DataFrame(bars)[["date", "open", "high", "low", "close"]].iloc[:-1]
            df.set_index("date", inplace=True)           
            self.strategy_params['df'] = df 
            df = self.strategy(**self.strategy_params)
            
            #execute trade
            if not self.paper_trading:
                print("executing...")
                target = df["position"][-1]                
                self.executeTrade(target, contract=bars.contract)           
            
            
            #display
            clear_output(wait=True)
            display(df)
        else:
            print("no new bars")
            
            
    def backtest(self, bars):
        df = pd.DataFrame(bars)[["date", "open", "high", "low", "close"]]
        df.set_index("date", inplace=True)           
        self.strategy_params['df'] = df 
        df = self.strategy(**self.strategy_params)     
        display(df)
        
        #calculate PnL
        
        
            
    
    @AddStrategy
    @LogFunctionCall      
    def SMA(self, low, high, df):
        sma_s = low
        sma_l = high
        
        df = df[["close"]].copy()
        df["sma_s"] = df.close.rolling(sma_s).mean()
        df["sma_l"] = df.close.rolling(sma_l).mean()
        df.dropna(inplace=True)
        df["position"] = np.where(df["sma_s"] > df["sma_l"], 1, -1)
        return df       
               
    def cancelBars(self, bars):
        self.ib.cancelHistoricalData(bars)                       
        
    @PrintFunction
    def executeTrade(self, target, contract):
        #get current positions
        
        conID = contract.conId
        try:
            current_positions = [pos.position for pos in ib.positions() if pos.contract.conId == conID][0]
        except:
            current_positions = 0
            
        print('current position:', current_positions)
        trades = target - current_positions        
        
        #execute trade
        if trades > 0:
            print("buy")
            side = "BUY"
            order = MarketOrder(side, abs(trades))
            trade = self.ib.placeOrder(contract, order)
        elif trades < 0:
            print("sell")
            side = "SELL"
            order = MarketOrder(side, abs(trades))
            trade = self.ib.placeOrder(contract, order)    
        else:
            print("no change")
                  
    def tradeReport(self):
        time_lst=[tr.time for tr in self.ib.fills()]
        time_str = [i.strftime("%H:%M:%S") for i in time_lst]
        print('self',self.ib)
        print('times:', time_lst)
        print('fills:', self.ib.fills())
        print([tr.commissionReport for tr in self.ib.fills()])
        df_PnL = util.df([tr.commissionReport for tr in self.ib.fills()])[["execId", "realizedPNL"]].set_index("execId")
        df_PnL["Time"] = time_str
        df_PnL["Cum PnL"] = df_PnL["realizedPNL"].cumsum()
        df_PnL.set_index("Time", inplace=True)
        display(df_PnL)
        return df_PnL
    
    def plotCumPnL(self, df):
        plt.plot(df["Cum PnL"])
   
 
    
    def SetSentiment(self, current_sentiment):                 
        print('set sentiment')
        if self.sentiment.get(self.strategy.__name__) is None:
            self.sentiment[self.strategy.__name__] = current_sentiment
            print('first sentiment')
            return

        if self.sentiment[self.strategy.__name__] != current_sentiment:
            print('change in sentiment')
            
            if current_sentiment == Sentiment.BEAR:
                self.signal[self.strategy.__name__] = Signal.SELL               

            elif current_sentiment == Sentiment.BULL:
                self.signal[self.strategy.__name__] = Signal.BUY                        
        else:
            print('no change in sentiment')

                              
        self.sentiment[self.strategy.__name__] = current_sentiment
       
    
    def AddSignalPlot(self):      
        self.df_plot = pd.DataFrame()
        self.df_plot = self.account.df_account.copy()

        for signal, value_array in self.signal_dict.items():
            missing = len(self.df_plot) - len(value_array)
            value_array = np.insert(value_array, 0, [0]*missing)
            self.df_plot[signal] = value_array    
        self.df_plot.dropna(inplace=True) 
                            
    @AddStrategy
    @LogFunctionCall
    def EMAverage(self, smoothing, period, close):
        if not hasattr(self, "SMA_lst"):
            self.SMA_lst = []
        
        self.SMA_lst.append(close)        
        self.account.updateBTCPrice(BTC_price=close)        
        
        if len(self.SMA_lst) < period:            
            return            
        
        k = smoothing / (1 + period)
        
        if len(self.SMA_lst) == period:            
            self.EMA = np.array([])
            self.EMA = np.append(self.EMA,sum(self.SMA_lst[0:period])/period)
            
#             self.signal_dict['EMA'] = np.array([])
#             self.signal_dict['EMA'] = np.append(self.signal_dict['EMA'],
#                                                 sum(self.SMA_lst[0:period])/period)
            
            return
        
        new_value = close * k  + self.EMA[-1] * (1 - k)
        self.EMA = np.append(self.EMA, new_value)
        print(self.EMA)
        
                
        print('last value:', self.EMA[-1])
        print('close', close)
        if close < self.EMA[-1]:            
            current_sentiment_EMA = Sentiment.BEAR
            print('bearish')
        elif close > self.EMA[-1]:
            current_sentiment_EMA = Sentiment.BULL
            print('bullish')
        else:
            current_sentiment_EMA = Sentiment.NEUTRAL   
            
        self.signal_dict['EMA'] = self.EMA
        self.SetSentiment(current_sentiment_EMA)                       


     
        
    @LogFunctionCall
    @PrintFunction
    def SignalProcess(self, contract, *signals, date, close):        
        print('signal dict',self.signal.items())
        signal_list = [values for keys,values in self.signal.items() if keys in signals]
        
        if len(set(signal_list))==1 and signal_list[0] is not None: #i.e. all signals are the same
            print(date)
            signal = signal_list[0]
            print(str(signal) + ' @ ' + str(self.account.BTC_price))
            order = self.account.trade(contract, signal=signal, quantity=1, date=date,
                                       price=close)
            
            if not self.paper_trading:
                self.trades.append(self.ib.placeOrder(contract, order))
            
            self.ClearSignals(*signals)                
        
        self.account.updateBalance(date=date, close=close)

    
    def ClearSignals(self, *signals):
        for s in signals:
            self.signal[s]=None                  

    def SetStrategy(self, strategy, **kwargs):
        self.strategy = strategy_dict[strategy]
        self.strategy_params = kwargs
        self.strategy_params['self'] = self
        
        
# def Main():
    
if __name__ == "__main__":    
    print("starting algo")
    session_start = pd.to_datetime(datetime.datetime.utcnow()).tz_localize("utc")
    print("start time:", session_start)
    run_time_minutes = float(input("how many minutes to run?"))
#     run_time_minutes = 1
    timedelta = datetime.timedelta(minutes=run_time_minutes)
    end_time = session_start + timedelta    
     
    
    ib = IB()
    contract1 = Future('NQ', '20230616', 'CME')
    contract2 = Future('ES', '20230616', 'CME')
    contract3 = Forex('EURUSD')
    
    contract = contract1
    S = Strategy(ib, client_id=1, contract=contract, paper_trading=False)
    S.SetStrategy('SMA',low=5,high=20)
    
    bars = S.GetBars(duration='1 D', barSize='5 secs', endDate="")
    bars.updateEvent += S.onBarUpdate
    
    while True:
        ib.sleep(5)
        current_time = pd.to_datetime(datetime.datetime.utcnow()).tz_localize("utc")
    #     eastern_time = session_start.tz_convert(pytz.timezone('US/Eastern'))        
        if current_time >= end_time:
            S.executeTrade(target=0, contract=contract)
            S.cancelBars(bars) 
            ib.sleep(10)
#             print('account summary:', ib.accountSummary())            
            start_time = session_start
#             print('trades', S.ib.trades())        
        
            S.tradeReport()
            print("session stopped")
            ib.disconnect()
            break
        else:
            pass
    
    ib.disconnect()
        