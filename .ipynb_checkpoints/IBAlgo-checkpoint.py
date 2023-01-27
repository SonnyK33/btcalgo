import pandas as pd
from matplotlib import pyplot as plt
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.tsa.ar_model import AutoReg
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error
from math import sqrt
# import pandas_ta as ta
import numpy as np
from ib_insync import *
from collections import deque
from enum import Enum
import datetime
import logging
# util.startLoop()

contract_dict = {'Stock':Stock, 'Crypto':Crypto}

today = datetime.datetime.today().date()
logging.basicConfig(level = logging.INFO,
                   filename = 'algo_log_'+str(today)+'_.log',
                   filemode='w')

class Algo():
    def __init__(self, client, addr, port, client_id):
        self.client = client
        self.addr = addr
        self.port = port
        self.client_id = client_id      
        self.client.connect(addr, port, client_id)
        
    def SetPaperBalances(self):
        self.USD_balance = 1000
        self.BTC_balance = 0
        
        print('USD balance: {}'.format(self.USD_balance))
        print('BTC balance: {}'.format(self.BTC_balance))
    
    def GetContract(self, contract_type, ticker, exchange, currency):
        return contract_dict[contract_type](ticker, exchange, currency)
    
    def GetHistoricalData(self, contract, duration='1 D', bar_size='1 hour'):
        bars = self.client.reqHistoricalData(contract=contract, endDateTime='', durationStr=duration,
                                    barSizeSetting=bar_size, whatToShow='MIDPOINT', useRTH=True)
        return bars
    
    def BuildEMA(self, slow_period, fast_period):
        """using deques instead of dataframes"""
        self.slow_ema = deque(maxlen=slow_period)
        self.fast_ema = deque(maxlen=fast_period)
        
        """need to employ scanner to pull in data"""
        
           
    def GetDMI(self, df):
        df_adx = ta.adx(high=df['high'], low=df['low'], close=df['close'])
        df['ADX'] = df_adx['ADX_14']
        df['DMN'] = df_adx['DMN_14']
        df['DMP'] = df_adx['DMP_14']
        df['DM_diff'] = df['DMP'] > df['DMN']
        return df
        
    
    
    def GetMovingAverages(self, df, low, high, plot=True):
        df_MA = df.copy()
        if len(df) < high:
            return

        
        df_MA[str(low)+'_average'] = df.close.rolling(low).mean()
        df_MA[str(high)+'_average'] = df.close.rolling(high).mean()
#         df_MA.dropna(inplace=True)
        df_MA['MA_diff'] = df_MA[str(high)+'_average'] > df_MA[str(low)+'_average']
        
        if plot:
            df_MA['close'].plot(legend=True)
            df_MA[str(low)+'_average'].plot(legend=True)
            df_MA[str(high)+'_average'].plot(legend=True)
        
#         display('end of mov avg', df_MA)
        return df_MA
    
    
    def PlacePaperOrder(self, order_type, price=None, quantity=None, date=None):
        print('placing order:', order_type)

        if order_type=='sell':
            print(date)
            print('{} @ {}'.format(order_type, price))
            self.BTC_balance -= quantity
            self.USD_balance += quantity * price
            

        elif order_type=='buy':
            print(date)
            print('{} @ {}'.format(order_type, price))
            self.BTC_balance += quantity
            self.USD_balance -= quantity * price

        else:
            return 


        print('USD balance: {}'.format(self.USD_balance))
        print('BTC balance: {}'.format(self.BTC_balance))
        account_balance = self.USD_balance + self.BTC_balance * price
        print('Account balance: {}\n'.format(account_balance))       

    def RunDMI(self, df, ADX_MIN=20):
        for count, row in enumerate(df.iterrows()):
            if count==0:
                sentiment = row[1]['DM_diff']
            else:
                if (row[1]['ADX'] >= ADX_MIN):
                    if row[1]['DM_diff'] == sentiment:
                        self.PlacePaperOrder('no change')
                    else:
                        if row[1]['DM_diff']:
                            self.PlacePaperOrder('buy',row[1]['close'],1, row[1]['date'])
                        else:
                            self.PlacePaperOrder('sell', row[1]['close'],1, row[1]['date'])
                    sentiment = row[1]['DM_diff']
                else: continue                                     
                        
    
    def RunStrategy(self, df):   

        for count, row in enumerate(df.iterrows()):
            if count==0:
                sentiment = row[1]['MA_diff']
            else:
                if row[1]['MA_diff'] == sentiment:
                    self.PlacePaperOrder('no change')
                else:
                    if row[1]['MA_diff']:
                        self.PlacePaperOrder('sell',row[1]['close'],1)
                    else:
                        self.PlacePaperOrder('buy', row[1]['close'],1)
                    sentiment = row[1]['MA_diff']


def Main(client_id):
    
    ib = IB()
    algo = Algo(ib, '127.0.0.1', 7497, client_id=client_id)
    
    contract = algo.GetContract('Crypto','BTC', 'PAXOS', 'USD')
    
    bars = algo.GetHistoricalData(contract, '180 D', '1 day')
    
    df = util.df(bars)
    
    algo.SetPaperBalances()
    
#     df_MA =  algo.GetMovingAverages(df, 10, 30, True)
    df_DMI = algo.GetDMI(df)
    
#     algo.RunStrategy(df_MA)
    algo.RunDMI(df_DMI)
    
    print('USD balance: {}'.format(algo.USD_balance))
    print('BTC balance: {}'.format(algo.BTC_balance))

    account_value = algo.USD_balance + algo.BTC_balance * bars[-1].close
    print('Account value: {}'.format(account_value))


Sentiment = Enum('Sentiment','BEAR NEUTRAL BULL')
Signal = Enum('Signal','BUY SELL')



class Account():
    def __init__(self, USD_balance: int, BTC_balance: int, BTC_price: float = None):
        self.USD_balance = USD_balance
        self.BTC_balance = BTC_balance
        self.BTC_price = BTC_price                 
        self.df_account = pd.DataFrame(columns=['Date', 'USD', 'BTC', 'Account Value', 'BTC Px'])
        
    def __call__(self):
        return self.returnBalance()
        
    def __repr__(self):
        return 'USD balance: '+ str(self.USD_balance) + '\nBTC balance: ' + str(self.BTC_balance) + '\nAccount Value: '+str(self.USD_balance + self.BTC_balance * self.BTC_price) + '\nBTC Price:' + str(self.BTC_price)
        
        
    def updateBTCPrice(self, BTC_price: float):
        self.BTC_price = BTC_price
        
    def closePosition(self):
        self.BTC_balance = 0
        self.USD_balance -= self.BTC_balance * self.BTC_price
    
    def tradeBTC(self, signal: Signal, quantity: int, date: datetime): #-> order:
        logging.info(str(signal), str(quantity)+' @ ', str(self.BTC_price))
        
        direction = 1 if signal==Signal.BUY else -1
        
        self.BTC_balance += direction * quantity
        self.USD_balance -= direction * quantity * self.BTC_price
        
        order_type = 'BUY' if signal==Signal.BUY else 'SELL'
        
        order = LimitOrder(order_type, quantity, round(self.BTC_price, 0),
                           tif='GTC')
        return order         

                
    def updateBalance(self, date):
        total = self.USD_balance + self.BTC_balance * self.BTC_price
        self.df_account.loc[len(self.df_account.index)] = [date, self.USD_balance,
                                                          self.BTC_balance, total, self.BTC_price]       
        
    def returnBalance(self):
        return self.df_account
    
    def plotBalance(self):
        self.df_account.plot(x='Date', y=['Account Value'])
        
        

strategy_dict = dict()

def AddStrategy(strategy_fn):
    strategy_dict[strategy_fn.__name__] = strategy_fn
    return strategy_fn
  
def LogFunctionCall(fn):
    logging.info(fn.__name__+' called')
    return fn


class Strategy():
    def __init__(self, account: Account, ib, client_id: int, paper_trading: bool = True):
        self.client_id = client_id
        self.ib = ib     
        self.algo = Algo(ib, '127.0.0.1', 7497, client_id=self.client_id)
        self.algo.SetPaperBalances()
        self.contract = Crypto('BTC', exchange='PAXOS', currency='USD')
        self.account = account
        self.paper_trading=paper_trading
        self.sentiment = dict()
        self.signal = dict()
        self.trades = list()
        self.low_deque = deque()
        self.high_deque = deque()         

    def onBarUpdate(self, bars, newBar):
        if newBar:
#             print(bars)
            print('on bar update', bars[-1])
            self.strategy_params['close'] = bars[-1].close 
            self.strategy(**self.strategy_params)
            self.SignalProcess(self.strategy.__name__, date=bars[-1].time)
    

    @AddStrategy
    @LogFunctionCall
    def MovingAverage(self, low, high, close):
        
        if not (hasattr(self, "low_deque") and hasattr(self, "high_deque")):
            low_deque_len, high_deque_len = low, high
            self.low_deque = deque(maxlen=low_deque_len)
            self.high_deque = deque(maxlen=high_deque_len)       
                
        self.low_deque.append(close)
        self.high_deque.append(close)
        self.account.updateBTCPrice(BTC_price=close)        
        
        if len(self.high_deque) < self.high_deque.maxlen:
            return
        low_avg = sum(self.low_deque)/len(self.low_deque)
        high_avg = sum(self.high_deque)/len(self.high_deque)

        if low_avg < high_avg:
            current_sentiment_MA = Sentiment.BEAR
        elif low_avg > high_avg:
            current_sentiment_MA = Sentiment.BULL
        else:
            current_sentiment_MA = Sentiment.NEUTRAL
        
        print('low avg:', low_avg, 'high avg:', high_avg)
        
        if self.sentiment.get('MA') is None:
            print('1st change')
            self.sentiment['MA'] = current_sentiment_MA
            return

        if self.sentiment['MA'] != current_sentiment_MA:
            print('previous sentiment:', self.sentiment.get('MA'))
            print('current sentiment:', current_sentiment_MA)
            
            if current_sentiment_MA == Sentiment.BEAR:
                self.signal['MA'] = Signal.SELL               

            elif current_sentiment_MA == Sentiment.BULL:
                self.signal['MA'] = Signal.BUY            
            print(self.signal['MA'])
        else:
            print('no trades')
                              
        self.sentiment['MA'] = current_sentiment_MA
        print('current sentiment: ', current_sentiment_MA)
    
    @LogFunctionCall
    def SignalProcess(self, *signals, date):
        signal_list = [values for keys,values in self.signal.items() if keys in signals]

        if len(set(signal_list))==1 and signal_list[0] is not None: #i.e. all signals are the same
            print(date)
            signal = signal_list[0]
            print(str(signal) + ' @ ' + str(self.account.BTC_price))
            order = self.account.tradeBTC(signal=signal, quantity=1, date=date)
            
            if not self.paper_trading:
                self.trades.append(self.ib.placeOrder(self.contract, order))
            
            self.ClearSignals(*signals)                
        
        self.account.updateBalance(date=date)
        print(self.account)
    
    def ClearSignals(self, *signals):
        for s in signals:
            self.signal[s]=None      

    def SetStrategy(self, strategy, **kwargs):
#         self.strategy_name=strategy
        self.strategy = strategy_dict[strategy]
        self.strategy_params = kwargs
        self.strategy_params['self'] = self           

    
    def RunStrategy(self):       
       
        # low, high = 5, 10
        # self.low_deque = deque(maxlen=low)
        # self.high_deque = deque(maxlen=high)

        bars = self.ib.reqRealTimeBars(contract=self.contract,
                                  barSize=5,
                                  whatToShow='MIDPOINT',
                                 useRTH=False)
        bars.updateEvent += self.onBarUpdate

        self.ib.sleep(20)
        self.ib.cancelRealTimeBars(bars)        
    
        
    def Backtest(self, duration='1 D', barSize='5 secs'):
        self.paper_trading = True
        bars = self.ib.reqHistoricalData(
            self.contract, endDateTime='', durationStr=duration,
            barSizeSetting=barSize, whatToShow='MIDPOINT', useRTH=True)
        
        df = util.df(bars)
        
        low, high = 5, 10
        low_deque = deque(maxlen=low)
        high_deque = deque(maxlen=high)
        for row in df.itertuples(name='bar', index=False):            
            low_deque.append(row.close)
            high_deque.append(row.close)
            self.account.updateBTCPrice(BTC_price=row.close)
            self.MovingAverage(low_deque, high_deque, row.close)
            self.SignalProcess('MA', date=row.date)



        
        