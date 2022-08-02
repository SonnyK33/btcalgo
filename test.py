from locale import currency
from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# contract = Forex('EURUSD')
contract = Crypto('BTC', exchange='PAXOS', currency='USD')

bars = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='30 D',
    barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True)

# convert to pandas dataframe:
df = util.df(bars)
print(df)
ib.disconnect()


