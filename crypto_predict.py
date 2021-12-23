import pandas_datareader as web

data = web.DataReader('ETH-USD', 'yahoo', '2020-12-01', '2021-12-01')

print(len(data))
