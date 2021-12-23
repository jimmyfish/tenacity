import requests as r
import datetime as dt
import json

today = dt.datetime.now()
start = today - dt.timedelta(days=1)

base_url = 'https://fapi.binance.com'

params = {
    'limit': 1000,
    'symbol': 'ETHUSDT',
    'interval': '15m',
    'startTime': round(start.timestamp() * 1000),
    'endTime': round(today.timestamp() * 1000)
}

response = r.get(f"{base_url}/fapi/v1/klines", params=params)

response = json.loads(response.text)

data = []

for i in response:
    newdt = dt.datetime.fromtimestamp(i[6] / 1000)
    data.append({
        'date': (newdt + dt.timedelta(seconds=1)).strftime('%D %T'),
        'close': i[4]
    })

del data[-1]  # Remove last data

print(data)
