import requests
import datetime as dt
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import os
from discord import Webhook, RequestsWebhookAdapter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

base_url = "https://fapi.binance.com"
pair = 'ETH-USDT'
period = '4hours'

# ADVANCED SETTINGS - DO NOT TOUCH CODE BELOW THIS
company = "".join(pair.split('-'))
epochs = 25
prediction_steps = 60
webhook = Webhook.from_url(
    "https://discord.com/api/webhooks/920232996743835659/vLsy6QW7-Av0gnd4f67JXUcKbhwWVOSfhJRuOQirUkH65YM9kw6hLKYudk2sp5nxgI9R",
    adapter=RequestsWebhookAdapter())

timestamps = {
    'quarterly': {
        'period': '15m',
        'days': 3,
        'limit': 500
    },
    'halvely': {
        'period': '30m',
        'days': 22,
        'limit': 1000
    },
    'hourly': {
        'period': '1h',
        'days': 41,
        'limit': 1000
    },
    '4hours': {
        'period': '4h',
        'days': 120,
        'limit': 800
    },
    'daily': {
        'period': '1d',
        'days': 1001,
        'limit': 800
    },
    'weekly': {
        'period': 'D7',
        'days': 1695,
        'limit': 1000
    },
    'monthly': {
        'period': '1M',
        'days': 1695,
        'limit': 1000
    }
}
captions = {
    'quarterly': 'the next 15 minutes',
    'halvely': 'the next 30 minutes',
    'hourly': 'the next hour',
    '4hours': 'the next 4 hours',
    'daily': 'tomorrow',
    'weekly': 'the next week',
    'monthly': 'the next month'
}

# CORE CODE - DO NOT IN ANY CIRCUMSTANCES CHANGE THE CODE
today = dt.datetime.now()
start = today - dt.timedelta(days=timestamps[period]['days'])

params = {
    'limit': timestamps[period]['limit'],
    'symbol': company,
    'interval': timestamps[period]['period'],
    'startTime': round(start.timestamp() * 1000)
}

request = requests.get(f"{base_url}/fapi/v1/klines", params=params)
response = json.loads(request.text)

data = []
for i in response:
    closeTime = (dt.datetime.fromtimestamp(i[6] / 1000) + dt.timedelta(seconds=1)).strftime('%D %T')
    data.append({
        'Date': closeTime,
        'Close': float(i[4])
    })

print(len(data))
