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
period = 'hourly'

# ADVANCED SETTINGS - DO NOT TOUCH CODE BELOW THIS
company = "".join(pair.split('-'))
epochs = 128
prediction_steps = 60
webhook = Webhook.from_url("https://discord.com/api/webhooks/920232996743835659/vLsy6QW7-Av0gnd4f67JXUcKbhwWVOSfhJRuOQirUkH65YM9kw6hLKYudk2sp5nxgI9R", adapter=RequestsWebhookAdapter())

timestamps = {
    'quarterly': {
        'period': '15m',
        'days': 10
    },
    'halvely': {
        'period': '30m',
        'days': 22
    },
    'hourly': {
        'period': '1h',
        'days': 41
    },
    '4hours': {
        'period': '4h',
        'days': 168
    },
    'daily': {
        'period': '1d',
        'days': 1001
    },
    'weekly': {
        'period': 'D7',
        'days': 1695
    },
    'monthly': {
        'period': '1M',
        'days': 1695
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
    'limit': 1000,
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

# Remove not closed timestamp
del data[-1]
data = pd.DataFrame(data=data)
data = data.set_index('Date')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(data['Close']).reshape(-1, 1))

x_train, y_train = [], []

for x in range(prediction_steps, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_steps:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Prepare model inputs
model_inputs = data[len(data) - prediction_steps:]['Close'].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

real_data = [model_inputs[len(model_inputs + 1) - prediction_steps:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

print(f"Training model using {epochs} epoch...", end='', flush=True)
hist = model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
print("DONE")

# Calculate forecast
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
last_data = data['Close'][-1]

webhook.send(
    f"---------------------\n{today.strftime('%D %T')}\n---------------------\nSymbol : {pair}\n {data.index[-1]}: {round(last_data)}\n{captions[period]} : {round(prediction[0][0])}\nDirection : " + (
        "LONG" if last_data < prediction[0][0] else "SHORT"))
