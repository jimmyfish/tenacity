import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, Sequential, models
from tensorflow.keras.layers import LSTM, Dropout, Dense
import requests
import json
from discord import Webhook, RequestsWebhookAdapter
import os
import dateutil.parser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

webhook = Webhook.from_url(
    'https://discord.com/api/webhooks/922028899364376596'
    '/2coptzDeEGwNr3IcZanGeKx9oXvH9fI20UrqyyX8bM1lx21O1TP_Zi_Ciqe8yAPkKMrp',
    adapter=RequestsWebhookAdapter())

company = 'ETH-USDT'
epochs = 25
timestamp = 'hourly'

pair = company.split('-')

timestamps = {
    'quarterly': {
        'period': 'M15',
        'days': 11
    },
    'halvely': {
        'period': 'M30',
        'days': 22
    },
    'hourly': {
        'period': 'H1',
        'days': 42
    },
    '4hours': {
        'period': 'H4',
        'days': 168
    },
    'daily': {
        'period': 'D1',
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


def requestData(params):
    return requests.get('https://api.hitbtc.com/api/3/public/price/history', params=params)


def getData(pair, trainData=True):
    today = dt.datetime.now()
    start = today - dt.timedelta(days=1)
    end = today

    if (trainData):
        start = today - dt.timedelta(days=timestamps[timestamp]['days'])
        end = today - dt.timedelta(days=1)

    print(
        f"Fetching data from {start.strftime('%d/%m/%y %H:%M')} until {end.strftime('%d/%m/%y %H:%M')}...", end="",
        flush=True)

    params = {
        "from": pair[0],
        "to": pair[1],
        "limit": 1000,
        "sort": "ASC",
        "period": timestamps[timestamp]['period'],
        "since": start,
        "until": end
    }

    try:
        request = requestData(params)
    except:
        request = requestData(params)

    print("DONE")

    response = json.loads(request.text)[pair[0]]['history']
    collections = []

    collections = {
        'Date': [i['timestamp'] for i in response],
        'Close': [float(i['close']) for i in response]
    }

    collections = pd.DataFrame(data=collections)
    collections = collections.set_index('Date')

    return collections


# Core code

collections = getData(pair, True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(collections['Close']).reshape(-1, 1))

prediction_days = 60

x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building RNN Model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True,
               input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Prepare test data
test_data = getData(pair, False)
actual_prices = test_data

print(len(test_data))
print(len(collections))

total_dataset = pd.concat((collections['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

real_data = [model_inputs[len(model_inputs + 1) - prediction_days:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

print(f"Training model using {epochs} epoch...", end="", flush=True)
hist = model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
print("DONE")

print(f"Sending prediction to Discord...", end="", flush=True)

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
today = dt.datetime.now()
last_data = collections['Close'][-1]

webhook.send(
    f"---------------------\n{str(today)}(GMT+0)\n---------------------\nLast tick at {dateutil.parser.isoparse(test_data.index[-1]).strftime('%D %T')}(GMT+0): {last_data}\n{company}'s price forecast for {captions[timestamp]} : {prediction[0][0]}\nDirection : " + (
        "LONG" if last_data < prediction[0][0] else "SHORT"))
print("DONE")

# print(f"---------------------\n{str(today)}\n---------------------\nLast tick : {last_data}\nPrediction {company} price for {captions[timestamp]} : {prediction[0][0]}\nAI decision making : " + ("LONG" if last_data < prediction[0][0] else "SHORT"))
