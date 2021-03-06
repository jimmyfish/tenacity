# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, Sequential, models
from tensorflow.keras.layers import Dense, LSTM, Dropout
import os
import shutil
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

company = 'EMTK.JK'

# args = sys.argv
# try:
#     epoch_price = int(args[-1])
# except ValueError:
#     epoch_price = 128

# company = args[-2] if args[-2] != os.path.basename(__file__) else 'ETH-USD'
epoch_price = 128

today = dt.datetime.now()

start = dt.datetime(today.year - 1, today.month, today.day)

print(f"Fetching data for {company} from {start.strftime('%b')} {start.year} until {today.strftime('%b')} {today.year}... ", end='', flush=True)
data = web.DataReader(company, 'yahoo', start, today)

last_data = data['Close'].values[-1]
print('DONE')

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building RNN Learning model
model = Sequential()

# Adding invisible layer(s)
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

print(f"Training model with {epoch_price} epoch(s)... ", end="", flush=True)

model.fit(x_train, y_train, epochs=epoch_price, batch_size=32, verbose=0)

print('DONE', end=" ")

test_start = dt.datetime(today.year - 1, today.month, today.day)

test_data = web.DataReader(company, 'yahoo', test_start, today)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - 60:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Graph next day's price
real_data = [model_inputs[len(model_inputs + 1) - 60:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"\nLast tick : {last_data}")
print(f"Prediction {company} price for tomorrow : {prediction[0][0]}")

print("AI decision making : " + ("LONG" if last_data < prediction[0][0] else "SHORT"))