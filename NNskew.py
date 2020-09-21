import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sklearn
import math


source = "10yrSOD.csv"
model = "linregmodel.pickle"

source_df = pd.read_csv(source)
rows = len(source_df)
source_df = source_df.T

X = []
y = []
y_temp = []
for i in range(7, rows-1):
    try:
        temps = [(float(source_df[i - j]["DailyMaximumDryBulbTemperature"]) - 32) * (5/9) + 273 for j in range(7)]
        temps.reverse()
        X.append(np.array(temps))
        y.append((float(source_df[i + 1]["DailyMaximumDryBulbTemperature"]) - 32) * (5/9) + 273)
        y_temp.append(i)
    except:
        if len(X) > len(y):
            X = X[:-1]

X = np.array(X)
y = np.array(y)

pickle_in = open(model, "rb")
linear = pickle.load(pickle_in)

predictions = linear.predict(X)

inputs = []
outputs = []
for index in range(len(predictions)):
    i = y_temp[index]
    try:
        next_input = []
        next_input.append(i % 365)
        for j in range(7):
            humidity = float(source_df[i - j]["DailyAverageRelativeHumidity"])/ 100
            if math.isnan(humidity):
                raise Exception("hi")
            else:
                next_input.append(humidity)
            pressure = float(source_df[i - j]["DailyAverageStationPressure"]) - 29
            if math.isnan(pressure):
                raise Exception("hi")
            else:
                next_input.append(pressure)
            try:
                next_input.append(float(source_df[i - j]["DailyPrecipitation"]))
            except:
                next_input.append(0)
            try:
                next_input.append(float(source_df[i - j]["DailySnowfall"]))
            except:
                next_input.append(0)

        inputs.append(next_input)
        outputs.append(((y[index]/predictions[index]) - .85) * 4)
        # outputs.append(y[index] - predictions[index])
    except:
        pass
print(inputs)
print(outputs)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(inputs, outputs, test_size=0.1)

tf.keras.backend.set_floatx('float64')
model = keras.Sequential([
    keras.layers.Dense(29, input_dim=29, activation='relu'),
    keras.layers.Dense(5, input_dim=5, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
    ])
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(x_train, y_train,batch_size = 16, epochs=50)

prediction = model.predict(x_test)
print(prediction)
print(outputs)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Tested accuracy:', test_acc)

model.save('skew.h5')