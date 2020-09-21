import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from tensorflow import keras
from matplotlib import style
import math

source = "2020SOD.csv"
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
predict_temp = []
actual_temp = []
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

        predict_temp.append(predictions[index])
        actual_temp.append(y[index])
        # outputs.append(y[index] - predictions[index])
    except:
        pass
print(inputs)
print(outputs)

inputs = np.array(inputs)
outputs = np.array(outputs)

model = keras.models.load_model('skew.h5')
prediction = model.predict(inputs)

error = 0
for i in range(len(inputs)):
    multiplier = (prediction[i] / 4) + .85
    pred = ((predict_temp[i] * multiplier) -273) * 1.8 + 32
    act = (actual_temp[i] - 273) * 1.8 + 32

    error += abs((act - pred) / act)

error /= len(predictions)
accuracy = (1 - error) * 100
print(accuracy[0])