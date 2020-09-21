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

error = 0
for i in range(len(predictions)):
    pred = (predictions[i] -273) * 1.8 + 32
    act = (y[i] - 273) * 1.8 + 32

    error += abs((act - pred) / act)

error /= len(predictions)
accuracy = (1 - error) * 100
print(accuracy)