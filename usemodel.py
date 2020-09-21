import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

model = "linregmodel.pickle"
input = np.array([[301.4,299.8,297.6,299.3,303.7,300.4,303.7]])

pickle_in = open(model, "rb")
linear = pickle.load(pickle_in)

predictions = linear.predict(input)
print(predictions)
for _ in range(6):
    input = np.append(input[0][1:], predictions)
    input = np.array([input])

    predictions = linear.predict(input)
    print(predictions)
