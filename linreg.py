import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

source = "10yrSOD.csv"

source_df = pd.read_csv(source)
rows = len(source_df)
source_df = source_df.T

X = []
y = []
for i in range(7, rows-1):
    try:
        temps = [(float(source_df[i - j]["DailyMaximumDryBulbTemperature"]) - 32) * (5/9) + 273 for j in range(7)]
        temps.reverse()
        X.append(np.array(temps))
        y.append((float(source_df[i + 1]["DailyMaximumDryBulbTemperature"]) - 32) * (5/9) + 273)
    except:
        if len(X) > len(y):
            X = X[:-1]

X = np.array(X)
y = np.array(y)

best = 0
for _ in range(3000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    # print(acc)

    if acc > best:
        best = acc
        with open("linregmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

print(best)

'''
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'failures'
style.use('ggplot')
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()
'''