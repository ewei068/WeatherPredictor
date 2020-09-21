import pandas as pd
import numpy as np

source = "2020SOD.csv"

source_df = pd.read_csv(source)
rows = len(source_df)
source_df = source_df.T

error = 0
for i in range(9, 229):
    temps = [int(source_df[i - j]["DailyMaximumDryBulbTemperature"]) for j in range(7)]
    temps.reverse()

    # First we compute our estimated coefficients
    A = np.zeros((7, 3))
    b = np.zeros((7, 1))
    for k in range(7):
        A[k][0] = k ** 2
        A[k][1] = k
        A[k][2] = 1
        b[k] = temps[k]

    At = np.transpose(A)
    AtA1 = np.linalg.inv(np.dot(At, A))
    params = np.dot(np.dot(AtA1, At), b)
    # print(params)
    '''
    for l in range(7):
        print("Day " + str(l) + ": " + str(temps[l]))
            '''

    prediction = params[0] * (7 ** 2) + params[1] * 7 + params[2]
    prediction = prediction[0]
    # print("Prediction: " + str(prediction))

    actual = int(source_df[i + 1]["DailyMaximumDryBulbTemperature"])
    # print("Actual: " + str(actual))

    error += abs((actual - prediction) / actual)

error /= 220
accuracy = (1 - error) * 100
print(accuracy)

