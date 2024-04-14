import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def ParzenWindow(X, h, test):
    X = X.to_numpy()
    sigma = 0
    for x in X:
        sigma += np.exp(-np.dot((test - x), (test - x).T) / 2 * h ** 2)
    return sigma


def Parzen(X, y, h, test):
    test = np.array(test)
    yList = y.drop_duplicates().tolist()
    res = np.zeros([len(yList), 1])
    for index,testX in enumerate(test):
        K = np.zeros(len(yList))
        for i, yEle in enumerate(yList):
            XList = X[y == yEle]
            k = ParzenWindow(XList, h, testX)
            K[i] = k / (XList.shape[0] * h ** XList.shape[1])
        res[index] = yList[np.argmax(K)]
    return res


df = pd.read_excel('../../data/exp2-3.xlsx')
df.to_csv('../../data/exp2-3.csv', index=False)
y = df['y']
X = df.drop('y', axis=1)
Xtest = [[0.5, 1.0, 0.0],
         [0.31, 1.51, -0.50],
         [-0.3, 0.44, -0.1]]
yPred = Parzen(X, y, 1, Xtest)
print(yPred)
