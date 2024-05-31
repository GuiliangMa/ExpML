import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from KNN import kNNClassifier
import matplotlib

# matplotlib.use('TkAgg')

yMapping = {'didntLike': 0, 'smallDoses': 1, 'largeDoses': 2}
yList = ['didntLike', 'smallDoses', 'largeDoses']


def dealForData(data):
    data['x0'] = data['x0'] / 10000
    data['x1'] = data['x1'] / 10
    data['y'] = data['y'].replace(yMapping)
    return data


def DrawScatterPlot(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 为每个分类的点设置不同的颜色和标签
    colors = ['red', 'green', 'blue']
    labels = ['didntLike', 'smallDoses', 'largeDoses']
    for i in range(3):
        subset = X[y == i]
        ax.scatter(subset['x0'], subset['x1'], subset['x2'], c=colors[i], label=labels[i])
    # 设置图例和坐标轴标签
    ax.legend()
    ax.set_xlabel('X0')
    ax.set_ylabel('X1')
    ax.set_zlabel('X2')
    plt.show()

def splitForData(data, test_size, random_state):
    np.random.seed(random_state)
    data_shuffled = data.sample(frac=1).reset_index(drop=True)
    train_size = int((1 - test_size) * len(data))
    train_data = data_shuffled[:train_size]
    test_data = data_shuffled[train_size:]
    X_train = train_data.drop('y', axis=1)
    y_train = train_data['y']
    X_test = test_data.drop('y', axis=1)
    y_test = test_data['y'].copy()
    return X_train, y_train, X_test, y_test

def plotAccuracyK(X_train, y_train, X_test, y_test):
    x = np.linspace(1,20,20)
    accuracy = np.zeros(20)
    for k in range(20):
        knn = kNNClassifier()
        y_pred = knn.execute(X_train, y_train, X_test,5, 0)
        accuracy[k]=np.mean(y_pred == y_test)
    plt.plot(x, accuracy,'r-')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()

def plotAccuracySize(data,random_state=42):
    x = np.linspace(0.05,0.4,8)
    accuracy = np.zeros(8)
    for i,size in enumerate(x):
        X_train, y_train, X_test, y_test = splitForData(data, size, 7)
        y_test = y_test.to_numpy()
        knn = kNNClassifier()
        y_pred = knn.execute(X_train, y_train, X_test,5, 0)
        accuracy[i] = np.mean(y_pred == y_test)
    plt.plot(x, accuracy, 'r-')
    plt.xlabel('size')
    plt.ylabel('accuracy')
    plt.show()



data = pd.read_csv('../../data/e2.txt', sep="\t", names=['x0', 'x1', 'x2', 'y'])
data = dealForData(data)

X_train, y_train, X_test, y_test = splitForData(data, 0.2, 7)
y_test = y_test.to_numpy()
DrawScatterPlot(X_train,y_train)
knn = kNNClassifier()

# 基于一般欧氏距离方法
y_pred = knn.execute(X_train, y_train, X_test,5,0)
accuracy = np.mean(y_pred == y_test)
print(f'一般欧式-Accuracy: {accuracy}')

# 基于马氏距离方法
y_pred = knn.execute(X_train, y_train, X_test,5,1)
accuracy = np.mean(y_pred == y_test)
print(f'一般马氏-Accuracy: {accuracy}')

# 基于单类欧式距离的方法
y_pred = knn.execute(X_train, y_train, X_test,5,2)
accuracy = np.mean(y_pred == y_test)
print(f'独类欧式-Accuracy: {accuracy}')

plotAccuracyK(X_train, y_train, X_test, y_test)
plotAccuracySize(data)

Temp = [40000,8,0.9]
Temp = pd.DataFrame([Temp], columns=['x0', 'x1', 'x2'])
y_pred = knn.execute(X_train, y_train, Temp,5,1)
index = int(y_pred[0])  # 转换为整数
print(yList[index])


