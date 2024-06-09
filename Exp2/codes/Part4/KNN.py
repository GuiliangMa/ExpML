import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv
import math


class kNNClassifier:
    def __init__(self, X, y, k=1):
        self.X = X
        self.y = y
        self.k = k

    def KNN_Vn(self, X, k, t):
        '''
        基于 p_{n}(x) = k_{n}/(V_{n}*n) 来计算概率密度
        :param X: 原始数据集的某一类别
        :param k: KNN 的k
        :param t: 测试集
        :return: 一个概率密度函数
        '''
        res = np.zeros((t.shape[0],))
        for i, test in enumerate(t):
            dist = np.linalg.norm(X - test, axis=1)
            dist = np.sort(dist)
            if dist[k - 1] != 0:
                res[i] = k / (X.shape[0] * dist[k - 1] ** X.shape[1])
            else:
                res[i] = k / (X.shape[0] * (1e-5) ** X.shape[1])
        return res

    def kNN_Vn(self, X_train, y_train, X_test, k):
        res = [{} for _ in range(X_test.shape[0])]
        yValueList = y_train.unique().tolist()
        for y in yValueList:
            trainSet = X_train[y_train == y].copy()
            yRes = self.KNN_Vn(trainSet, k, X_test)
            for i in range(len(yRes)):
                # 为 res 中的每一行的字典添加键值对，键为 y，值为 yRes 中对应的元素
                res[i][y] = yRes[i]
        return res

    def gamma(self, x):
        # 递归利用已知的Gamma(1/2) = sqrt(pi)
        if abs(x - 0.5) < 1e-6:
            return math.sqrt(math.pi)
        elif abs(x - 1) < 1e-6:
            return 1
        else:
            return (x - 1) * self.gamma(x - 1)

    def kNN_Euler_Count(self, X_train, y_train, X_test, k):
        '''
        基于欧拉距离的计数方法来计算类条件概率密度
        :param X_train: 训练集
        :param y_train: 训练标签
        :param X_test: 测试集
        :param k: k近邻的k
        :return: 测试集的类条件概率密度
        '''
        res = []
        # 超球体体积系数 $V_n(r) = \frac{\pi^{n/2}}{\Gamma(\frac{n}{2} + 1)} r^n$ 令 $ alpha = \frac{\pi^{n/2}}{\Gamma(\frac{n}{2} + 1)} $
        pi_power = math.pi ** (X_train.shape[1] / 2)
        gamma_value = self.gamma(((X_train.shape[1] / 2) + 1))
        alpha = ((pi_power) / gamma_value) if gamma_value != 0 else float('inf')
        for i, test in enumerate(X_test):
            dist = np.linalg.norm(X_train - test, axis=1)
            KNN_indices = np.argsort(dist)[:k]
            KNN_labels = y_train[KNN_indices]
            d = dist[KNN_indices[-1]]
            if d <= 1e-2:
                d = 1e-2
            # print(X_train.shape[0] * alpha * (d ** X_train.shape[1]))
            class_probs = {
                cls: np.sum(KNN_labels == cls) / (X_train.shape[0] * alpha * (d ** X_train.shape[1]))
                for cls in np.unique(y_train)}
            res.append(class_probs)
        return res

    def mahalanobis_distance(self, x, dataset, invCov):
        '''
        计算单个样本与数据集所有样本的马氏距离
        :param x:
        :param dataset:
        :param invCov:
        :return:
        '''
        dist = np.zeros(dataset.shape[0])
        dataset = dataset.to_numpy()
        for i, data in enumerate(dataset):
            delta = x - data
            dist[i] = np.sqrt(delta.dot(invCov).dot(delta.T))
        return dist

    def kNN_Mahalanobis_Count(self, X_train, y_train, X_test, k):
        '''
        基于马氏距离的计数方法来计算类条件概率密度
        :param X_train: 训练集
        :param y_train: 训练标签
        :param X_test: 测试集
        :param k: k近邻的k
        :return: 测试集的类条件概率密度
        '''
        # 超球体体积系数 $V_n(r) = \frac{\pi^{n/2}}{\Gamma(\frac{n}{2} + 1)} r^n$ 令 $ alpha = \frac{\pi^{n/2}}{\Gamma(\frac{n}{2} + 1)} $
        pi_power = math.pi ** (X_train.shape[1] / 2)
        gamma_value = self.gamma((X_train.shape[1] / 2) + 1)
        alpha = ((pi_power) / gamma_value) if gamma_value != 0 else float('inf')
        cov = np.cov(X_train.T)
        invCov = inv(cov)
        res = []
        for i, test in enumerate(X_test):
            dist = self.mahalanobis_distance(test, X_train, invCov)
            KNN_indices = np.argsort(dist)[:k]
            KNN_labels = y_train[KNN_indices]
            d = dist[KNN_indices[-1]]
            if d == 0:
                d = 1e-5
            class_probs = {
                cls: np.sum(KNN_labels == cls) / (X_train.shape[0] * alpha * (d ** X_train.shape[1]))
                for cls in np.unique(y_train)}
            res.append(class_probs)
        return res

    def execute(self, X_test, k, typ=0):
        yValueList = self.y.unique().tolist()
        prior_prob = {}
        for y in yValueList:
            prior_prob[y] = self.X[self.y == y].shape[0]
        if typ == 0:
            probs = self.kNN_Euler_Count(self.X, self.y, X_test.to_numpy(), k)
        elif typ == 1:
            probs = self.kNN_Mahalanobis_Count(self.X, self.y, X_test.to_numpy(), k)
        elif typ == 2:
            probs = self.kNN_Vn(self.X, self.y, X_test.to_numpy(), k)

        predict = np.zeros(X_test.shape[0])
        # print(probs)
        for i, class_prob in enumerate(probs):
            max_prob = -1
            for y in class_prob:
                current_prob = class_prob[y] * prior_prob[y]
                if current_prob > max_prob:
                    max_prob = current_prob
                    predict[i] = y
        return predict

    def predict(self,X,k):
        return self.execute(X,k)

    def density(self, X_train, y_train, X_test, k, typ=0):
        yValueList = y_train.unique().tolist()
        prior_prob = {}
        if not isinstance(X_test, np.ndarray):
            # 如果不是，转换它为numpy数组
            X_test = np.array(X_test)
        for y in yValueList:
            prior_prob[y] = X_train[y_train == y].shape[0]
        if typ == 0:
            probs = self.kNN_Euler_Count(X_train, y_train, X_test, k)
        elif typ == 1:
            probs = self.kNN_Mahalanobis_Count(X_train, y_train, X_test, k)
        elif typ == 2:
            probs = self.kNN_Vn(X_train, y_train, X_test, k)
        return probs


yMapping = {'didntLike': 0, 'smallDoses': 1, 'largeDoses': 2}
yList = ['didntLike', 'smallDoses', 'largeDoses']


def dealForData(data):
    data['x0'] = data['x0'] / 10000
    data['x1'] = data['x1'] / 10
    data['y'] = data['y'].replace(yMapping)
    return data


def splitForData(data, test_size, random_state):
    np.random.seed(random_state)
    data_shuffled = data.sample(frac=1).reset_index(drop=True)
    train_size = int((1 - test_size) * len(data))
    train_data = data_shuffled[:train_size]
    test_data = data_shuffled[train_size:]
    X_train = train_data.drop('y', axis=1)
    y_train = train_data['y']
    X_test = test_data.drop('y', axis=1)
    y_test = test_data['y']
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    data = pd.read_csv('../../data/e2.txt', sep="\t", names=['x0', 'x1', 'x2', 'y'])
    data = dealForData(data)

    X_train, y_train, X_test, y_test = splitForData(data, 0.2, 7)
    y_test = y_test.to_numpy()
    knn = kNNClassifier()
    y_pred = knn.density(X_train, y_train, X_test, 3, 0)
    print(y_pred)
    values = np.array([d[2] for d in y_pred]).reshape(-1, 1)
    # print(values)
    # accuracy = np.mean(y_pred == y_test)
    # print(f"Accuracy: {accuracy}")
