import numpy as np
from numpy.linalg import inv

class KNNClassifier:

    def KNN_Vn(self,X, k, t):
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

    def KNN_Euler_Count(self,X, y, k, t):
        '''
        基于在一个欧氏距离内计数来计算概率密度
        :param X: 原始数据集
        :param k: KNN 的k
        :param t: 测试集
        :return: 一个概率密度函数
        '''
        res = np.zeros((t.shape[0], np.unique(y).size))
        for i, test in enumerate(t):
            dist = np.linalg.norm(X - test, axis=1)
            KNN_indices = np.argsort(dist)[:k]
            KNN_labels = y[KNN_indices]
            class_probs = [np.sum(KNN_labels == cls) / k for cls in np.unique(y)]
            res[i] = class_probs
        return res

    def mahalanobis_distance(self,x, dataset, invCov):
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

    def KNN_Mahalanobis_Count(self,X, y, k, t):
        '''
            基于在一个马氏距离内计数来计算概率密度
            :param X: 原始数据集
            :param k: KNN 的k
            :param t: 测试集
            :return: 一个概率密度函数
        '''
        cov = np.cov(X.T)
        invCov = inv(cov)
        res = np.zeros((t.shape[0], np.unique(y).size))
        for i, test in enumerate(t):
            dist = self.mahalanobis_distance(test, X, invCov)
            KNN_indices = np.argsort(dist)[:k]
            KNN_labels = y[KNN_indices]
            class_probs = [np.sum(KNN_labels == cls) / k for cls in np.unique(y)]
            res[i] = class_probs
        return res

    def exeKNN_Vn(self,X_train, y_train, k, X_test):
        yValueList = y_train.unique().tolist()
        prob = np.zeros((len(yValueList), X_test.shape[0]))
        for y in yValueList:
            trainSet = X_train[y_train == y].copy()
            prob[y] = self.KNN_Vn(trainSet, k, X_test)
        return prob.T

    def exe(self,X_train, y_train, k, X_test, typ=0):
        yValueList = y_train.unique().tolist()
        prob = np.zeros((len(yValueList), X_test.shape[0]))
        prior_prob = np.zeros(len(yValueList))
        for y in yValueList:
            prior_prob[y] = X_train[y_train == y].shape[0]

        if typ == 0:
            prob = self.KNN_Euler_Count(X_train, y_train, k, X_test.to_numpy())
        elif typ == 1:
            prob = self.exeKNN_Vn(X_train, y_train, k, X_test.to_numpy())
        elif typ == 2:
            prob = self.KNN_Mahalanobis_Count(X_train, y_train, k, X_test.to_numpy())

        post_prob = np.zeros(prob.shape)
        post_prob = prior_prob * prob
        res = np.argmax(post_prob, axis=1)
        return res

    def countDensity(self,X_train, y_train, k, X_test, typ=0):
        yValueList = y_train.unique().tolist()
        prob = np.zeros((len(yValueList), X_test.shape[0]))
        if typ == 0:
            prob = self.KNN_Euler_Count(X_train, y_train, k, X_test)
        elif typ == 1:
            prob = self.exeKNN_Vn(X_train, y_train, k, X_test)
        elif typ == 2:
            prob = self.KNN_Mahalanobis_Count(X_train, y_train, k, X_test)
        return prob