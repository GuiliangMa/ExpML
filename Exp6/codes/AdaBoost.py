import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# 将项目根目录添加到sys.path
sys.path.append(project_root)

from Exp1.codes.NaiveBayesClassifier import NaiveBayesClassifier
from Exp2.codes.Part4.KNN import kNNClassifier
from Exp3.codes.CART import CART
from Exp4.codes.NNetwork import NeuralNetwork
from Exp5.codes.SVM import SVM
from IPython.display import Image, display


def to_one_hot(labels, num_classes):
    one_hot_labels = np.zeros((labels.size, num_classes))
    for index, label in enumerate(labels):
        one_hot_labels[index, label] = 1
    return one_hot_labels


class AdaBoost:
    def __init__(self, T):
        self.weight = None
        self.T = T
        self.modelList = []
        self.alphaList = []

    def pick(self, X, y, w):
        if isinstance(X, pd.DataFrame):
            # 获取索引数组并进行加权采样
            indices = np.random.choice(X.index, size=X.shape[0], p=w)
            # 使用iloc通过位置索引获取采样后的数据
            X_train = X.loc[indices].reset_index(drop=True)
            y_train = y.loc[indices].reset_index(drop=True)
        else:
            # 对于Numpy数组，直接进行加权采样
            indices = np.random.choice(range(X.shape[0]), size=X.shape[0], p=w)
            X_train = X[indices]
            y_train = y[indices]
        return X_train, y_train

    def fit(self, X, y, model_type='Bayes'):
        self.weight = np.ones(X.shape[0]) / X.shape[0]
        self.model_type = model_type
        for index in range(self.T):
            X_train, y_train = self.pick(X, y, self.weight)
            if model_type == 'Bayes':
                y_train = y_train.replace(0, -1)
                model = NaiveBayesClassifier(alpha=1)
                model.fit(X_train, y_train)

            if model_type == 'BPNet':
                model = NeuralNetwork(layers=[X_train.shape[1], 2048, 512, 256, 128, 2],
                                      activations=['relu', 'relu', 'relu', 'relu', 'softmax'],
                                      dropout_rate=0,
                                      loss='cross_entropy')
                X_train = np.array(X_train)
                y_train = np.array(y_train).reshape(-1, 1)
                y_train = to_one_hot(y_train, 2)

                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_train = X_train[indices]
                y_train = y_train[indices]

                model.train(X_train, y_train, 30, 32, 0.001, patience=5, decay_factor=0.1, task='Exp6',
                            val_size=0.1)

            if model_type == 'SVM':
                y_train = y_train.replace(0, -1)
                model = SVM(C=10)
                model.fit(X_train, y_train, kernel='rbf', gamma=100)

            if model_type == 'KNN':
                y_train = y_train.replace(0, -1)
                model = kNNClassifier(X_train, y_train)

            if model_type == 'CART':
                model = CART(max_depth=5, min_samples_split=5)
                model.fit(X_train, y_train, isPruned=True)

            if model_type == 'KNN':
                preds = model.predict(X, 7)
            else:
                preds = model.predict(X)

            y_temp = np.array(y)
            preds = np.array(preds)
            preds = np.sign(preds).astype(int)
            preds = np.where(preds == -1, 0, preds).reshape(y_temp.shape)
            error = np.sum(self.weight * (y_temp != preds))

            best_model = model
            best_error = error
            for models in self.modelList:
                if model_type == 'KNN':
                    preds = models.predict(X, 7)
                else:
                    preds = models.predict(X)
                y_temp = np.array(y)
                preds = np.array(preds)
                preds = np.sign(preds).astype(int)
                preds = np.where(preds == -1, 0, preds).reshape(y_temp.shape)
                error = np.sum(self.weight * (y_temp != preds))
                if error < best_error:
                    best_error = error
                    best_model = models

            error = best_error
            alpha = 1 / 2 * np.log((1 - error) / error)
            self.alphaList.append(alpha)
            if model_type == 'KNN':
                preds = best_model.predict(X, 7)
            else:
                preds = best_model.predict(X)

            y_temp = np.array(y)
            preds = np.array(preds)
            preds = np.sign(preds).astype(int)
            preds = np.where(preds == -1, 0, preds).reshape(y_temp.shape)
            error = np.sum(self.weight * (y_temp != preds))

            Z = np.sum(self.weight * np.exp(-alpha * y_temp * preds))
            self.weight = (self.weight * np.exp(-alpha * y_temp * preds)) / Z

            self.modelList.append(best_model)
            print(f"T = {index + 1}, Training is Finished")

            if model_type == 'CART':
                dot = best_model.plot_tree()
                dot.render(f"../pic/CART/3CART{index}", format='png')

    def predict(self, X):
        res = np.zeros(X.shape[0])
        for index in range(self.T):
            model = self.modelList[index]
            alpha = self.alphaList[index]
            if self.model_type == 'SVM':
                res += np.sign(model.predict(X).reshape(res.shape)) * alpha
            elif self.model_type == 'KNN':
                res += np.array(model.predict(X, 7)) * alpha
            else:
                res += np.array(model.predict(X)) * alpha
        res = np.sign(res)
        res = np.where(res == -1, 0, res)
        return res


if __name__ == "__main__":
    # train_data = pd.read_csv('../data/preTrain/train.csv')
    # test_data = pd.read_csv('../data/preTrain/test.csv')
    train_data = pd.read_csv('../data/preTrain/train_pca95.csv')
    test_data = pd.read_csv('../data/preTrain/test_pca95.csv')

    y_train = train_data['Survived']

    X_train = train_data.drop(['Survived'], axis=1)
    X_test = test_data

    X_train = X_train.drop(['LastName'], axis=1)
    X_test = X_test.drop(['LastName'], axis=1)
    X_train = X_train.drop(['PassengerId'], axis=1)

    X_test = X_test.drop(['PassengerId'], axis=1)

    adaboost = AdaBoost(T=3)
    adaboost.fit(X_train, y_train, 'SVM')

    y_train_preds = adaboost.predict(X_train)
    y_train_temp = np.array(y_train)
    preds = np.array(y_train_preds)
    print(f"Error on train data = {np.sum(y_train_temp != preds) / y_train_temp.shape[0]}")

    y_test = adaboost.predict(X_test)

    result = pd.read_csv('../data/titanic/gender_submission.csv')
    result['Survived'] = y_test.astype(int)
    result.to_csv('../data/submit/submission_pca.csv', index=False)

    std = pd.read_csv('../data/answer/titanic/ground_truth.csv')
    submit = pd.read_csv('../data/submit/submission_pca.csv')

    total = 419
    diff_count = (std['Survived'] != submit['Survived']).sum()

    print(f"Acc = {1 - diff_count / total}")
