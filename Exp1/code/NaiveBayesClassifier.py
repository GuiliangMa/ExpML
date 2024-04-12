import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class NaiveBayesClassifier:
    def __init__(self, alpha):
        self.alpha = alpha
        self.class_prior = {}
        self.cond_prob = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.features = [np.unique(X.iloc[:, col]) for col in range(X.shape[1])]
        self.X = X
        self.y = y
        total_count = len(y)
        for cls in self.classes:
            cls_count = np.sum(y == cls)
            self.class_prior[cls] = (cls_count + self.alpha) / (total_count + len(self.classes) * self.alpha)
            self.cond_prob[cls] = {}
            for i, feature in enumerate(self.features):
                self.cond_prob[cls][i] = {}
                for value in feature:
                    feature_count = np.sum((X.iloc[:, i] == value) & (y == cls))
                    self.cond_prob[cls][i][value] = (feature_count + self.alpha) / (
                                cls_count + len(feature) * self.alpha)

    def predict(self,X_test):
        predictions = []
        for x in X_test.values:
            probs ={}
            for cls in self.classes:
                probs[cls] = self.class_prior[cls]
                for i,value in enumerate(x):
                    if value in self.cond_prob[cls][i]:
                        probs[cls] *= self.cond_prob[cls][i][value]
                    else:
                        probs[cls] *= self.alpha / (np.sum(self.y == cls) + len(self.features[i]) * self.alpha)
            # print(max(probs, key=probs.get))
            predictions.append(max(probs, key=probs.get))
        return predictions