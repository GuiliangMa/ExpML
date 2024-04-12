import numpy as np
from DataProcessor import PrepareForTraining, Sigmoid


class LinearClassifier:
    def __init__(self, alpha=0.01, iterations=1000):
        self.alpha = alpha
        self.iterations = iterations
        self.weights = None

    def gradientDescent(self, X, y):
        n = X.shape[0]
        pred = Sigmoid(np.dot(X, self.weights))
        delta = pred - y
        # self.weights -= self.alpha * (1 / n) * np.dot(X.T, delta) #交叉熵 $L(w) = -yln[Sigmoid(Xw)]-(1-y)ln[1-Sigmoid(Xw)]$
        self.weights -= self.alpha * (1 / n) * np.dot(X.T, delta * pred * (1 - pred)) # $L(w)=\frac{1}{2n}[Sigmoid(Xw)-y]^2$

    def fit(self, X, y):
        X = PrepareForTraining(X)
        n, m = X.shape
        self.weights = np.zeros([m, 1])
        for _ in range(self.iterations):
            self.gradientDescent(X, y)
        return self.weights

    def predict(self, X):
        X = PrepareForTraining(X)
        return Sigmoid(np.dot(X, self.weights))>=0.5