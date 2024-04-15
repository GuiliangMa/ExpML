import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class NeuralNetwork:
    def __init__(self, layers, act_funs):
        self.layers = layers
        self.act_funs = act_funs
        self.params = {}
        self.init_weights()

    def init_weights(self):
        for i in range(1, len(self.layers)):
            self.params['W' + str(i)] = np.random.randn(self.layers[i], self.layers[i - 1])*0.1
            self.params['b' + str(i)] = np.zeros((self.layers[i], 1))

    def activate(self, x, activation):
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'relu':
            return np.maximum(0, x)
        else:
            raise Exception('Unsupported activation function')

    def activation_derivative(self, a, activation):
        if activation == 'sigmoid':
            return a * (1 - a)
        elif activation == 'tanh':
            return 1 - np.power(a, 2)
        elif activation == 'relu':
            return np.where(a <= 0, 0, 1)
        else:
            raise ValueError("Unsupported activation function.")

    def forward_propagation(self, X):
        cache = {'Y0': X}
        Y = X
        for i in range(1, len(self.layers)):
            print(self.act_funs[i - 1])
            E = np.dot(self.params['W' + str(i)], Y) + self.params['b' + str(i)]
            Y = self.activate(E, self.act_funs[i - 1])
            cache['E' + str(i)] = E
            cache['Y' + str(i)] = Y
        return Y, cache

    def compute_cost(self, Y, y):
        # 计算样本数量
        m = y.shape[1]
        # 损失函数选择 平均交叉熵
        cost = -np.sum(y * np.log(Y + 1e-8) + (1 - y) * np.log(1 - Y + 1e-8)) / m
        print(cost)
        return cost

    def back_propagation(self, y, cache):
        grad = {}
        m = y.shape[1]
        dY = -(np.divide(y, cache['Y' + str(len(self.layers) - 1)] + 1e-8) - np.divide(1 - y, 1 - cache[
            'Y' + str(len(self.layers) - 1)] + 1e-8))
        for i in reversed(range(1, len(self.layers))):
            dE = dY * self.activation_derivative(cache['E' + str(i)], self.act_funs[i - 1])
            dW = np.dot(dE, cache['Y' + str(i - 1)].T) / m
            db = np.sum(dE, axis=1, keepdims=True) / m
            if i > 1:
                dY = np.dot(self.params['W' + str(i)].T, dE)
            grad['dW' + str(i)] = dW
            grad['db' + str(i)] = db
        print(grad)
        return grad

    def update_weights(self, gradients, learning_rate):
        for i in range(1, len(self.layers)):
            self.params['W' + str(i)] -= learning_rate * gradients['dW' + str(i)]
            self.params['b' + str(i)] -= learning_rate * gradients['db' + str(i)]

    def train(self, X, y, learning_rate, batch_size, epochs):
        m = X.shape[1]
        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(m)
            X_shuffled = X[:, shuffled_indices]
            y_shuffled = y[:, shuffled_indices]
            cost = 0
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i + batch_size]
                y_batch = y_shuffled[:, i:i + batch_size]
                Y, cache = self.forward_propagation(X_batch)
                cost = self.compute_cost(Y, y_batch)
                grad = self.back_propagation(y_batch,cache)
                self.update_weights(grad, learning_rate)
            print("Cost after epoch {}: {}".format(epoch, cost))

    def predict(self, X):
        Y, _ = self.forward_propagation(X)
        return Y


if __name__ == '__main__':
    layers = [3072, 1024, 10]
    activations = ['relu', 'sigmoid']
    nn = NeuralNetwork(layers, activations)