import os
import pickle

import numpy as np

import numpy as np


def sigmoid(x):
    # print(x)
    x = np.clip(x, -20, 20)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


import numpy as np


def softmax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - x_max)
    sum_e_x = e_x.sum(axis=1, keepdims=True)
    return e_x / sum_e_x


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def cross_entropy(y_true, y_pred):
    # Ensure numerical stability with a small constant epsilon
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred + 1e-9)+(1-y_true+1e-9)*np.log(1-y_pred+1e-9)) / y_true.shape[0]


def cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    derivative = -y_true / y_pred + (1 - y_true) / (1 - y_pred)
    return derivative / y_pred.shape[0]


def cross_entropy_der_softmax(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return (y_pred - y_true) / y_pred.shape[0]


class NeuralNetwork:
    def __init__(self, layers, activations, loss='mse', l2_lambda=0.01):
        if len(activations) != len(layers) - 1:
            raise ValueError("Number of activations must be equal to number of layers - 1")

        self.layers = layers
        self.activations_info = activations
        self.loss = loss
        self.weights = []
        self.biases = []
        self.activation_funcs = []
        self.activation_derivs = []
        self.l2_lambda = l2_lambda
        self.d_weights = [None] * (len(self.layers) - 1)  # Initializing gradients of weights
        self.d_biases = [None] * (len(self.layers) - 1)  # Initializing gradients of biases

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, layers[i + 1])))

            activation, activation_derivative = self._get_activation(activations[i])
            self.activation_funcs.append(activation)
            self.activation_derivs.append(activation_derivative)

        self.loss_func, self.loss_derivative = self._get_loss_function(loss, activations[-1])

    def _get_activation(self, activation_name):
        activations = {
            'sigmoid': (sigmoid, sigmoid_derivative),
            'relu': (relu, relu_derivative),
            'tanh': (tanh, tanh_derivative),
            'softmax': (softmax, lambda x: 1)  # Dummy derivative for softmax at output
        }
        return activations.get(activation_name, (None, None))

    def _get_loss_function(self, loss_name, activation_name):
        if activation_name == 'softmax':
            loss_functions = {
                'mse': (mse, mse_derivative),
                'cross_entropy': (cross_entropy, cross_entropy_der_softmax)
            }
        else:
            loss_functions = {
                'mse': (mse, mse_derivative),
                'cross_entropy': (cross_entropy, cross_entropy_derivative)
            }
        return loss_functions.get(loss_name, (None, None))

    def forward(self, x):
        self.activations = [x]
        self.linearcombination = [x]
        for w, b, activation_func in zip(self.weights, self.biases, self.activation_funcs):
            z = np.dot(self.activations[-1], w) + b
            a = activation_func(z)
            self.linearcombination.append(z)
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, y_true):
        error = self.loss_derivative(y_true, self.activations[-1])
        for i in reversed(range(len(self.weights))):
            error *= self.activation_derivs[i](self.linearcombination[i + 1])
            self.d_weights[i] = np.dot(self.activations[i].T, error)
            self.d_biases[i] = np.sum(error, axis=0, keepdims=True)
            error = np.dot(error, self.weights[i].T)

    def update_weights(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * (self.d_weights[i] + self.l2_lambda * self.weights[i])
            # self.weights[i] -= learning_rate * (self.d_weights[i])
            self.biases[i] -= learning_rate * self.d_biases[i]

    def calculate_accuracy(self, y_true, y_pred):
        # 将热编码的真实标签转换为类别索引
        y_true_labels = np.argmax(y_true, axis=1)
        # 计算准确率
        correct_predictions = np.sum(y_true_labels == y_pred)
        accuracy = 100 * correct_predictions / len(y_true_labels)
        return accuracy

    def train(self, X, y, epochs, batch_size, learning_rate):
        X_temp = X[:49000]
        y_temp = y[:49000]
        X_test = X[49000:]
        y_test = y[49000:]
        for epoch in range(epochs):
            indices = np.arange(X_temp.shape[0])
            np.random.shuffle(indices)
            X_train = X_temp[indices]
            y_train = y_temp[indices]
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                self.forward(X_batch)
                self.backward(y_batch)
                self.update_weights(learning_rate)
                if ((i / batch_size) + 1) % 100 == 0:
                    print(
                        f"Epoch {epoch + 1}, Batch {int((i / batch_size) + 1)}, Loss = {self.loss_func(y_test, self.forward(X_test))}")
            print(f"Epoch {epoch + 1}, Loss: {self.loss_func(y, self.forward(X))}")
            y_pred = self.predict(X_test)
            test_accuracy = self.calculate_accuracy(y_test, y_pred)
            print(f"Epoch {epoch + 1}, Test Accuracy: {test_accuracy:.2f}%")

        y_pred = self.predict(X_test)
        test_accuracy = self.calculate_accuracy(y_test, y_pred)
        print(f"Test Accuracy: {test_accuracy:.2f}%")

    def predict(self, x):
        # 进行前向传播得到预测结果
        predictions = self.forward(x)
        return np.argmax(predictions, axis=1)


def load_cifar10_batch(file_path):
    """ Load a single batch of CIFAR-10 data. """
    with open(file_path, 'rb') as file:
        datadict = pickle.load(file, encoding="latin1")  # 读取全部内容
        data = np.reshape(datadict['data'], (10000, 3072))
        labels = np.array(datadict['labels'])
        return data, labels


def load_all_batches(data_dir, batches):
    """ Load all CIFAR-10 data batches. """
    all_images = []
    all_labels = []
    for batch_name in batches:
        file_path = os.path.join(data_dir, batch_name)
        images, labels = load_cifar10_batch(file_path)
        all_images.append(images)
        all_labels.append(labels)
    return np.concatenate(all_images), np.concatenate(all_labels)


def to_one_hot(labels, num_classes):
    one_hot_labels = np.zeros((labels.size, num_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels


if __name__ == "__main__":
    data_dir = '../data/cifar-10-batches-py'
    batch_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    images, labels = load_all_batches(data_dir, batch_files)
    images = images.astype(np.float32) / 255.0
    labels = to_one_hot(labels, 10)

    # np.random.seed(42)
    nn = NeuralNetwork(layers=[3072, 1024,10],
                       activations=['relu', 'sigmoid'],
                       loss='cross_entropy')
    nn.train(images, labels, 30, 64, 0.005)
