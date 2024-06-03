import cvxopt
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap


def LinearBoundary(weights, b, x):
    y = (-weights[0] / weights[1]) * x - b / weights[1]
    return y


def rbf_kernel(X1, X2, gamma=0.1):
    if X1.ndim == 1 and X2.ndim == 1:
        return np.exp(-gamma * np.linalg.norm(X1 - X2) ** 2)
    elif X1.ndim > 1 and X2.ndim == 1:
        return np.exp(-gamma * np.linalg.norm(X1 - X2, axis=1) ** 2)
    elif X1.ndim == 1 and X2.ndim > 1:
        return np.exp(-gamma * np.linalg.norm(X1 - X2, axis=1) ** 2)
    elif X1.ndim > 1 and X2.ndim > 1:
        return np.exp(-gamma * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2) ** 2)


class SVM:
    """
    基于对偶问题分析的 SVM
    """

    def __init__(self, C=1.0):
        self.weight = None
        self.bias = None
        self.Lambda = None
        self.trained = False
        # 用于限制约束拉格朗日因子 lambda 的范围 0<=lambda<=C
        self.C = C
        self.stategy = None

    def compute_error(self, i, X, y):
        """计算第i个数据点的预测误差"""
        f_x = np.dot(self.Lambda.T * y.T, X @ X[i].T) + self.bias
        return f_x - y[i]

    def select_j(self, i, E_i, X, y):
        """选择第二个乘子的索引"""
        valid_indices = [j for j in range(X.shape[0]) if j != i]
        j = max(valid_indices, key=lambda x: abs(E_i - self.compute_error(x, X, y)))
        return j

    def choose_lambda_pair(self, i, X, y, tol):
        """选择一对拉格朗日乘子进行优化，以i为起始乘子"""
        num_changed = 0
        E_i = self.compute_error(i, X, y)
        if ((y[i] * E_i < -tol and self.Lambda[i] < self.C) or
                (y[i] * E_i > tol and self.Lambda[i] > 0)):
            # 选择第二个乘子
            j = self.select_j(i, E_i, X, y)
            E_j = self.compute_error(j, X, y)

            # 保存旧的alphas
            Lambda_i_old = self.Lambda[i].copy()
            Lambda_j_old = self.Lambda[j].copy()

            # 计算剪辑边界
            if y[i] != y[j]:
                L = max(0, self.Lambda[j] - self.Lambda[i])
                H = min(self.C, self.C + self.Lambda[j] - self.Lambda[i])
            else:
                L = max(0, self.Lambda[i] + self.Lambda[j] - self.C)
                H = min(self.C, self.Lambda[i] + self.Lambda[j])

            if L == H:
                return 0

            # 计算eta
            eta = 2 * np.dot(X[i], X[j].T) - np.dot(X[i], X[i].T) - np.dot(X[j], X[j].T)
            if eta >= 0:
                return 0

            # 更新alpha_j
            self.Lambda[j] -= y[j] * (E_i - E_j) / eta
            self.Lambda[j] = np.clip(self.Lambda[j], L, H)

            if abs(self.Lambda[j] - Lambda_j_old) < 1e-5:
                return 0

            # 更新alpha_i
            self.Lambda[i] += y[i] * y[j] * (Lambda_j_old - self.Lambda[j])

            # 更新阈值b
            b1 = self.bias - E_i - y[i] * (self.Lambda[i] - Lambda_i_old) * np.dot(X[i], X[i].T) - y[j] * (
                    self.Lambda[j] - Lambda_j_old) * np.dot(X[i], X[j].T)
            b2 = self.bias - E_j - y[i] * (self.Lambda[i] - Lambda_i_old) * np.dot(X[i], X[j].T) - y[j] * (
                    self.Lambda[j] - Lambda_j_old) * np.dot(X[j], X[j].T)
            if 0 < self.Lambda[i] < self.C:
                self.bias = b1
            elif 0 < self.Lambda[j] < self.C:
                self.bias = b2
            else:
                self.bias = (b1 + b2) / 2

            num_changed = 1
        return num_changed

    def fitBySMO(self, X, label, C=1.0, margin='soft', kernel='none', gamma=10, tolerance=1e-6):
        self.trained = True
        self.kernel = kernel
        self.gamma = gamma
        self.stategy = 'smo'
        X = X.values
        n_samples, n_features = X.shape
        y = label.astype(float)
        y[label == 0] = -1
        y = np.array(y).reshape(-1, 1)
        self.Lambda = np.zeros((n_samples, 1))
        self.bias = 0

        def compute_kernel(x1, x2):
            if self.kernel == 'none':
                return np.dot(x1, x2.T)
            elif self.kernel == 'rbf':
                return rbf_kernel(x1,x2,gamma)
            else:
                return np.dot(x1, x2.T)  # 默认使用线性核

        def compute_error(i):
            fx_i = np.dot((self.Lambda * y).T, compute_kernel(X, X[i])) + self.bias
            E_i = fx_i - y[i]
            return E_i

        def choose_alpha_pair(i):
            E_i = compute_error(i)
            if (y[i] * E_i < -tolerance and self.Lambda[i] < C) or (y[i] * E_i > tolerance and self.Lambda[i] > 0):
                j = np.random.choice(list(set(range(n_samples)) - {i}))
                E_j = compute_error(j)
                return i, j, E_i, E_j
            return i, None, E_i, None

        def update_alpha_pair(i, j, E_i, E_j):
            alpha_i_old = self.Lambda[i].copy()
            alpha_j_old = self.Lambda[j].copy()

            if y[i] != y[j]:
                L = max(0, alpha_j_old - alpha_i_old)
                H = min(C, C + alpha_j_old - alpha_i_old)
            else:
                L = max(0, alpha_i_old + alpha_j_old - C)
                H = min(C, alpha_i_old + alpha_j_old)

            if L == H:
                return False

            eta = 2 * compute_kernel(X[i], X[j]) - compute_kernel(X[i], X[i]) - compute_kernel(X[j], X[j])
            if eta >= 0:
                return False

            self.Lambda[j] -= y[j] * (E_i - E_j) / eta
            self.Lambda[j] = max(L, min(H, self.Lambda[j]))

            self.Lambda[i] += y[i] * y[j] * (alpha_j_old - self.Lambda[j])

            b1 = self.bias - E_i - y[i] * (self.Lambda[i] - alpha_i_old) * compute_kernel(X[i], X[i]) \
                 - y[j] * (self.Lambda[j] - alpha_j_old) * compute_kernel(X[i], X[j])
            b2 = self.bias - E_j - y[i] * (self.Lambda[i] - alpha_i_old) * compute_kernel(X[i], X[j]) \
                 - y[j] * (self.Lambda[j] - alpha_j_old) * compute_kernel(X[j], X[j])

            self.bias = (b1 + b2) / 2
            return True

        num_changed = 0
        examine_all = True
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(n_samples):
                    _, j, E_i, E_j = choose_alpha_pair(i)
                    if j is not None:
                        if update_alpha_pair(i, j, E_i, E_j):
                            num_changed += 1
            else:
                for i in range(n_samples):
                    if 0 < self.Lambda[i] < C:
                        _, j, E_i, E_j = choose_alpha_pair(i)
                        if j is not None:
                            if update_alpha_pair(i, j, E_i, E_j):
                                num_changed += 1
            examine_all = (examine_all == False)

        sv_X = []
        sv_y = []
        sv_lambda = []
        for index, lambda_i in enumerate(self.Lambda):
            print(lambda_i)
            if lambda_i > 1e-5:
                sv_lambda.append(lambda_i)
                sv_X.append(X[index])
                sv_y.append(y[index])
        self.Lambda = np.array(sv_lambda)
        self.sv_X = np.array(sv_X)
        self.sv_y = np.array(sv_y)
        # print(self.sv_X)
        return self.sv_X

    def fitByStd(self, X, label, margin='soft', kernel='none', gamma=10):
        self.stategy = 'std'
        self.kernel = kernel
        self.gamma = gamma
        self.trained = True
        X = X.values
        n_samples, n_features = X.shape
        # print(n_samples, n_features)
        y = label.astype(float)
        y[label == 0] = -1
        y = np.array(y).reshape(-1, 1)
        """
        调用cvxopt.solvers.qp()需要将原对偶问题转换为如下的标准形式
        min 1/2 x^T P x + q^T x
        s.t. G x <= h
             A x = b
        与之对应的即：  x = lambda,P = D,q= -1
                        G = -1, h = 0
                        A = y^T , b = 0
        """
        if kernel == 'none':
            K = np.dot(X, X.T)
        if kernel == 'rbf':
            K = rbf_kernel(X, X, gamma)
        P = np.dot(y, y.T) * K
        q = np.ones(n_samples) * -1
        A = y.reshape(1, -1)
        b = np.zeros(1)
        if margin == 'hard':
            G = -np.eye(n_samples)
            h = np.zeros(n_samples)
        if margin == 'soft':
            G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
            h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C))

        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.Lambda = np.ravel(solution['x'])
        # print(self.Lambda)
        sv = self.Lambda > 1e-5
        index = np.arange(len(self.Lambda))[sv]
        self.Lambda = self.Lambda[sv]
        sv_X = X[sv]
        sv_y = y[sv]
        self.sv_X = sv_X
        self.sv_y = sv_y

        temp_y = np.zeros(self.sv_X.shape[0]).reshape(-1, 1)
        for i in range(len(self.Lambda)):
            if self.kernel == 'none':
                temp_y += self.Lambda[i] * sv_y[i] * np.dot(sv_X, sv_X[i].T).reshape(-1, 1)
            if self.kernel == 'rbf':
                temp_y += self.Lambda[i] * sv_y[i] * rbf_kernel(sv_X, sv_X[i].T, gamma).reshape(-1, 1)
        self.bias = np.mean(sv_y - temp_y)
        print(self.bias)
        return self.sv_X

    def fitByGradDescent(self, X, label, margin='soft', iters=10000, learning_rate=0.00001, kernel='none'):
        self.kernel = kernel
        self.trained = True
        X = X.values
        X = np.array(X)
        n_samples, n_features = X.shape
        y = label.astype(float)
        y[label == 0] = -1
        y = np.array(y).reshape(-1, 1)
        if margin == 'hard':
            self.C = 10000

        if kernel != 'none':
            raise ValueError("Grad Descent only support kernel 'none'")

        self.weight = np.zeros(n_features)
        self.bias = 0
        for _ in range(iters):
            for index, x_i in enumerate(X):
                condition = (y[index] * (np.dot(x_i, self.weight.reshape(-1, 1)) + self.bias)) >= 1
                if condition:
                    self.weight -= learning_rate * (self.weight)
                else:
                    self.weight -= learning_rate * (self.weight - self.C * x_i * y[index])
                    self.bias -= -learning_rate * self.C * y[index]

        self.stategy = 'grad'
        sv_X = []
        sv_y = []
        self.weight = self.weight.reshape(-1, 1)
        for index, X_i in enumerate(X):
            if y[index] * (np.dot(X_i, self.weight) + self.bias) - 1 < 1e-2:
                sv_X.append(X_i)
                sv_y.append(y[index])
        self.sv_X = np.array(sv_X)
        self.sv_y = np.array(sv_y)
        return self.sv_X

    def fit(self, X, y, strategy='std', margin='soft',
            iters=10000, learning_rate=0.0001,
            kernel='none', gamma=10,
            tolerance=1e-6):
        if margin != 'soft' and margin != 'hard':
            raise ValueError(" Parameter margin must be 'soft' or 'hard' ")

        if strategy != 'std' and strategy != 'grad' and strategy != 'smo':
            raise ValueError(" Parameter strategy must be 'std' or 'grad' or 'smo'")

        if kernel != 'none' and kernel != 'rbf':
            raise ValueError(" Parameter kernel must be 'none' or 'rbf' ")

        if strategy == 'std':
            return self.fitByStd(X, y, margin, kernel, gamma)

        if strategy == 'grad':
            return self.fitByGradDescent(X, y, margin, iters, learning_rate)

        if strategy == 'smo':
            if margin == 'hard':
                self.C = 10000
            return self.fitBySMO(X, y, self.C, margin, kernel, gamma, tolerance)

    def predict(self, X):
        if self.stategy == "grad":
            y = np.dot(X, self.weight) + self.bias
            return y
        else:
            y = np.zeros(X.shape[0]).reshape(-1, 1)
            for i in range(len(self.Lambda)):
                if self.kernel == 'none':
                    y += self.Lambda[i] * self.sv_y[i] * (np.dot(X, self.sv_X[i].T).reshape(-1, 1))
                if self.kernel == 'rbf':
                    y += self.Lambda[i] * self.sv_y[i] * rbf_kernel(X, self.sv_X[i].T, self.gamma).reshape(-1, 1)
            y += self.bias
            return y


if __name__ == "__main__":
    data = scipy.io.loadmat('../data/ex5data1.mat')
    X = pd.DataFrame(data['X'], columns=['x1', 'x2'])
    y = pd.DataFrame(data['y'], columns=['y'])
    X_true = X[y['y'] == 1]
    X_false = X[y['y'] == 0]

    # plt.plot(X_true['x1'], X_true['x2'], 'r.', markersize=12, label='True')
    # plt.plot(X_false['x1'], X_false['x2'], 'b.', markersize=12, label='False')
    # plt.title("Data Distribution Chart")
    # plt.legend()
    # plt.show()

    model = SVM(C=1)
    sv_X = model.fit(X, y, 'smo', margin='hard',kernel='rbf')
    # exit(0)

    x_min, x_max = X['x1'].min(), X['x1'].max()
    y_min, y_max = X['x2'].min(), X['x2'].max()
    x_thresold = (x_max - x_min) / 20
    y_thresold = (y_max - y_min) / 20
    x_min -= x_thresold
    x_max += x_thresold
    y_min -= y_thresold
    y_max += y_thresold
    custom_cmap = ListedColormap(['#0000ff',  # 浅红
                                  '#000000',
                                  '#ff0000'])

    custom_cmap2 = ListedColormap(['#9898ff',  # 浅红
                                   '#FFC0CB', ])

    plt.plot(sv_X[:, 0], sv_X[:, 1], 'k.', markersize=20, label='Support Vector')
    plt.plot(X_true['x1'], X_true['x2'], 'r.', markersize=12, label='True')
    plt.plot(X_false['x1'], X_false['x2'], 'b.', markersize=12, label='False')
    x0, x1 = np.meshgrid(np.linspace(x_min, x_max, 1000).reshape(-1, 1),
                         np.linspace(y_min, y_max, 1000).reshape(-1, 1))
    X_new = np.c_[x0.ravel(), x1.ravel()]
    Y_new = model.predict(X_new)
    z = Y_new.reshape(x0.shape)
    z_label = np.where(z > 0, 1, -1)

    plt.contourf(x0, x1, z_label, cmap=custom_cmap2)
    plt.contour(x0, x1, z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], cmap=custom_cmap)
    plt.axis([x_min, x_max, y_min, y_max])
    plt.show()
