import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from SVM import LinearBoundary,SVM

if __name__ == "__main__":
    data = scipy.io.loadmat('../data/ex5data1.mat')
    X = pd.DataFrame(data['X'], columns=['x1', 'x2'])
    y = pd.DataFrame(data['y'], columns=['y'])
    X_true = X[y['y'] == 1]
    X_false = X[y['y'] == 0]

    plt.plot(X_true['x1'], X_true['x2'], 'r.', markersize=12, label='True')
    plt.plot(X_false['x1'], X_false['x2'], 'b.', markersize=12, label='False')
    plt.title("Data Distribution Chart")
    plt.legend()
    plt.show()

    model = SVM(C=1.0)
    w, b, sv_X = model.fit(X, y, 'std')

    x_min, x_max = X['x1'].min() - 0.2, X['x1'].max() + 0.2
    y_min, y_max = X['x2'].min() - 0.2, X['x2'].max() + 0.2
    x = np.linspace(x_min, x_max, 100)
    midLine = LinearBoundary(w, b, x)
    falseLine = LinearBoundary(w, b + 1, x)
    trueLine = LinearBoundary(w, b - 1, x)

    plt.plot(sv_X[:, 0], sv_X[:, 1], 'k.', markersize=20, label='Support Vector')
    plt.plot(X_true['x1'], X_true['x2'], 'r.', markersize=12, label='True')
    plt.plot(X_false['x1'], X_false['x2'], 'b.', markersize=12, label='False')
    plt.plot(x, midLine, 'k')
    plt.plot(x, trueLine, 'r')
    plt.plot(x, falseLine, 'b')
    plt.legend(loc='lower left')
    plt.show()