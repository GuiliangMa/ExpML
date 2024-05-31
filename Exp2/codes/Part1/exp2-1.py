import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from LinearClassifier import LinearClassifier
from DataProcessor import PrepareForTraining, LinearBoundary

data = scipy.io.loadmat('../../data/e2data1.mat')
X = pd.DataFrame(data['X'], columns=['x1', 'x2'])
y = pd.DataFrame(data['y'], columns=['y'])

X_true = X[y['y'] == 1]
X_false = X[y['y'] == 0]

plt.plot(X_true['x1'], X_true['x2'], 'r.', markersize=12,label='True')
plt.plot(X_false['x1'], X_false['x2'], 'b.', markersize=12,label='False')

LC = LinearClassifier(alpha=0.1, iterations=10000)
weight = LC.fit(X, y)

# 以下内容便于图像展示

x_min, x_max = X['x1'].min() - 0.2, X['x1'].max() + 0.2
y_min, y_max = X['x2'].min() - 0.2, X['x2'].max() + 0.2
x = np.linspace(x_min, x_max, 100)
liny = LinearBoundary(weight, x)
plt.plot(x, liny, 'k')

custom_cmap = ListedColormap(['#9898ff',  # 浅红
                              '#FFC0CB',])

x0, x1 = np.meshgrid(np.linspace(x_min, x_max, 1000).reshape(-1, 1),
                     np.linspace(y_min, y_max, 1000).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]
X_new = pd.DataFrame(X_new)
Y_new = LC.predict(X_new)
z = Y_new.reshape(x0.shape)
plt.contourf(x0, x1, z, cmap=custom_cmap)

plt.axis([x_min, x_max, y_min, y_max])
plt.legend(loc='lower left')
plt.show()
