import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def KNN(X, k, t):
    res = np.zeros((t.shape[0],))
    for i, test in enumerate(t):
        dist = np.linalg.norm(X - test, axis=1)
        dist = np.sort(dist)
        if dist[k - 1] != 0:
            res[i] = k / (X.shape[0] * dist[k - 1] ** X.shape[1])
        else:
            res[i] = k / (X.shape[0] * (1e-5) ** X.shape[1])
    return res


def Part1(X, y):
    X = X['x1']
    X_train = X[y == 3]
    x_min = X_train.min()
    x_max = X_train.max()
    X_train = X_train.to_numpy().reshape(-1, 1)
    X_test = np.linspace(x_min, x_max, 1000, endpoint=True).reshape(-1, 1)
    y1_test = KNN(X_train, 1, X_test)
    y3_test = KNN(X_train, 3, X_test)
    y5_test = KNN(X_train, 5, X_test)

    plt.figure(figsize=(16, 9))
    plt.subplot(131)
    plt.plot(X_test, y1_test, 'r')
    plt.xlim([x_min, x_max])
    plt.title('k=1')

    plt.subplot(132)
    plt.plot(X_test, y3_test, 'b')
    plt.xlim([x_min, x_max])
    plt.title('k=3')

    plt.subplot(133)
    plt.plot(X_test, y5_test, 'g')
    plt.xlim([x_min, x_max])
    plt.title('k=5')

    plt.suptitle('Part1')
    plt.show()


def DrawPart2(x0, x1, z, k):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x0, x1, z, cmap='viridis')
    # 添加标签和标题
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('p')
    ax.set_title(f'k={k}, 3D')
    fig.patch.set_alpha(0)
    ax.grid(False)
    # 显示图形
    plt.show()

    plt.contourf(x0, x1, z, cmap='viridis')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'k={k}, 2D')
    plt.show()


def Part2(X, y):
    X = pd.concat([X['x1'], X['x2']], axis=1)
    X_train = X[y == 2].copy()
    x1_min = X_train['x1'].min()
    x1_max = X_train['x1'].max()
    x2_min = X_train['x2'].min()
    x2_max = X_train['x2'].max()
    x0, x1 = np.meshgrid(np.linspace(x1_min, x1_max, 100).reshape(-1, 1),
                         np.linspace(x2_min, x2_max, 100).reshape(-1, 1))
    X_test = np.c_[x0.ravel(), x1.ravel()]
    X_train = X_train.to_numpy()
    y1_test = KNN(X_train, 1, X_test)
    y3_test = KNN(X_train, 3, X_test)
    y5_test = KNN(X_train, 5, X_test)
    DrawPart2(x0, x1, y1_test.reshape(x0.shape), 1)
    DrawPart2(x0, x1, y3_test.reshape(x0.shape), 3)
    DrawPart2(x0, x1, y5_test.reshape(x0.shape), 5)


def Part3(X, y, X_test):
    result = np.zeros((3, X_test.shape[0]))
    for i in range(3):
        X_train = X[y == i + 1].copy()
        result[i] = KNN(X_train, 3, X_test)
    result = result.T
    result = pd.DataFrame(result, columns=['P(class1)', 'P(class2)', 'P(class3)'])
    print("KNN求得的概率密度")
    print(result)


df = pd.read_excel('../../data/exp2-3.xlsx')
df.to_csv('../../data/exp2-3.csv', index=False)
y = df['y']
X = df.drop('y', axis=1)
Xtest = [[-0.41, 0.82, 0.88],
         [0.14, 0.72, 4.1],
         [-0.81, 0.61, -0.38]]
Xtest = np.array(Xtest)
# Part1(X, y)
Part2(X, y)
# Part3(X, y, Xtest)
