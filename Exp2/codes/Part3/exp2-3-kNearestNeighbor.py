import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Exp2.codes.Part4.KNN import kNNClassifier
import matplotlib


def DealPart1(X_train, y_train):
    x_min = X_train.min()
    x_max = X_train.max()
    X_train = X_train.to_numpy().reshape(-1, 1)
    X_test = np.linspace(x_min, x_max, 1000, endpoint=True).reshape(-1, 1)
    knn = kNNClassifier()

    # 获得概率密度函数
    y1_density = knn.density(X_train, y_train, X_test, 1)
    y3_density = knn.density(X_train, y_train, X_test, 3)
    y5_density = knn.density(X_train, y_train, X_test, 5)

    y1_test = np.array([d[3] for d in y1_density]).reshape(-1, 1)
    y3_test = np.array([d[3] for d in y3_density]).reshape(-1, 1)
    y5_test = np.array([d[3] for d in y5_density]).reshape(-1, 1)

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
    # matplotlib.use('TkAgg')
    # 创建一个图形和两个子图（一个2D，一个3D）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 第一个子图为3D图
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(x0, x1, z, cmap='viridis')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('p')
    ax1.set_title(f'k={k}, 3D')
    ax1.patch.set_visible(False)
    ax1.grid(False)

    # 第二个子图为2D等高线图
    contour = ax2.contourf(x0, x1, z, cmap='viridis')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title(f'k={k}, 2D')
    # 调整布局
    plt.tight_layout()
    # 显示图形
    plt.show()


def DealPart2(X_train, y_train):
    x1_min = X_train['x1'].min()
    x1_max = X_train['x1'].max()
    x2_min = X_train['x2'].min()
    x2_max = X_train['x2'].max()
    x0, x1 = np.meshgrid(np.linspace(x1_min, x1_max, 100).reshape(-1, 1),
                         np.linspace(x2_min, x2_max, 100).reshape(-1, 1))
    X_test = np.c_[x0.ravel(), x1.ravel()]
    X_train = X_train.to_numpy()
    knn = kNNClassifier()
    y1_post = knn.density(X_train, y_train, X_test, 1)

    y3_post = knn.density(X_train, y_train, X_test, 3)

    y5_post = knn.density(X_train, y_train, X_test, 5)

    # 获得概率密度函数
    y1_density = knn.density(X_train, y_train, X_test, 1)
    y3_density = knn.density(X_train, y_train, X_test, 3)
    y5_density = knn.density(X_train, y_train, X_test, 5)

    y1_test = np.array([d[2] for d in y1_density]).reshape(-1, 1)
    y3_test = np.array([d[2] for d in y3_density]).reshape(-1, 1)
    y5_test = np.array([d[2] for d in y5_density]).reshape(-1, 1)

    DrawPart2(x0, x1, y1_test.reshape(x0.shape), 1)
    DrawPart2(x0, x1, y3_test.reshape(x0.shape), 3)
    DrawPart2(x0, x1, y5_test.reshape(x0.shape), 5)

def UnderStanding1(X, y):
    '''
    这是我对题面的第一种理解，仅考虑题面所提及的类3（题1），和类2（题2）。
    仅使用这一个类，利用knn求得这一类得类条件概率密度（与其他类无关）
    '''

    # 以下为第一小问：
    X1 = X['x1'].copy()
    X_train = X1[y == 3].copy().reset_index(drop=True)
    y_train = y[y == 3].copy().reset_index(drop=True)
    DealPart1(X_train, y_train)

    # 以下为第二部分
    X2 = pd.concat([X['x1'], X['x2']], axis=1)
    X_train = X2[y == 2].copy().reset_index(drop=True)
    y_train = y[y == 2].copy().reset_index(drop=True)
    DealPart2(X_train, y_train)


def UnderStanding2(X,y):
    '''
        这是我对题面的第二种理解，使用多个类，但是只展示类3（题1），类2（题2）的概率密度
    '''

    # 以下为第一小问：
    X1 = X['x1'].copy()
    X_train = X1.copy().reset_index(drop=True)
    y_train = y.copy().reset_index(drop=True)
    DealPart1(X_train, y_train)

    # 以下为第二部分
    X2 = pd.concat([X['x1'], X['x2']], axis=1)
    X_train = X2.copy().reset_index(drop=True)
    y_train = y.copy().reset_index(drop=True)
    DealPart2(X_train, y_train)


df = pd.read_excel('../../data/exp2-3.xlsx')
df.to_csv('../../data/exp2-3.csv', index=False)
y = df['y']
X = df.drop('y', axis=1)
Xtest = [[-0.41, 0.82, 0.88],
         [0.14, 0.72, 4.1],
         [-0.81, 0.61, -0.38]]
Xtest = np.array(Xtest)
# UnderStanding1(X, y)
UnderStanding2(X,y)
knn = kNNClassifier()
for k in range(5):
    density = knn.density(X, y, Xtest, k + 1)
    print(f'k={k + 1} 时的概率密度')
    for index in range(len(density)):
        print(density[index])
