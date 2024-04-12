import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def Part1(data):
    '''
        编写程序，对类 1 和类 2 中的三个特征𝑥𝑖分别求解最大似然估计的均值𝜇̂和方差𝜎̂2。
        当前要处理的工作即将类中三个特征各自看作独立，求其均值和方差
        即此时 μ和 Σ 均不知
    '''
    # 手动实现计算均值和方差
    data = data.to_numpy()
    n, m = data.shape
    miu = np.zeros(m)
    sigma2 = np.zeros(m)
    for x in data:
        miu += x / n
    for x in data:
        sigma2 += (x - miu) ** 2 / n
    print(f'当对三个特征分别手算求最大似然估计时:\nmiu=\n{miu}\nsigma2=\n{sigma2}')

    # 也可以采用pandas中的包来计算均值和方差
    miu = np.mean(data, axis=0)
    sigma2 = np.var(data, axis=0, ddof=0)
    print(f'当对三个特征分别利用numpy求最大似然估计时:\nmiu=\n{miu}\nsigma2=\n{sigma2}\n')


def DealForMatrix(data):
    data = data.to_numpy()
    n, m = data.shape
    miu = np.zeros(m)
    sigma2 = np.zeros([m, m])
    for x in data:
        miu += x
    miu = miu / n
    for x in data:
        delta = (x - miu).reshape(-1, 1)
        sigma2 += np.dot(delta, delta.T)
    sigma2 = sigma2 / n
    print(f'当手算求最大似然估计时:\nmiu=\n{miu}\nsigma2=\n{sigma2}')

    # 也可以采用pandas中的包来计算均值和方差
    miu = np.mean(data, axis=0)
    sigma2 = np.cov(data, rowvar=False, ddof=0)
    print(f'当利用numpy求最大似然估计时:\nmiu=\n{miu}\nsigma2=\n{sigma2}\n')


def Part2(data):
    n, m = data.shape
    for i in range(m):
        for j in range(i + 1, m):
            print(f'采用x{i + 1}和x{j + 1}进行计算所得:')
            tmp = pd.concat([data.iloc[:, i], data.iloc[:, j]], axis=1).copy()
            DealForMatrix(tmp)


def Part3(data):
    n, m = data.shape
    DealForMatrix(data)

def Part4(data):
    '''
    绝大多数代码在Part1中已经呈现，因此此处直接使用numpy实现
    '''
    data = data.to_numpy()
    miu = np.mean(data,axis=0)
    sigma2 = np.var(data,axis=0,ddof=0)
    cov = np.diag(sigma2)
    print(f'当利用numpy求最大似然估计时:\nmiu=\n{miu}\ncov=\n{cov}\n')


if __name__ == '__main__':
    df = pd.read_excel('../../data/exp2-2.xlsx')
    y = df['y']
    X = df.drop('y', axis=1)
    X1 = X[y == 1]
    X2 = X[y == 0]

    print('第一部分:')
    print("For Class 1:")
    Part1(X1)
    print('--------------------------------------------')
    print("第二部分:")
    print("For Class 1:")
    Part2(X1)
    print('--------------------------------------------')
    print("第三部分:")
    print("For Class 1:")
    Part3(X1)
    print('--------------------------------------------')
    print("第四部分:")
    print("For Class 1:")
    Part4(X1)
