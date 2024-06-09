import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

loss1024 = []
with open('../result/batch64hidden128_10.txt', 'r') as f:
    for line in f:
        loss1024.append(float(line.strip()))
print(loss1024)

loss128 = []
with open('../result/test1_10.txt', 'r') as f:
    for line in f:
        loss128.append(float(line.strip()))
print(loss128)

loss64 = []
with open('../result/test2_10.txt', 'r') as f:
    for line in f:
        loss64.append(float(line.strip()))
print(loss64)

loss32 = []
with open('../result/test3_10.txt', 'r') as f:
    for line in f:
        loss32.append(float(line.strip()))
print(loss32)


epochs = np.arange(0, len(loss1024))
# 绘制损失函数折线图
plt.figure(figsize=(20, 8))
plt.plot(loss1024, label='origin')
plt.plot(loss128, label='test1')
plt.plot(loss64, label='test2')
plt.plot(loss32, label='test3')
plt.xlabel(f'Epochs=10')
plt.ylabel('Loss')
plt.title('Loss Function Over Time')
plt.legend()
plt.grid(True)
# 保存图像
plt.xticks(epochs)
plt.savefig(f'../pic/loss_with_diff_structure10.png')
# 显示图像
plt.show()


# list = [-2,-3,-4,-5,-6,-7,-4,-5,-6,-7,-8,-4,-5,-6,-7,-8,-9,-4,-5,-6]
# epochs = np.arange(0, len(list))
# plt.plot(list, label='1ex')
# plt.xlabel(f'epochs')
# plt.ylabel('1e x')
# plt.title('Learning Rate Change')
# plt.legend()
# plt.grid(True)
# plt.xticks(epochs)
# plt.axis([-1, 19, -10, -1])
# plt.savefig(f'../pic/learningRateChange.png')
# plt.show()