import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def fillNan(column, type=0, defaultValue=0):
    if type not in [0, 1, 2, 3]:
        raise ValueError("type must be 0, 1, 2, or 3.")
    if type == 0:
        column.fillna(column.mode()[0], inplace=True)
    if type == 1:
        column.fillna(column.median(), inplace=True)
    if type == 2:
        column.fillna(column.mean(), inplace=True)
    if type == 3:
        column.fillna(defaultValue, inplace=True)
    return column


def dealString(column, mapping):
    values = column.values.tolist()
    for value in values:
        if value not in mapping:
            raise ValueError("Value " + value + " is not in dictionary!")
    column = column.map(mapping)
    return column


df = pd.read_csv('../data/train.csv')

# 删除若干列,从训练集和测试集可以初步看出
# loan_id和user_id是基本不重复的，因此可以删去。当前数据下policy_code下均为1，也可删去。
df = df.drop(['loan_id'], axis=1)
df = df.drop(['user_id'], axis=1)
df = df.drop(['policy_code'], axis=1)

# 默认均采用众数来填充所有缺失值
for columnName in df.columns:
    df[columnName] = fillNan(df[columnName])

# 先处理字符串变为数值型
## 第一步处理 class属性，其对应的Mapping如下
class_Mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
df['class'] = dealString(df['class'], class_Mapping)

## 提取检测值
y = df['isDefault']
X = df.drop(['isDefault'], axis=1)
X.to_csv('../process/Processed_TrainData.csv', index=False)
