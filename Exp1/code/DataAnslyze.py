# 在进行贝叶斯分类之前重点是对数据进行预处理操作。
# 如，缺失值的填充、将文字表述转为数值型、日期处理格式（处理成“年-月-日”三列属性或者以最早时间为基准计算差值）、无关属性的删除、多列数据融合等方面。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check(column_name):
    values_counts = df[column_name].value_counts()
    print(values_counts)

    has_null = df[column_name].isnull().any()
    print(has_null)

    unique_values = df[column_name].drop_duplicates().tolist()
    print(len(unique_values))
    if len(unique_values) < 30:
        print(unique_values)


def checknull(columns):
    nulllist = []
    for column in columns:
        has_null = df[column].isnull().any()
        if has_null == 1:
            nulllist.append(column)

    print(len(nulllist))
    print(nulllist)


def plot_bar(columnName):
    df[columnName].value_counts().sort_index().plot(kind='bar')
    plt.xlabel('values')
    plt.ylabel('Frequency')
    plt.show()


train_data_path = '../data/train.csv'
df = pd.read_csv(train_data_path)
columns = df.columns.tolist()
# check('early_return_amount_3mon')
# checknull(columns)
plot_bar('region')


