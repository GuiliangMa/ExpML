# 在进行贝叶斯分类之前重点是对数据进行预处理操作。
# 如，缺失值的填充、将文字表述转为数值型、日期处理格式（处理成“年-月-日”三列属性或者以最早时间为基准计算差值）、无关属性的删除、多列数据融合等方面。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataAnasly = {}

def checkColumn(columns):
    for column in columns:
        dataAnasly[column]={}
        data = df[column]
        values_counts = data.value_counts()
        dataAnasly[column]['count'] = len(values_counts)

        if(len(values_counts) < 30):
            unique_values = df[column].drop_duplicates().tolist()
            dataAnasly[column]['unique'] = unique_values
        else:
            dataAnasly[column]['unique'] = []

        dtype = df[column].dtype
        dataAnasly[column]['dtype'] = dtype

        has_null = df[column].isnull().any()
        dataAnasly[column]['has_null'] = has_null

    analysis_df = pd.DataFrame(dataAnasly).T
    analysis_df.to_excel('../data/数据分析器.xlsx')

def plot_bar(columnName):
    print(df[columnName].value_counts().sort_index())
    plt.figure(figsize=(10, 6))  # 设置图像大小
    value_counts = df[columnName].value_counts().sort_index()
    value_counts.plot(kind='bar')
    plt.xlabel('values')
    plt.xticks([])
    plt.ylabel('Frequency')
    plt.savefig('../pic/' + columnName + '_Bar.png')
    plt.show()


def check_relation(column1, column2):
    correlation = column1.corr(column2)
    print(correlation)

def check_correlation_matrix():
    correlation_matrix = df.corr().abs()

    correlation_matrix_absolute = correlation_matrix.abs()

    # 获取上三角矩阵，忽略对角线
    upper_tri = np.triu(np.ones(correlation_matrix_absolute.shape))

    # 从上三角矩阵中筛选出所有大于等于0.7并且小于1的元素的位置
    to_select = (correlation_matrix_absolute >= 0.7) & (correlation_matrix_absolute < 1) & upper_tri

    # 获取满足条件的行名与列名的组合
    selected_correlations = correlation_matrix_absolute.where(to_select).stack()

    print("Selected Column Pairs with Correlation >= 0.7 and < 1 and Their Correlations:")
    for (pair, correlation) in selected_correlations.items():
        print(f"{pair}: {correlation}")

train_data_path = '../data/train.csv'
train_data_process_path = '../process/Processed_TrainData.csv'

df = pd.read_csv(train_data_process_path)
# checkColumn(df.columns.tolist())

# plot_bar('interest')

# correlation_matrix = df.corr()
# correlation_matrix = correlation_matrix.abs()
# correlation_matrix.to_excel('../data/相关系数矩阵.xlsx', sheet_name='Correlation Matrix')

# check_correlation_matrix()

# plot_bar('f4')

column1_name = 'total_loan'
column2_name = 'monthly_payment'
column1 = df[column1_name]
column2 = df[column2_name]

plt.subplot(1,2,1)
value_counts = column1.value_counts().sort_index()
value_counts.plot(kind='bar')
plt.xlabel('values')
plt.xticks([])
plt.ylabel('Frequency')
plt.title(column1_name)

plt.subplot(1,2,2)
value_counts = column2.value_counts().sort_index()
value_counts.plot(kind='bar')
plt.xlabel('values')
plt.xticks([])
plt.ylabel('Frequency')
plt.title(column2_name)

plt.savefig('../pic/' + column1_name+'&'+column2_name + '_Bar.png')
plt.show()
