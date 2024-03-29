# 在进行贝叶斯分类之前重点是对数据进行预处理操作。
# 如，缺失值的填充、将文字表述转为数值型、日期处理格式（处理成“年-月-日”三列属性或者以最早时间为基准计算差值）、无关属性的删除、多列数据融合等方面。

# 检验结果：
# loan_id: 无重复值，无缺失值，其数值类型为整数。足够离散，但个人感觉归根到底该键对贝叶斯预测无关，应当作为消去列
# user_id: 无重复值，无缺失值，其数值类型为整数。足够离散，但个人感觉归根到底该键对贝叶斯预测无关，应当作为消去列
# total_loan: 存在重复值，无缺失值，其数据类型为疑似连续的实数，需要进行处理。该属性应当与贝叶斯预测相关，不应当消去
# year_of_loan: 共计两个不同的整数值，3(6862),5(2138)，无缺失值。不应当消去
# interest: 存在重复值，无缺失值，其数据类型为疑似连续的实数，需要处理。不消去
# monthly_payment: 存在重复值，无缺失值，其数据类型为疑似连续的实数，需要处理。不消去
# class: 存在重复值，无缺失值，A-G为字符，需要转换(0-6?)，需要处理。不消去
# employer_type: 存在重复值，无缺失值，汉字字符[普通企业,政府机构,幼教与中小学校,上市企业,世界五百强,高等教育机构]，需要处理。不消去
# industry: 存在重复值，无缺失值，汉字字符['金融业', '国际组织', '文化和体育业', '建筑业', '电力、热力生产供应业', '批发和零售业', '采矿业', '房地产业', '交通运输、仓储和邮政业', '公共服务、社会组织', '住宿和餐饮业', '信息传输、软件和信息技术服务业', '农、林、牧、渔业', '制造业']，需要处理。不消去

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


train_data_path = '../data/train.csv'
df = pd.read_csv(train_data_path)
columns = df.columns.tolist()
# check('early_return_amount_3mon')
checknull(columns)


