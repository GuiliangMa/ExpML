# 机器学习基础 实验一 实验报告

## 实验目的

本实验以贷款违约为背景，要求使用贝叶斯决策论的相关知识在训练集上构建模型，在测试集上进行贷款违约预测并计算分类准确度。

## 实验要求

1. 实验不限制使用何种高级语言，推荐使用python中pandas库处理csv文件。
   - 安装：pip install pandas/conda install pandas【在使用conda命令，需安装anaconda环境】
   - 导入：import pandas as pd【建议】
2. 在进行贝叶斯分类之前重点是对数据进行预处理操作，如，缺失值的填充、将文字表述转为数值型、日期处理格式（处理成“年-月-日”三列属性或者以最早时间为基准计算差值）、无关属性的删除、多列数据融合等方面。
3. 数据中存在大量连续值的属性，不能直接计算似然，需要将连续属性离散化。
4. 另外，特别注意零概率问题，贝叶斯算法中如果乘以0的话就会失去意义，需要使用平滑技术。【可以百度了解一下拉普拉斯平滑】
5. 实验目的是使用贝叶斯处理实际问题，不得使用现成工具包直接进行分类。【该点切记！！！这个一定要自己写，才能感受贝叶斯的魅力】
6. 实验代码中需要有必要的注释，具有良好的可读性。

## 实验过程

### 1.数据简易分析

在进行贝叶斯决策之前，我应当先对数据进行一个整体的分析。对此我编写了如下代码来粗略检测每个特征的基本情况，即重复性，缺失性，数值类型，特征值个数。来对特征进行一个简单的处理。并进行手动记录，得到如下的表格。

```python
'''
DataAnaslyze.py
'''

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
    plt.xticks([])
    plt.ylabel('Frequency')
    plt.show()


train_data_path = '../data/train.csv'
df = pd.read_csv(train_data_path)
columns = df.columns.tolist()
# check('early_return_amount_3mon')
# checknull(columns)
# plot_bar('total_loan')
```

**特征属性（仅以训练集来做例子，此处假设训练集与测试集独立同分布）**

| 特征名称                 | 唯一性 | 缺失性 | 数据类型         | 特征个数                                                     | 预计处理方法                                         |
| ------------------------ | ------ | ------ | ---------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| loan_id                  | 唯一   | 无     | 整数             | 9000                                                         | **舍去**                                             |
| user_id                  | 唯一   | 无     | 整数             | 9000                                                         | **舍去**                                             |
| total_loan               | 否     | 无     | 实数             | 1526                                                         | 将该部分进行分段处理。分段后的代表值应当为一个整数。 |
| year_of_loan             | 否     | 无     | 整数             | 2[3, 5]                                                      |                                                      |
| interest                 | 否     | 无     | 实数             | 1001                                                         |                                                      |
| monthly_payment          | 否     | 无     | 实数             | 5872                                                         |                                                      |
| class                    | 否     | 无     | 字符             | 7['B', 'C', 'A', 'G', 'D', 'E', 'F']                         |                                                      |
| employer_type            | 否     | 无     | 字符串           | 6['幼教与中小学校', '政府机构', '上市企业', '普通企业',  '世界五百强', '高等教育机构'] |                                                      |
| industry                 | 否     | 无     | 字符串           | 14['金融业', '国际组织', '文化和体育业', '建筑业',  '电力、热力生产供应业', '批发和零售业', '采矿业', '房地产业', '交通运输、仓储和邮政业', '公共服务、社会组织', '住宿和餐饮业',  '信息传输、软件和信息技术服务业', '农、林、牧、渔业', '制造业'] |                                                      |
| work_year                | 否     | 有     | 字符串           | 12[nan, '8 years', '10+  years', '3 years', '7 years', '4 years', '1 year', '< 1 year', '6 years',  '2 years', '9 years', '5 years'] |                                                      |
| house_exist              | 否     | 无     | 整数             | 5[2, 1, 0, 3, 4]                                             |                                                      |
| censor_status            | 否     | 无     | 整数             | 3[2, 0, 1]                                                   |                                                      |
| issue_date               | 否     | 无     | 日期(yyyy/mm/dd) | 126                                                          |                                                      |
| use                      | 否     | 无     | 整数             | 14[0, 4, 2, 9, 5, 3, 6, 1, 7, 8, 10,  12, 11, 13]            |                                                      |
| post_code                | 否     | 无     | 整数             | 786                                                          |                                                      |
| region                   | 否     | 无     | 整数             | 50                                                           |                                                      |
| debt_loan_ratio          | 否     | 无     | 实数             | 5822                                                         |                                                      |
| del_in_18month           | 否     | 无     | 整数             | 12[0, 3, 1, 4, 2, 9, 5, 6, 7, 8, 10,  15]                    |                                                      |
| scoring_low              | 否     | 无     | 实数             | 132                                                          |                                                      |
| scoring_high             | 否     | 无     | 实数             | 360                                                          |                                                      |
| known_outstanding_loan   | 否     | 无     | 整数             | 47                                                           |                                                      |
| known_dero               | 否     | 无     | 整数             | 13[1, 0, 2, 3, 8, 10, 4, 9, 5, 6, 11,  12, 7]                |                                                      |
| pub_dero_bankrup         | 否     | 有     | 整数             | 6[1.0, 0.0, 3.0, 2.0, nan,  5.0]                             |                                                      |
| recircle_b               | 否     | 否     | 实数             | 8558                                                         |                                                      |
| recircle_u               | 否     | 否     | 实数             | 2991                                                         |                                                      |
| initial_list_status      | 否     | 否     | 整数             | 2[1,  0]                                                     |                                                      |
| app_type                 | 否     | 否     | 整数             | 2[0,  1]                                                     |                                                      |
| earlies_credit_mon       | 否     | 否     | 日期(mm/dd)      | 520                                                          |                                                      |
| title                    | 否     | 否     | 整数             | 843                                                          |                                                      |
| policy_code              | 恒为1  | 否     | 整数             | 1                                                            | **舍去**                                             |
| f0                       | 否     | 是     | 整数             | 30                                                           |                                                      |
| f1                       | 否     | 是     | 整数             | 3[0.0, nan, 1.0]                                             |                                                      |
| f2                       | 否     | 是     | 整数             | 59                                                           |                                                      |
| f3                       | 否     | 是     | 整数             | 63                                                           |                                                      |
| f4                       | 否     | 是     | 整数             | 40                                                           |                                                      |
| early_return             | 否     | 否     | 整数             | 6[0, 3, 1, 2, 5, 4]                                          |                                                      |
| early_return_amount      | 否     | 否     | 整数             | 3967                                                         |                                                      |
| early_return_amount_3mon | 否     | 否     | 整数             | 3484                                                         |                                                      |

**预测值**

| 属性名    | 数值类型 | 属性类型 |
| --------- | -------- | -------- |
| isDefault | 整数     | 2[0,1]   |



### 2.数据处理

#### 2.1 对total_loan 的处理

考虑total_loan数据值的分布图（如下），可以发现total_loan的分布疑似会聚集在39个值附近，考虑用kmeans将其聚拢，分成39类来进行简单处理。由上表可知，上表中total_loan这个属性可能存在1500余种，若进行这样分类，则将其下降为39种值。为了让后续计算遍历，每一类的中值点并不一定去代表这个类型，取一个实数值可能更好。

![](G:\ExpMachineLearn\ExpML\Exp1\pic\total_loan_Bar.png)

