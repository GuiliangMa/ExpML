---
Field:
    - 预测

Ext:
    - .zip

DatasetUsage:
    - 91056
---

## **背景描述**
Titanic数据集是非常适合数据科学和机器学习新手入门练习的数据集。
数据集为1912年泰坦尼克号沉船事件中一些船员的个人信息以及存活状况。这些历史数据已经非分为训练集和测试集，你可以根据训练集训练出合适的模型并预测测试集中的存活状况。

另外，添加了测试集的ground_truth，方便大家与自己预测结果进行对比，从而对自己的工作有个客观的评价。

## **数据说明**
数据描述

| **变量名**   | PassengerId | Survived                 | Pclass                                 | Name     | Sex         | Age         | SibSp                    | Parch                | Ticket   | Fare    | Cabin  | Embarked                                                     |
| :----------- | :---------- | :----------------------- | :------------------------------------- | :------- | :---------- | :---------- | :----------------------- | :------------------- | :------- | :------ | :----- | :----------------------------------------------------------- |
| **变量解释** | 乘客编号    | 乘客是否存活(0=NO 1=Yes) | 乘客所在的船舱等级,(1=1st,2=2nd,3=3rd) | 乘客姓名 | 乘客性别    | 乘客年龄    | 乘客的兄弟姐妹和配偶数量 | 乘客的父母与子女数量 | 票的编号 | 票价    | 座位号 | 乘客登船码头。 C = Cherbourg; Q = Queenstown; S = Southampton |
| **数据类型** | numeric     | categorical              | categorical                            | string   | categorical | categorical | numeric                  | numeric              | string   | numeric | string | categorical                                                  |

## **数据来源**
[Titanic Competition : How top LB got their score - Kaggle](https://www.kaggle.com/tarunpaparaju/titanic-competition-how-top-lb-got-their-score)

## **问题描述**
数据科学竞赛入门

## **引用格式**
```
@misc{titanic_ans7798,
    title = { 泰坦尼克号生存预测（有答案版） },
    author = { 王大毛 },
    howpublished = { \url{https://www.heywhale.com/mw/dataset/5e785ba398d4a8002d2c2ce4} },
    year = { 2020 },
}
```