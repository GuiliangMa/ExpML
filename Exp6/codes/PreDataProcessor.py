import subprocess

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import scipy

train_data = pd.read_csv('../data/titanic/train.csv')
test_data = pd.read_csv('../data/titanic/test.csv')
train_processed_data_name = '../data/processed/train.csv'
test_processed_data_name = '../data/processed/test.csv'


def Fill_Embarked_And_Fare_And_Cabin():
    data = pd.concat([train_data, test_data], ignore_index=True)
    train_data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    test_data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    train_data['Fare'].fillna(data['Fare'].median(), inplace=True)
    test_data['Fare'].fillna(data['Fare'].median(), inplace=True)

    train_data['Cabin'] = train_data['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')
    test_data['Cabin'] = test_data['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')


def extract_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


# 提取姓氏的函数
def extract_last_name(name):
    last_name_search = re.search('^(.+?),', name)
    if last_name_search:
        return last_name_search.group(1)
    return ""


def extract_ticket_prefix(ticket):
    match = re.match(r'([A-Za-z][A-Za-z0-9./]*)', ticket)
    if match:
        prefix = match.group(1).strip()
        prefix = prefix.replace('.', '').replace('/', '')
        return prefix
    return 'Unknown'


# 统计票号中去除前缀后的数字长度的函数
def ticket_num_length(ticket):
    ticket_number = re.sub(r'^[A-Za-z][A-Za-z0-9./]*', '', ticket).strip()
    ticket_number = re.sub(r'\D', '', ticket_number)
    return len(ticket_number)


def fill_missing_age(row, predict_dict):
    if pd.isna(row['Age']):
        return predict_dict.get(row['PassengerId'], row['Age'])
    return row['Age']


Fill_Embarked_And_Fare_And_Cabin()
train_data['Title'] = train_data['Name'].apply(extract_title)
train_data['LastName'] = train_data['Name'].apply(extract_last_name)
test_data['Title'] = test_data['Name'].apply(extract_title)
test_data['LastName'] = test_data['Name'].apply(extract_last_name)
train_data['TicketPrefix'] = train_data['Ticket'].apply(extract_ticket_prefix)
test_data['TicketPrefix'] = test_data['Ticket'].apply(extract_ticket_prefix)
train_data['TicketNumLen'] = train_data['Ticket'].apply(ticket_num_length)
test_data['TicketNumLen'] = test_data['Ticket'].apply(ticket_num_length)

train_data = train_data.drop('Name', axis=1)
test_data = test_data.drop('Name', axis=1)
train_data = train_data.drop('Ticket', axis=1)
test_data = test_data.drop('Ticket', axis=1)

data = pd.concat([train_data, test_data], ignore_index=True)
data = data.drop('Survived', axis=1)
data.to_csv("../data/processed/data.csv", index=False)

# 保存为了处理年龄的数据
age_train = data[data['Age'].notnull()]
age_test = data[data['Age'].isnull()]

age_train.to_csv("../data/forAge/train.csv", index=False)
age_test.to_csv("../data/forAge/test.csv", index=False)

print("!!!")
subprocess.run(["python", "AgeModel.py"])

predict_df = pd.read_csv('../data/forAge/predicted.csv')
predict_dict = pd.Series(predict_df['Predicted_Age'].values, index=predict_df['PassengerId']).to_dict()
train_data['Age'] = train_data.apply(lambda row: fill_missing_age(row, predict_dict), axis=1)
test_data['Age'] = test_data.apply(lambda row: fill_missing_age(row, predict_dict), axis=1)
data['Age'] = data.apply(lambda row: fill_missing_age(row, predict_dict), axis=1)
train_data.to_csv(train_processed_data_name, index=False)
test_data.to_csv(test_processed_data_name, index=False)
data.to_csv("../data/processed/data.csv", index=False)
