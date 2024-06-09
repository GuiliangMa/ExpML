import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

train_data = pd.read_csv('../data/processed/train.csv')
test_data = pd.read_csv('../data/processed/test.csv')
data = pd.read_csv('../data/processed/data.csv')

numerical_features = ['Age', 'SibSp', 'Parch', 'Fare', 'TicketNumLen']
categorical_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Title', 'TicketPrefix']
# categorical_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Title']


def manual_one_hot_encode(df, column):
    unique_values = df[column].unique()
    for value in unique_values:
        df[f"{column}_{value}"] = df[column].apply(lambda x: 1 if x == value else 0)
    df.drop(column, axis=1, inplace=True)

for num_feature in numerical_features:
    mean = data[num_feature].mean()
    std = data[num_feature].std()
    train_data[num_feature] = (train_data[num_feature] - mean) / std
    test_data[num_feature] = (test_data[num_feature] - mean) / std

for column in categorical_features:
    unique_values = data[column].unique()
    for unique in unique_values:
        train_data[f"{column}_{unique}"] = train_data[column].apply(lambda x: 1 if x == unique else 0)
        test_data[f"{column}_{unique}"] = test_data[column].apply(lambda x: 1 if x == unique else 0)
    train_data = train_data.drop(column, axis=1)
    test_data = test_data.drop(column, axis=1)
# train_data = train_data.drop('TicketNumLen', axis=1)
# test_data = test_data.drop('TicketNumLen', axis=1)


train_data.to_csv('../data/preTrain/train.csv', index=False)
test_data.to_csv('../data/preTrain/test.csv', index=False)

# train_data.to_csv('../data/preTrain/trainNoTicketLen.csv', index=False)
# test_data.to_csv('../data/preTrain/testNoTicketLen.csv', index=False)
