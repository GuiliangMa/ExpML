import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataProcessor import DataProcessor
from NaiveBayesClassifier import NaiveBayesClassifier

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')
dp = DataProcessor(train_data)
x_train, y_train = dp.Process()
x_test, y_test = dp.Deal(test_data)
x_test.to_csv('../process/testData.csv', index=False)

nbc = NaiveBayesClassifier(alpha=1)
nbc.fit(x_train, y_train)
pred = nbc.predict(x_test)

correct_predictions = sum(p == t for p, t in zip(pred, y_test))
accuracy = correct_predictions / len(y_test)
print(accuracy)


