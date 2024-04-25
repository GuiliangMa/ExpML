import pandas as pd
from DataProcess import DataProcessor
from NaiveBayesClassifier import NaiveBayesClassifier

segmentMap = {'post_code': 10, 'title': 18, 'known_outstanding_loan': 11, 'monthly_payment': 20, 'issue_date': 15,
              'debt_loan_ratio': 20, 'scoring_low': 12, 'scoring_high': 13, 'recircle_b': 20, 'recircle_u': 17,
              'f0': 16, 'f2': 17, 'f3': 10, 'early_return_amount': 8, 'early_return_amount_3mon': 11}

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

columnName = 'monthly_payment'
dp = DataProcessor(train_data, segmentMap)

x_train, y_train = dp.Process()
x_test, y_test = dp.Deal(test_data)
# x_test.to_csv('../process/testData.csv', index=False)

nbc = NaiveBayesClassifier(alpha=1)
nbc.fit(x_train, y_train)
pred = nbc.predict(x_test)

correct_predictions = sum(p == t for p, t in zip(pred, y_test))
accuracy = correct_predictions / len(y_test)
print(f'Accuracy: {accuracy}')
