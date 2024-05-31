import pandas as pd
from DataProcess import DataProcessor
from NaiveBayesClassifier import NaiveBayesClassifier

# 0.827
segmentMapOrigin = {'post_code': 14, 'title': 14, 'known_outstanding_loan': 14, 'monthly_payment': 14, 'issue_date': 14,
                    'debt_loan_ratio': 14, 'scoring_high': 14, 'recircle_b': 14, 'recircle_u': 14, 'f0': 14, 'f2': 14,
                    'f3': 14, 'early_return_amount_3mon': 14}

# 0.849
segmentMap849 = {'post_code': 10, 'title': 18, 'known_outstanding_loan': 11, 'monthly_payment': 20, 'issue_date': 15,
                 'debt_loan_ratio': 20, 'scoring_low': 12, 'scoring_high': 13, 'recircle_b': 20, 'recircle_u': 17,
                 'f0': 16, 'f2': 17, 'f3': 10, 'early_return_amount_3mon': 11}

# 0.85
segmentMap850 = {'post_code': 13, 'title': 20, 'known_outstanding_loan': 11, 'monthly_payment': 13, 'issue_date': 10,
                 'debt_loan_ratio': 16, 'scoring_low': 18, 'scoring_high': 15, 'recircle_b': 14, 'recircle_u': 15,
                 'f0': 15, 'f2': 8, 'f3': 20, 'early_return_amount_3mon': 9}

segmentMapKFold = {'post_code': 18, 'title': 15, 'known_outstanding_loan': 12, 'monthly_payment': 12, 'issue_date': 19,
                   'debt_loan_ratio': 8, 'scoring_low': 10, 'scoring_high': 9, 'recircle_b': 13, 'recircle_u': 17,
                   'f0': 11, 'f2': 18, 'f3': 12, 'early_return_amount_3mon': 10}

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

columnName = 'monthly_payment'
dp = DataProcessor(train_data, segmentMapKFold)

x_train, y_train = dp.Process()
x_test, y_test = dp.Deal(test_data)
# x_test.to_csv('../process/testData.csv', index=False)

nbc = NaiveBayesClassifier(alpha=1)
nbc.fit(x_train, y_train)
pred = nbc.predict(x_test)

correct_predictions = sum(p == t for p, t in zip(pred, y_test))
accuracy = correct_predictions / len(y_test)
print(f'Accuracy: {accuracy}')
