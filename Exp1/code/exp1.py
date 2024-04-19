import pandas as pd
from DataProcess import DataProcessor
from NaiveBayesClassifier import NaiveBayesClassifier

segmentMap = {
        'post_code': 13,
        # 10
        'title': 8,
        'known_outstanding_loan': 13,

        # 'total_loan': 13,
        # 20,39
        'monthly_payment': 20,
        'issue_date': 12,
        'debt_loan_ratio': 13,
        'scoring_low': 13,
        'scoring_high': 13,
        'recircle_b': 13,
        'recircle_u': 11,
        'f0': 11,
        'f2': 13,
        'f3': 13,

        'early_return_amount': 13,
        'early_return_amount_3mon': 13,
}

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

columnName = 'monthly_payment'
dp = DataProcessor(train_data,segmentMap)

x_train, y_train = dp.Process()
x_test, y_test = dp.Deal(test_data)
# x_test.to_csv('../process/testData.csv', index=False)

nbc = NaiveBayesClassifier(alpha=1)
nbc.fit(x_train, y_train)
pred = nbc.predict(x_test)

correct_predictions = sum(p == t for p, t in zip(pred, y_test))
accuracy = correct_predictions / len(y_test)
print(accuracy)