import numpy as np
import pandas as pd

from Exp1.codes.DataProcess import DataProcessor
from Exp1.codes.NaiveBayesClassifier import NaiveBayesClassifier

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

segmentMap = {
    'post_code': 13, 'title': 8, 'known_outstanding_loan': 13,
    'monthly_payment': 20, 'issue_date': 12, 'debt_loan_ratio': 13,
    'scoring_low': 13, 'scoring_high': 13, 'recircle_b': 13,
    'recircle_u': 11, 'f0': 11, 'f2': 13, 'f3': 13,
    'early_return_amount': 13, 'early_return_amount_3mon': 13,
}

def evaluate_model(parameters, dataT):
    data = dataT.copy()
    dp = DataProcessor(data, parameters)
    x_train, y_train = dp.Process()
    x_test, y_test = dp.Deal(test_data)

    nbc = NaiveBayesClassifier(alpha=1)
    nbc.fit(x_train, y_train)
    predictions = nbc.predict(x_test)
    accuracy = np.mean(predictions == y_test)

    return accuracy

def simulated_annealing(data, iterations=500, temp=1.0, temp_decay=0.95):
    current_params = {key: np.random.randint(8, 21) for key in segmentMap.keys()}
    current_score = evaluate_model(current_params, data)
    best_params = current_params.copy()
    best_score = current_score

    for i in range(iterations):
        new_params = current_params.copy()
        for key in new_params.keys():
            if np.random.rand() < 0.5:
                new_params[key] = np.random.randint(8, 21)

        new_score = evaluate_model(new_params, data)

        if new_score > current_score:
            accept = True
        else:
            delta = new_score - current_score
            accept_prob = np.exp(delta / temp)
            accept = np.random.rand() < accept_prob

        if accept:
            current_params, current_score = new_params, new_score
            if new_score > best_score:
                best_params, best_score = new_params.copy(), new_score

        temp *= temp_decay

        print(f"Iteration {i+1}: Current Params: {current_params}, Score: {current_score}, Temp: {temp}")

    return best_params, best_score

best_params, best_score = simulated_annealing(train_data)
print("Best Params:", best_params)
print("Best Score:", best_score)
