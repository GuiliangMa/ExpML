import numpy as np
import pandas as pd

from Exp1.codes.DataProcess import DataProcessor
from Exp1.codes.NaiveBayesClassifier import NaiveBayesClassifier


train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

segmentMap = {
        'post_code': 13,
        'title': 8,
        'known_outstanding_loan': 13,
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
        'early_return_amount_3mon': 13,
}

def evaluate_model(parameters, dataT):
    # 这个函数应该基于提供的参数处理数据并返回模型准确率
    data = dataT.copy()
    dp = DataProcessor(data, parameters)
    x_train, y_train = dp.Process()  # 处理训练数据
    x_test, y_test = dp.Deal(test_data)  # 处理测试数据

    nbc = NaiveBayesClassifier(alpha=1)
    nbc.fit(x_train, y_train)
    predictions = nbc.predict(x_test)
    accuracy = np.mean(predictions == y_test)

    dp.Segment={}
    dp.isProcessed = False
    dp.data = None

    return accuracy


def random_search(data, iterations=100):
    best_params = None
    best_score = 0

    for _ in range(iterations):
        # 随机生成参数
        params = {key: np.random.randint(8, 21) for key in segmentMap.keys()}

        # 评估当前参数集
        score = evaluate_model(params, data)

        # 检查是否是最佳参数集
        if score > best_score:
            best_score = score
            best_params = params

        print(f"Current Params: {params} Score: {score}")

    return best_params, best_score


# 调用随机搜索
best_params, best_score = random_search(train_data)
print("Best Params:", best_params)
print("Best Score:", best_score)
