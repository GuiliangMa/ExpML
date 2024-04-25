import numpy as np


def splitForData(data, test_size, target, random_state=42):
    np.random.seed(random_state)
    data_shuffled = data.sample(frac=1).reset_index(drop=True)
    train_size = int((1 - test_size) * len(data))
    train_data = data_shuffled[:train_size]
    test_data = data_shuffled[train_size:]
    X_train = train_data.drop(target, axis=1)
    y_train = train_data[target]
    X_test = test_data.drop(target, axis=1)
    y_test = test_data[target]
    return X_train, y_train, X_test, y_test
