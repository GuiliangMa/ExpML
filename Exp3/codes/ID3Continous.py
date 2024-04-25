import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SplitData import splitForData


class ID3Continuous:
    def __init__(self,max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.result = None

    def entropy(self, y):
        labels = np.unique(y)
        Ent = 0.0
        for label in labels:
            p = len(y[y == label]) / len(y)
            Ent -= p * np.log2(p)
        return Ent

    def best_spilt(self, X, y):
        base_entropy = self.entropy(y)
        best_information_gain = -1
        best_threshold = None
        best_feature = None

        for i in range(X.shape[1]):
            values = X[:, i]
            unique_values = np.unique(values)
            for threshold in unique_values:
                y_left = y[values <= threshold]
                y_right = y[values > threshold]
                p_left = len(y_left) / len(values)
                p_right = len(y_right) / len(values)

                information_gain = base_entropy - (p_left * self.entropy(y_left) + p_right * self.entropy(y_right))
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_threshold = threshold
                    best_feature = i
        return best_feature, best_threshold

    def build_tree(self, X, y, features, depth=0, max_depth=10):
        if len(np.unique(y)) == 1:
            return self.result[np.unique(y)[0]]

        if self.max_depth and depth > self.max_depth:
            return self.result[np.bincount(y).argmax()]

        best_feature, best_threshold = self.best_spilt(X, y)
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        node = {}
        node['feature'] = features[best_feature]
        node['threshold'] = best_threshold
        node['left'] = self.build_tree(X[left_indices], y[left_indices], features, depth + 1, max_depth)
        node['right'] = self.build_tree(X[right_indices], y[right_indices], features, depth + 1, max_depth)
        return node

    def fit(self, X, y, max_depth=10):
        features = X.columns.tolist()
        X = X.values
        unique_labels = pd.unique(y)  # 获取y中的唯一值
        self.result = list(unique_labels)  # 保存标签列表

        # 转换y为索引值
        label_to_index = {label: index for index, label in enumerate(unique_labels)}  # 创建从标签到索引的映射
        y_indexed = y.map(label_to_index)  # 将所有y值替换为对应的索引
        self.tree = self.build_tree(X, y_indexed, features, 0, max_depth)

    def predict(self, X):
        predictions = []
        for index,sample in X.iterrows():
            node = self.tree
            while isinstance(node, dict):
                if sample[node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(node)
        return predictions


if __name__ == '__main__':
    df = pd.read_csv('../data/iris.csv')
    X_train, y_train, X_test, y_test = splitForData(df, 0.2, 'Species')
    tree = ID3Continuous(max_depth=10)
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)
    print(predictions)
    accuracy = np.mean([predictions[i] == y_test.iloc[i] for i in range(len(y_test))])
    print(f"Accuracy: {accuracy:.2f}")