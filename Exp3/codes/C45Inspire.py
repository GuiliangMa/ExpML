import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SplitData import splitForData


class C45:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.result = None

    def fit(self, X, y):
        self.features = X.columns.tolist()
        # 创建从类别标签到索引的映射，并保存原始标签至self.result
        unique_labels = pd.unique(y)  # 获取y中的唯一值
        self.result = list(unique_labels)  # 保存标签列表

        # 转换y为索引值
        label_to_index = {label: index for index, label in enumerate(unique_labels)}  # 创建从标签到索引的映射
        y_indexed = y.map(label_to_index)  # 将所有y值替换为对应的索引

        # 构建决策树
        self.tree = self.build_tree(X, y_indexed, 1)

    def build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1:
            return self.result[np.unique(y)[0]]

        if self.max_depth and depth > self.max_depth:
            print(np.bincount(y))
            return self.result[np.bincount(y).argmax()]

        best_feature, best_threshold, best_gain = self.best_spilt(X, y)
        left_X, right_X, left_y, right_y = self.split(X, y, best_feature, best_threshold)

        node = {'feature': best_feature, 'threshold': best_threshold}
        node['left'] = self.build_tree(left_X, left_y, depth + 1)
        node['right'] = self.build_tree(right_X, right_y, depth + 1)

        return node

    def best_spilt(self, X, y):
        best_gain_ratio = 0
        best_feature = None
        best_threshold = None

        all_gains = []

        for feature in X.columns:
            thresholds, gains = self.information_gain(X[feature], y)
            all_gains.extend(gains)

        average_gain = np.mean(all_gains)

        for feature in X.columns:
            thresholds, gains = self.information_gain(X[feature], y)
            for i, gain in enumerate(gains):
                if gain > average_gain:
                    gain_ratio = self.information_gain_ratio(y,X[feature],thresholds[i])
                    if gain_ratio > best_gain_ratio:
                        best_threshold = thresholds[i]
                        best_feature = feature
                        best_gain = gain_ratio
        return best_feature, best_threshold, best_gain

    def information_gain(self, X_feature, y):
        thresholds = np.unique(X_feature)
        gains = np.zeros(len(thresholds))
        for i, threshold in enumerate(thresholds):
            left_y = y[X_feature <= threshold]
            right_y = y[X_feature > threshold]
            gains[i] = self.gain(y, left_y, right_y)
        return thresholds, gains

    def gain(self, y, left_y, right_y):
        parent_entropy = self.entropy(y)
        left_entropy = self.entropy(left_y)
        right_entropy = self.entropy(right_y)
        p_left = len(left_y) / len(y)
        p_right = len(right_y) / len(y)
        return parent_entropy - (p_left * left_entropy + p_right * right_entropy)

    def entropy(self, y):
        labels = np.unique(y)
        Ent = 0.0
        for label in labels:
            p = len(y[y == label]) / len(y)
            Ent -= p * np.log2(p)
        return Ent

    def information_gain_ratio(self, y, X_feature, threshold):
        left_y = y[X_feature <= threshold]
        right_y = y[X_feature > threshold]
        gain = self.gain(y, left_y, right_y)
        split_info = -((len(left_y) / len(y)) * np.log2(len(left_y) / len(y) + 1e-10) +
                       (len(right_y) / len(y)) * np.log2(len(right_y) / len(y) + 1e-10))
        return gain / split_info if split_info > 1e-10 else 0

    def split(self, X, y, feature, threshold):
        left_mask = X[feature] <= threshold
        right_mask = X[feature] > threshold
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def predict(self, X):
        predictions = []
        for index, sample in X.iterrows():
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
    X_train, y_train, X_test, y_test = splitForData(df, 0.2 , 'Species')
    tree = C45(max_depth=10)
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)
    print(predictions)
    accuracy = np.mean([predictions[i] == y_test.iloc[i] for i in range(len(y_test))])
    print(f"Accuracy: {accuracy:.2f}")
