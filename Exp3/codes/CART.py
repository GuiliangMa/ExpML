import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SplitData import splitForData


class CART:
    def __init__(self, max_depth=None,min_samples_split=10, max_gini=0.9):
        self.tree = None
        self.result = None

        # 预剪枝
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_gini = max_gini

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
            return self.result[np.bincount(y).argmax()]

        # 预剪枝
        if len(y) < self.min_samples_split:
            return self.result[np.bincount(y).argmax()]

        best_feature, best_threshold,best_gini = self.best_split(X, y)

        # 预剪枝
        if best_gini > self.max_gini:
            return self.result[np.bincount(y).argmax()]

        left_indices = X[best_feature] <= best_threshold
        left_subtree = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_indices = X[best_feature] > best_threshold
        right_subtree = self.build_tree(X[right_indices], y[right_indices], depth + 1)
        return {"feature": best_feature,
                "threshold": best_threshold,
                "left": left_subtree,
                "right": right_subtree}

    def best_split(self, X, y):
        best_gini = 1.0
        best_feature, best_threshold = None, None
        num_samples, num_features = X.shape
        for feature in X.columns:
            thresholds = np.unique(X[feature])
            for threshold in thresholds:
                gini = self.gini_index(X[feature], y, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold,best_gini

    def gini_index(self, X_feature, y, threshold):
        left_mask = X_feature <= threshold
        right_mask = X_feature > threshold
        left_gini = self.gini(y[left_mask])
        right_gini = self.gini(y[right_mask])
        p_left = float(np.sum(left_mask)) / len(y)
        p_right = float(np.sum(right_mask)) / len(y)
        return p_left * left_gini + p_right * right_gini

    def gini(self, labels):
        if labels.size == 0:
            return 0
        class_probs = np.array([np.sum(labels == c) for c in np.unique(labels)]) / len(labels)
        return 1 - np.sum(class_probs ** 2)

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
    X_train, y_train, X_test, y_test = splitForData(df, 0.2, 'Species')
    tree = CART(max_depth=10)
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)
    print(len(predictions))
    print(predictions)
    accuracy = np.mean([predictions[i] == y_test.iloc[i] for i in range(len(y_test))])
    print(f"Accuracy: {accuracy:.2f}")
