import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SplitData import splitForData
import pydotplus
from graphviz import Source
from IPython.display import Image, display


def entropy(y):
    labels = np.unique(y)
    Ent = 0.0
    for label in labels:
        p = len(y[y == label]) / len(y)
        Ent -= p * np.log(p)
    return Ent


class ID3Discrete:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.result = None

    def best_split(self, X, y):
        base_entropy = entropy(y)
        best_information_gain = -1
        best_feature = None
        best_splits = None

        for i in range(X.shape[1]):
            values = X[:, i]
            unique_values = np.unique(values)
            splits = {}
            for value in unique_values:
                splits[value] = y[values == value]

            new_entropy = 0
            for subset in splits.values():
                new_entropy += len(subset) / len(y) * entropy(subset)

            information_gain = base_entropy - new_entropy

            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature = i
                best_splits = splits

        return best_feature, best_splits

    def build_tree(self, X, y, features, depth=0):
        if len(np.unique(y)) == 1:
            return self.result[np.unique(y)[0]]
        if len(features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return self.result[np.bincount(y).argmax()]

        best_feature, best_splits = self.best_split(X, y)
        if best_feature is None:
            return self.result[np.bincount(y).argmax()]

        tree = {features[best_feature]: {}}

        remaining_features = features[:best_feature] + features[best_feature + 1:]

        for feature_value, subset in best_splits.items():
            subset_X = X[X[:, best_feature] == feature_value]
            subset_y = subset
            subset_X = np.delete(subset_X, best_feature, axis=1)

            subtree = self.build_tree(subset_X, subset_y, remaining_features, depth + 1)
            tree[features[best_feature]][feature_value] = subtree
        return tree

    def fit(self, X, y):
        features = X.columns.tolist()
        X = X.values
        unique_labels = pd.unique(y)  # 获取y中的唯一值
        self.result = list(unique_labels)  # 保存标签列表

        # 转换y为索引值
        label_to_index = {label: index for index, label in enumerate(unique_labels)}  # 创建从标签到索引的映射
        y_indexed = y.map(label_to_index)  # 将所有y值替换为对应的索引
        self.tree = self.build_tree(X, y_indexed, features, 0)

    def predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        feature = list(tree.keys())[0]
        if x[feature] in tree[feature]:
            subtree = tree[feature][x[feature]]
            return self.predict_one(x, subtree)
        else:
            return self.result[0]

    def predict(self, X):
        return X.apply(lambda x: self.predict_one(x, self.tree), axis=1)

    def plot_tree(self):
        dot_data = pydotplus.graphviz.Dot(graph_type="graph")

        def add_nodes_edges(tree, parent_name=None):
            if isinstance(tree, dict):
                for k, v in tree.items():
                    print(k, v)
                    node_name = str(k)
                    dot_data.add_node(pydotplus.graphviz.Node(node_name, shape="box"))
                    if parent_name:
                        dot_data.add_edge(pydotplus.graphviz.Edge(parent_name, node_name, dir="none"))
                    add_nodes_edges(v, node_name)
            else:
                node_name = str(tree)
                dot_data.add_node(pydotplus.graphviz.Node(node_name, color="red"))
                if parent_name:
                    dot_data.add_edge(pydotplus.graphviz.Edge(parent_name, node_name, dir="none"))

        add_nodes_edges(self.tree)
        return dot_data


if __name__ == "__main__":
    df = pd.read_csv('../data/iris.csv')
    X_train, y_train, X_test, y_test = splitForData(df, 0.2, 'Species', 42)
    model = ID3Discrete(max_depth=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = np.array(predictions)
    # print(predictions)
    accuracy = np.mean([predictions[i] == y_test.iloc[i] for i in range(len(y_test))])
    # print(f"Accuracy: {accuracy:.2f}")

    dot_data = model.plot_tree()
    graph = pydotplus.graph_from_dot_data(dot_data.to_string())
    graph.write_png("ID3discrete.png")
    image = Image(filename="ID3discrete.png")
    display(image)
