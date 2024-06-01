import numpy as np
import pandas as pd
from SplitData import splitForData
from graphviz import Digraph
from IPython.display import Image, display


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
            return self.result[np.bincount(y).argmax()]

        best_feature, best_threshold, best_gain = self.best_split(X, y)
        if best_gain == 0:
            return np.bincount(y).argmax()

        left_X, right_X, left_y, right_y = self.split(X, y, best_feature, best_threshold)

        node = {'feature': best_feature, 'threshold': best_threshold}
        node['left'] = self.build_tree(left_X, left_y, depth + 1)
        node['right'] = self.build_tree(right_X, right_y, depth + 1)
        node['gain_ratio'] = best_gain

        return node

    def best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None

        for feature in X.columns:
            thresholds, gains = self.information_gain(X[feature], y)
            max_gain_idx = np.argmax(gains)
            if gains[max_gain_idx] > best_gain:
                best_gain = gains[max_gain_idx]
                best_feature = feature
                best_threshold = thresholds[max_gain_idx]

        return best_feature, best_threshold, best_gain

    def information_gain(self, X_feature, y):
        thresholds = np.unique(X_feature)
        gains = np.zeros(thresholds.shape)

        for i, threshold in enumerate(thresholds):
            left_y = y[X_feature <= threshold]
            right_y = y[X_feature > threshold]
            gains[i] = self.information_gain_ratio(y, left_y, right_y)

        return thresholds, gains

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    def information_gain_ratio(self, y, left_y, right_y):
        parent_entropy = self.entropy(y)
        left_entropy = self.entropy(left_y)
        right_entropy = self.entropy(right_y)
        p_left = len(left_y) / len(y)
        p_right = len(right_y) / len(y)
        gain = parent_entropy - (p_left * left_entropy + p_right * right_entropy)
        split_info = - (p_left * np.log2(p_left + 1e-10) + p_right * np.log2(p_right + 1e-10))
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

    def plot_tree(self):
        def add_nodes_edges(tree, dot=None, parent_name=None, edge_label=None, feature_name=None):
            if dot is None:
                dot = Digraph()
                # dot = Digraph(graph_attr={"splines": "line"})

            if not isinstance(tree, dict):
                if edge_label is not None:
                    node_name = f"{feature_name} {edge_label}\n{str(tree)}"
                else:
                    node_name = str(tree)
                dot.node(node_name, color="red")
                if parent_name:
                    dot.edge(parent_name, node_name)
            else:
                if edge_label:
                    node_name = f"{feature_name} {edge_label}\nDivided By {tree['feature']}\nGain-Ratio={round(tree['gain_ratio'], 4)}"
                else:
                    node_name = f"Root\nDivided By {tree['feature']}\nGain-Ratio={round(tree['gain_ratio'], 4)}"
                dot.node(node_name, color="blue", shape='box')
                if parent_name:
                    dot.edge(parent_name, node_name)

                if 'left' in tree:
                    add_nodes_edges(tree['left'], dot=dot, parent_name=node_name,
                                    edge_label="<=" + str(tree['threshold']),
                                    feature_name=tree['feature'])
                if 'right' in tree:
                    add_nodes_edges(tree['right'], dot=dot, parent_name=node_name,
                                    edge_label="> " + str(tree['threshold']),
                                    feature_name=tree['feature'])
            return dot

        dot = add_nodes_edges(self.tree)
        return dot


if __name__ == '__main__':
    df = pd.read_csv('../data/iris.csv')
    X_train, y_train, X_test, y_test = splitForData(df, 0.2, 'Species')
    model = C45(max_depth=10)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = np.mean([predictions[i] == y_test.iloc[i] for i in range(len(y_test))])
    print(f"Accuracy: {accuracy:.2f}")

    dot = model.plot_tree()
    dot.render("../pic/C45/C45", format='png')
    image = Image(filename="../pic/C45/C45.png")
    display(image)
