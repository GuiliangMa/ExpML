import numpy as np
import pandas as pd
from SplitData import splitForData
from graphviz import Digraph
from IPython.display import Image, display


class C45:
    def __init__(self, max_depth=None, min_samples_split=1, min_info_gain=0.01):
        self.tree = None
        self.result = None

        # 预剪枝
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_info_gain = min_info_gain

    def train_val_split(self, X, y, test_size=0.2, random_state=None):
        # 设置随机种子
        if random_state:
            np.random.seed(random_state)

        # 打乱索引
        shuffled_indices = np.random.permutation(len(X))
        test_set_size = int(len(X) * test_size)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]

        return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

    def fit(self, X, y, isPruned=False, test_size=0.2, random_state=42):
        self.features = X.columns.tolist()
        # 创建从类别标签到索引的映射，并保存原始标签至self.result
        unique_labels = pd.unique(y)  # 获取y中的唯一值
        self.result = list(unique_labels)  # 保存标签列表

        # 转换y为索引值
        label_to_index = {label: index for index, label in enumerate(unique_labels)}  # 创建从标签到索引的映射
        y_indexed = y.map(label_to_index)  # 将所有y值替换为对应的索引
        X_train, X_val, y_train, y_val = self.train_val_split(X, y_indexed, test_size, random_state)

        # 构建决策树
        self.tree = self.build_tree(X_train, y_train, 1)
        if isPruned:
            self.prune(self.tree, X_val, y_val)
        return self.evaluate_accuracy(X_val, y_val)

    def build_tree(self, X, y, depth):
        # 终止条件：检查是否所有目标变量的值相同
        if len(np.unique(y)) == 1:
            return {'type': 'leaf', 'class': self.result[np.unique(y)[0]]}

        # 终止条件：检查是否超过最大深度
        if self.max_depth and depth > self.max_depth:
            return {'type': 'leaf', 'class': self.result[np.bincount(y).argmax()]}

        # 终止条件：检查样本数是否少于最小分割样本数
        if len(y) < self.min_samples_split:
            return {'type': 'leaf', 'class': self.result[np.bincount(y).argmax()]}

        best_feature, best_threshold, best_gain = self.best_spilt(X, y)

        # 终止条件：检查信息增益是否足够
        if best_gain < self.min_info_gain:
            return {'type': 'leaf', 'class': self.result[np.bincount(y).argmax()]}

        left_X, right_X, left_y, right_y = self.split(X, y, best_feature, best_threshold)

        node = {'feature': best_feature,
                'threshold': best_threshold,
                'type': 'inner',
                'left': self.build_tree(left_X, left_y, depth + 1),
                'right': self.build_tree(right_X, right_y, depth + 1),
                'gain_ratio': best_gain,
                'most_label': self.result[np.bincount(y).argmax()]}

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
                    gain_ratio = self.information_gain_ratio(y, X[feature], thresholds[i])
                    if gain_ratio > best_gain_ratio:
                        best_threshold = thresholds[i]
                        best_feature = feature
                        best_gain_ratio = gain_ratio
        return best_feature, best_threshold, best_gain_ratio

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
                if node['type'] == 'leaf':
                    break
                if sample[node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(node['class'])
        return predictions

    def prune(self, node, X, y):
        if node['type'] == 'leaf':
            # print('leaf node')
            return

        self.prune(node['left'], X, y)
        self.prune(node['right'], X, y)

        # print('branches node')
        origin_accuracy = self.evaluate_accuracy(X, y)
        # print(f'origin:{origin_accuracy}')

        original_left = node.get('left', None)
        original_right = node.get('right', None)

        node['type'] = 'leaf'
        node['class'] = node['most_label']
        del node['left']
        del node['right']

        new_accuracy = self.evaluate_accuracy(X, y)
        # print(f'new:{new_accuracy}')

        if new_accuracy < origin_accuracy:
            # print("New accuracy is low")
            node['type'] = 'inner'
            node['left'] = original_left
            node['right'] = original_right

    def evaluate_accuracy(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean([predictions[i] == self.result[y.iloc[i]] for i in range(len(y))])
        return accuracy

    def plot_tree(self):
        def add_nodes_edges(tree, dot=None, parent_name=None, edge_label=None, feature_name=None):
            if dot is None:
                dot = Digraph()
                # dot = Digraph(graph_attr={"splines": "line"})

            if tree['type'] == 'leaf':
                if edge_label is not None:
                    node_name = f"{feature_name} {edge_label}\n{str(tree['class'])}"
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
    # 56:1 8:0.97 7可能有异常
    X_train, y_train, X_test, y_test = splitForData(df, 0.2, 'Species', random_state=10)

    tree = C45(max_depth=10)
    val_accuracy = tree.fit(X_train, y_train)
    print(f"Not Pruned Val Accuracy: {val_accuracy:.2f}")
    predictions = tree.predict(X_test)
    print(predictions)
    accuracy = np.mean([predictions[i] == y_test.iloc[i] for i in range(len(y_test))])
    print(f"Not Pruned Accuracy: {accuracy:.2f}")

    dot = tree.plot_tree()
    dot.render("../pic/C45/C45_EarlyPrune", format='png')
    image = Image(filename="../pic/C45/C45_EarlyPrune.png")
    display(image)

    print('--------------')

    pruned_tree = C45(max_depth=10)
    val_accuracy = pruned_tree.fit(X_train, y_train, True)
    print(f"Pruned Val Accuracy: {val_accuracy:.2f}")
    predictions = pruned_tree.predict(X_test)
    print(predictions)
    accuracy = np.mean([predictions[i] == y_test.iloc[i] for i in range(len(y_test))])
    print(f"Pruned Accuracy: {accuracy:.2f}")

    dot = pruned_tree.plot_tree()
    dot.render("../pic/C45/C45_AfterPrune", format='png')
    image = Image(filename="../pic/C45/C45_AfterPrune.png")
    display(image)
