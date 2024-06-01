import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SplitData import splitForData
from graphviz import Digraph
from IPython.display import Image, display


class CART:
    def __init__(self, max_depth=None, min_samples_split=1, max_gini=1):
        self.tree = None
        self.result = None

        # 预剪枝
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_gini = max_gini

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
            self.tree = self.prune_tree(self.tree, X_val, y_val)
        return self.evaluate_accuracy(self.tree, X_val, y_val)

    def cost_complexity_pruning(self, tree, X_val, y_val):
        def tree_cost(node):
            if node['type'] == 'leaf':
                return 0
            else:
                return 1 + tree_cost(node['left']) + tree_cost(node['right'])

        def subtree_error(node, X, y):
            if node['type'] == 'leaf':
                pred_class = node['class']
                return np.sum([pred_class != self.result[y.iloc[i]] for i in range(len(y))])
            else:
                feature = node['feature']
                threshold = node['threshold']
                left_indices = X[feature] <= threshold
                right_indices = X[feature] > threshold
                left_error = subtree_error(node['left'], X[left_indices], y[left_indices])
                right_error = subtree_error(node['right'], X[right_indices], y[right_indices])
                return left_error + right_error

        def prune(node, X, y, alpha):
            if node['type'] == 'leaf':
                return node, subtree_error(node, X, y)
            else:
                feature = node['feature']
                threshold = node['threshold']
                left_indices = X[feature] <= threshold
                right_indices = X[feature] > threshold

                node['left'], left_error = prune(node['left'], X[left_indices], y[left_indices], alpha)
                node['right'], right_error = prune(node['right'], X[right_indices], y[right_indices], alpha)

                leaf_error = subtree_error({'type': 'leaf', 'class': node['most_label']}, X, y)
                total_error = left_error + right_error
                if leaf_error + alpha * tree_cost(node) < total_error + alpha * tree_cost(node):
                    return {'type': 'leaf', 'class': self.result[np.bincount(y).argmax()]}, leaf_error
                else:
                    return node, total_error

        alphas = np.linspace(0, 0.1, 101)
        best_tree = tree
        best_acc = self.evaluate_accuracy(tree, X_val, y_val)

        for alpha in alphas:
            pruned_tree, _ = prune(tree, X_val, y_val, alpha)
            acc = self.evaluate_accuracy(pruned_tree, X_val, y_val)
            if acc >= best_acc:
                best_acc = acc
                best_tree = pruned_tree

        return best_tree

    def prune_tree(self, tree, X_val, y_val):
        return self.cost_complexity_pruning(tree, X_val, y_val)

    def build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1:
            return {'type': 'leaf', 'class': self.result[np.unique(y)[0]]}

        # if self.max_depth and depth > self.max_depth:
        #     return {'type': 'leaf', 'class': self.result[np.bincount(y).argmax()]}
        #
        # # 预剪枝
        # if len(y) < self.min_samples_split:
        #     return {'type': 'leaf', 'class': self.result[np.bincount(y).argmax()]}

        best_feature, best_threshold, best_gini = self.best_split(X, y)

        # 预剪枝
        # if best_gini > self.max_gini:
        #     return {'type': 'leaf', 'class': self.result[np.bincount(y).argmax()]}

        left_indices = X[best_feature] <= best_threshold
        left_subtree = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_indices = X[best_feature] > best_threshold
        right_subtree = self.build_tree(X[right_indices], y[right_indices], depth + 1)
        return {"feature": best_feature,
                "threshold": best_threshold,
                'gini': best_gini,
                'type': 'inner',
                "left": left_subtree,
                "right": right_subtree,
                "most_label": self.result[np.bincount(y).argmax()]}

    def predictForEvaluate(self, tree, x):
        """根据决策树预测单个样本的类别"""
        while tree['type'] != 'leaf':
            if x[tree['feature']] <= tree['threshold']:
                tree = tree['left']
            else:
                tree = tree['right']
        return tree['class']

    def evaluate_accuracy(self, tree, X_val, y_val):
        """评估决策树在验证数据集上的准确率"""
        y_pred = [self.predictForEvaluate(tree, x) for index, x in X_val.iterrows()]
        y_val_labels = [self.result[y_val.iloc[i]] for i in range(len(y_val))]
        accuracy = np.mean([y_pred[i] == y_val_labels[i] for i in range(len(y_val_labels))])
        return accuracy

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
        return best_feature, best_threshold, best_gini

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
                if node['type'] == 'leaf':
                    break
                if sample[node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(node['class'])
        return predictions

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
                    node_name = f"{feature_name} {edge_label}\nDivided By {tree['feature']}\nGini={round(tree['gini'], 4)}"
                else:
                    node_name = f"Root\nDivided By {tree['feature']}\nGini={round(tree['gini'], 4)}"
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
    X_train, y_train, X_test, y_test = splitForData(df, 0.2, 'Species', random_state=227)

    tree = CART(max_depth=10)
    val_accuracy = tree.fit(X_train, y_train)
    print(val_accuracy)
    predictions = tree.predict(X_test)
    # print(predictions)
    accuracy = np.mean([predictions[i] == y_test.iloc[i] for i in range(len(y_test))])
    print(f"Accuracy: {accuracy:.2f}")

    dot = tree.plot_tree()
    dot.render("../pic/CART/CART_EarlyPrune", format='png')
    image = Image(filename="../pic/CART/CART_EarlyPrune.png")
    display(image)

    print("---------------")

    prune_tree = CART(max_depth=10)
    val_accuracy = prune_tree.fit(X_train, y_train, True)
    print(val_accuracy)
    predictions = prune_tree.predict(X_test)
    # print(predictions)
    accuracy = np.mean([predictions[i] == y_test.iloc[i] for i in range(len(y_test))])
    print(f"Accuracy: {accuracy:.2f}")

    dot = prune_tree.plot_tree()
    dot.render("../pic/CART/CART_AfterPrune", format='png')
    image = Image(filename="../pic/CART/CART_AfterPrune.png")
    display(image)
