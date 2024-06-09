import numpy as np
import pandas as pd
from graphviz import Digraph
from IPython.display import Image, display

from SplitData import splitForData


class ID3Continuous:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.result = None

    def entropy(self, y):
        labels = np.unique(y)
        Ent = 0.0
        for label in labels:
            p = len(y[y == label]) / len(y)
            Ent -= p * np.log(p)
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
        return best_feature, best_threshold, best_information_gain

    def build_tree(self, X, y, features, depth=0):
        if len(np.unique(y)) == 1:
            return self.result[np.unique(y)[0]]

        if self.max_depth is not None and depth > self.max_depth:
            return self.result[np.bincount(y).argmax()]

        best_feature, best_threshold,best_gain = self.best_spilt(X, y)
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        node = {}
        node['feature'] = features[best_feature]
        node['threshold'] = best_threshold
        node['gain'] = best_gain
        node['left'] = self.build_tree(X[left_indices], y[left_indices], features, depth + 1)
        node['right'] = self.build_tree(X[right_indices], y[right_indices], features, depth + 1)
        return node

    def fit(self, X, y):
        features = X.columns.tolist()
        X = X.values
        unique_labels = pd.unique(y)  # 获取y中的唯一值
        self.result = list(unique_labels)  # 保存标签列表

        # 转换y为索引值
        label_to_index = {label: index for index, label in enumerate(unique_labels)}  # 创建从标签到索引的映射
        y_indexed = y.map(label_to_index)  # 将所有y值替换为对应的索引
        self.tree = self.build_tree(X, y_indexed, features, 0)

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
                    node_name = f"{feature_name} {edge_label}\nDivided By {tree['feature']}\nGain={round(tree['gain'],4)}"
                else:
                    node_name = f"Root\nDivided By {tree['feature']}\nGain={round(tree['gain'],4)}"
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

    def predict_and_plot(self,X,dot):
        def add_nodes_edges(X,tree, dot=None, parent_name=None, edge_label=None, feature_name=None):

            if not isinstance(tree, dict):
                if edge_label is not None:
                    node_name = f"{feature_name} {edge_label}\n{str(tree)}"
                else:
                    node_name = str(tree)
                dot.node(node_name, color="red",style='filled',fillcolor='yellow')
            else:
                if edge_label:
                    node_name = f"{feature_name} {edge_label}\nDivided By {tree['feature']}\nGain={round(tree['gain'], 4)}"
                else:
                    node_name = f"Root\nDivided By {tree['feature']}\nGain={round(tree['gain'], 4)}"
                dot.node(node_name, color="blue", shape='box',style='filled',fillcolor='yellow')

                if 'left' in tree and X[tree['feature']] <= tree['threshold']:
                    add_nodes_edges(X,tree['left'], dot=dot, parent_name=node_name,
                                    edge_label="<=" + str(tree['threshold']),
                                    feature_name=tree['feature'])
                if 'right' in tree and X[tree['feature']] > tree['threshold']:
                    add_nodes_edges(X,tree['right'], dot=dot, parent_name=node_name,
                                    edge_label="> " + str(tree['threshold']),
                                    feature_name=tree['feature'])
            return dot

        dot = add_nodes_edges(X,self.tree,dot)
        return dot


if __name__ == '__main__':
    df = pd.read_csv('../data/iris.csv')
    X_train, y_train, X_test, y_test = splitForData(df, 0.2, 'Species')
    model = ID3Continuous(max_depth=10)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(predictions)
    accuracy = np.mean([predictions[i] == y_test.iloc[i] for i in range(len(y_test))])
    print(f"Accuracy: {accuracy:.2f}")

    dot = model.plot_tree()

    X_predict = df.iloc[120, :]
    print(X_predict)
    dot = model.predict_and_plot(X_predict, dot)

    dot.render("../pic/ID3/ID3continous3", format='png')
    image = Image(filename="../pic/ID3/ID3continous3.png")
    display(image)
