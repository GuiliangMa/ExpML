import os
import pickle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from Exp4.codes.NNetwork import NeuralNetwork


def load_cifar10_batch(file_path):
    """ Load a single batch of CIFAR-10 data. """
    with open(file_path, 'rb') as file:
        datadict = pickle.load(file, encoding="latin1")  # 读取全部内容
        data = np.reshape(datadict['data'], (10000, 3072))
        labels = np.array(datadict['labels'])
        return data, labels


def load_all_batches(data_dir, batches):
    """ Load all CIFAR-10 data batches. """
    all_images = []
    all_labels = []
    for batch_name in batches:
        file_path = os.path.join(data_dir, batch_name)
        images, labels = load_cifar10_batch(file_path)
        all_images.append(images)
        all_labels.append(labels)
    return np.concatenate(all_images), np.concatenate(all_labels)


def to_one_hot(labels, num_classes):
    one_hot_labels = np.zeros((labels.size, num_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels


data_dir = '../data/cifar-10-batches-py'
batch_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
images, labels = load_all_batches(data_dir, batch_files)
images = images.astype(np.float32) / 255.0
labels = to_one_hot(labels, 10)

# np.random.seed(42)
# nn = NeuralNetwork(layers=[3072, 128, 64, 10],
#                    activations=['relu', 'relu', 'softmax'],
#                    dropout_rate=0,
#                    l2_lambda=0.01,
#                    loss='cross_entropy')
nn = NeuralNetwork(layers=[3072, 128,64,32, 10],
                   activations=['relu','relu','relu', 'softmax'],
                   dropout_rate=0,
                   l2_lambda=0.01,
                   loss='cross_entropy')
train_epochs = 10
# nn.train(images, labels, train_epochs, 50000, 0.01, patience=5, decay_factor=0.1)
nn.train(images, labels, train_epochs, 64, initial_lr=0.01, patience=100)

loss_list = nn.loss_list

with open(f'../result/test3_{train_epochs}.txt', 'w') as f:
    for loss in loss_list:
        f.write(f"{loss}\n")

epochs = np.arange(0, len(loss_list))
# 绘制损失函数折线图
plt.figure(figsize=(20, 8))
plt.plot(loss_list, label='Loss')
plt.xlabel(f'Epochs={train_epochs}')
plt.ylabel('Loss')
plt.title('Loss Function Over Time')
plt.legend()
plt.grid(True)
# 保存图像
plt.xticks(epochs)
plt.savefig(f'../pic/loss__iter{train_epochs}.png')
# 显示图像
plt.show()

test_files = ['test_batch']
test_images, test_labels = load_all_batches(data_dir, test_files)
test_images = test_images.astype(np.float32) / 255.0
test_labels = to_one_hot(test_labels, 10)
test_pred = nn.predict(test_images)
test_accuracy = nn.calculate_accuracy(test_labels, test_pred)
print('========================')
print(f"Test Accuracy: {test_accuracy:.2f}%")

# print("Loaded images shape:", images.shape)
# print("Loaded labels shape:", labels.shape)
