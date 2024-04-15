import os
import pickle
import numpy as np
import cv2 as cv
from NeuralNetwork import NeuralNetwork


def load_cifar10_batch(file_path):
    """ Load a single batch of CIFAR-10 data. """
    with open(file_path, 'rb') as file:
        datadict = pickle.load(file,encoding="latin1")  # 读取全部内容
        data = np.reshape(datadict['data'],(10000,3072))
        labels = np.array(datadict['labels'])
        return data,labels


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
images = images.reshape(images.shape[0], -1).T
labels = to_one_hot(labels, 10)
labels = labels.T

layers = [3072,1024,10]
act_funs = ['relu','sigmoid']

# print("Loaded images shape:", images.shape)
# print("Loaded labels shape:", labels.shape)