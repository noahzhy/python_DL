import os
from keras.datasets import mnist
from keras import models
from keras import layers


# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data(os.path.join(os.getcwd(), 'datasets/mnist.npz'))
print('train labels:', train_labels, len(train_images), train_images.shape)
print('test labels: ', test_labels, len(test_images), test_labels.shape)

# 网络架构
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))