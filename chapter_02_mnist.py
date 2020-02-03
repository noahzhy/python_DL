import os
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical


# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data(
    os.path.join(os.getcwd(), 'datasets/mnist.npz')
)

print('train labels:', train_labels, len(train_images), train_images.shape)
print('test labels: ', test_labels, len(test_images), test_labels.shape)

# 网络架构
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

# 编译
network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 预处理图像数据
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# 准备标签
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 拟合
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 评估
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc', test_acc)
