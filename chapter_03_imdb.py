import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(os.path.join(os.getcwd(), 'datasets/imdb.npz'), num_words=10000)
# print(train_data[0])
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(v, k) for (k, v) in word_index.items()]
)
decoded_review = ' '.join(
    [reverse_word_index.get(i-3, '?') for i in train_data[0]]
)


def vectorize_squences(squences, dimension=10000):
    results = np.zeros((len(squences), dimension))
    for i, squence in enumerate(squences):
        results[i, squence] = 1.
    return results


x_train = vectorize_squences(train_data)
x_test = vectorize_squences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 直接引用
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)

# 配置自定义优化器参数，损失和指标
# model.compile(
#     optimizer=optimizers.RMSprop(lr=0.001),
#     loss=losses.binary_crossentropy,
#     metrics=[metrics.accuracy]
# )

# 验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=5,
    batch_size=512,
    validation_data=(x_val, y_val)
)

history_dict = history.history
# print(history_dict.keys())

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

epochs = range(1, len(loss_values)+1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.plot(epochs, acc_values, 'go', label='Training acc')
plt.plot(epochs, val_acc_values, 'g', label='Validation acc')
plt.title('Training and validation loss and training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
