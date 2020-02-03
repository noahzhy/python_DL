import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical

from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    os.path.join(os.getcwd(), 'datasets/reuters.npz'),
    num_words=10000
)

print(len(train_data))

word_index = reuters.get_word_index(
    os.path.join(os.getcwd(), 'datasets/reuters_word_index.json')
)
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

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(
    optimizer=optimizers.RMSprop(),
    loss=losses.categorical_crossentropy,
    metrics=[metrics.categorical_accuracy]
)

# 验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=9,
    batch_size=512,
    validation_data=(x_val, y_val)
)

history_dict = history.history
print(history_dict.keys())

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['categorical_accuracy']
val_acc_values = history_dict['val_categorical_accuracy']

# 用于展示数据
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

results = model.evaluate(x_test, y_test)
print(results)
# print(model.predict(x_test))
