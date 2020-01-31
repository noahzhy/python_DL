import os
from keras.datasets import imdb


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(os.path.join(os.getcwd(), 'datasets/imdb.npz'), num_words=10000)
print(train_data[0])