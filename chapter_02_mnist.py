from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(len(train_images))