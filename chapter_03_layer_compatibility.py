from keras import models
from keras import layers


models = models.Sequential()
models.add(layers.Dense(32, input_shape=(784,)))
models.add(layers.Dense(32))