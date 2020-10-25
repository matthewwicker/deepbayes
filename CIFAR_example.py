import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import deepbayes
import deepbayes.optimizers as optimizers

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train/255.
X_test = X_test/255.
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(4, 4), activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(filters=32, kernel_size=(4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation = 'softmax'))

opt = optimizers.VariationalOnlineGuassNewton()

loss = tf.keras.losses.SparseCategoricalCrossentropy()

bayes_model = opt.compile(model, loss_fn=loss, epochs=45, learning_rate=0.25, inflate_prior=3)

bayes_model.train(X_train, y_train, X_test, y_test)

bayes_model.save("VOGN_CIFAR_Posterior")
