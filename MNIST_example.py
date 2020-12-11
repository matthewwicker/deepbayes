import deepbayes
import deepbayes.optimizers as optimizers

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train/255.
X_test = X_test/255.
X_train = X_train.astype("float32").reshape(-1, 28*28)
X_test = X_test.astype("float32").reshape(-1, 28* 28)


opt = optimizers.LangevinMonteCarlo()

likelihood = tf.keras.losses.SparseCategoricalCrossentropy()

model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(1, 28*28)))
model.add(Dense(10, activation="softmax"))

bayes_model = opt.compile(model, loss_fn=likelihood, 
                          epochs=5, learning_rate=0.25, 
                          inflate_prior=2.0)

bayes_model.train(X_train, y_train, X_test, y_test)

bayes_model.save("VOGN_Model")
