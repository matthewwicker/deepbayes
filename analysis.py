
import deepbayes
from deepbayes import PosteriorModel
from deepbayes import analyzers

from tqdm import trange
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train/255.
X_test = X_test/255.
X_train = X_train.astype("float32").reshape(-1, 28*28)
X_test = X_test.astype("float32").reshape(-1, 28* 28)


model = PosteriorModel("VOGN_Posterior")

loss = tf.keras.losses.SparseCategoricalCrossentropy()

num_images = 500

accuracy = tf.keras.metrics.Accuracy()
preds = model.predict(X_test[0:500]) #np.argmax(model.predict(np.asarray(adv)), axis=1)
accuracy.update_state(np.argmax(preds, axis=1), y_test[0:500])
fgsm = accuracy.result()
print("Accuracy: ", accuracy.result())

accuracy = tf.keras.metrics.Accuracy()
adv = analyzers.FGSM(model, X_test[0:500], eps=0.1, loss_fn=loss, num_models=10)
preds = model.predict(adv) #np.argmax(model.predict(np.asarray(adv)), axis=1)
accuracy.update_state(np.argmax(preds, axis=1), y_test[0:500])
fgsm = accuracy.result()
print("FGSM Robustness: ", accuracy.result())

