#Author: Matthew Wicker
# Impliments the BayesByBackprop optimizer for BayesKeras

import os
import math
import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from tqdm import tqdm
from tqdm import trange

from . import optimizer 
from . import losses
from abc import ABC, abstractmethod

# A dumb mistake on my part which needs to be factored out
def softplus(x):
     return tf.math.softplus(x)

class VariationalOnlineGuassNewton(optimizer.Optimizer):
    def __init__(self):
        super().__init__()

    # I set default params for each sub-optimizer but none for the super class for
    # pretty obvious reasons
    def compile(self, keras_model, loss_fn, batch_size=64, learning_rate=0.15, decay=0.0,
                      epochs=10, prior_mean=-1, prior_var=-1, **kwargs):
        super().compile(keras_model, loss_fn, batch_size, learning_rate, decay,
                      epochs, prior_mean, prior_var, **kwargs)


        # Now we get into the NoisyAdam specific enrichments to the class
        self.posti_mean = self.model.get_weights()
        self.beta_1 = kwargs.get('beta_1', 0.99)
        self.beta_2 = kwargs.get('beta_2', 0.9999)
        self.lam = kwargs.get('lam', 0.5)
        self.eta = kwargs.get('eta', 0.1)
        self.gam_ex = kwargs.get('gam_ex', 0.1)

        self.m = [0.0 for i in range(len(self.posterior_mean))]

        # To be implimented
        self.robust_train = False
        return self

    def step(self, features, labels, lrate):
        # OPTIMIZATION PARAMETERS:
        alpha = lrate #self.alpha
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        lam = self.lam

        eta = self.eta
        gam_ex = self.gam_ex

        N = float(self.batch_size) # batch size
        gam_in = lam/(N*eta)
        gam = gam_ex + gam_in

        init_weights = []
        for i in range(len(self.posterior_var)):
            sample_var = tf.math.multiply(tf.math.reciprocal(self.posterior_var[i]), (lam/N))
            sample = tf.random.normal(shape=self.posterior_var[i].shape, mean=self.posterior_mean[i], stddev=sample_var)
            init_weights.append(sample)
        self.model.set_weights(init_weights)
        # So, there is definitely a better way to do the below, but what I am
        # trying to accomplish is an all zeros template for the gradient and hessian sums
        weight_gradient = []; weight_hessian = []
        for i in range(len(init_weights)):
            zeros_vec = tf.math.multiply(init_weights[i], 0)
            weight_gradient.append(zeros_vec)
            weight_hessian.append(zeros_vec)
        weight_gradient = np.asarray(weight_gradient); weight_hessian = np.asarray(weight_hessian)
        # Define the GradientTape context
        index = 0
        for feature in features:
            feature = tf.expand_dims(feature, 0)
            #feature = tf.reshape(feature, (-1,tf.shape(feature))) # This may need to be done using TF ops
            with tf.GradientTape(persistent=True) as tape:
                # Get the probabilities
                predictions = self.model(feature)
                # Calculate the loss
                loss = self.loss_func(labels[index], predictions)
            index += 1
            # Get the gradients
            w_gradient = tape.gradient(loss, self.model.trainable_variables)
            weight_gradient += w_gradient
            sq_grad = []
            for i in range(len(w_gradient)):
                sq_grad.append(tf.math.square(w_gradient[i]))
            weight_hessian += sq_grad
        normed_gradient = []
        normed_hessian = []
        for i in range(len(weight_gradient)):
            normed_gradient.append(weight_gradient[i]/float(N))
            normed_hessian.append(weight_hessian[i]/float(N))
        weight_gradient = np.asarray(normed_gradient)
        weight_hessian = np.asarray(normed_hessian)
        v = weight_gradient - (gam_in*np.asarray(self.model.get_weights()))
        self.m = [(beta_1 * self.m[i]) + ((1-beta_1)*v[i]) for i in range(len(self.m))]
        self.m = np.asarray(self.m)
        self.posterior_var = np.asarray(self.posterior_var)
        self.posterior_var = (beta_2*self.posterior_var) + ((1-beta_2)*(weight_hessian))
        m_= self.m/(1-beta_1)
        m_ = [m_[i]/(self.posterior_var[i]+gam) for i in range(len(self.m))]
        self.posterior_mean = self.posterior_mean - (alpha*np.asarray(m_))

        self.train_loss(loss)
        self.train_metric(labels, predictions)
        return self.posterior_mean, self.posterior_var
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        super().train(X_train, y_train, X_test, y_test)
