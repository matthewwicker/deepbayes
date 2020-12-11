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

from deepbayes.optimizers import optimizer
from deepbayes.optimizers import losses
from deepbayes import analyzers
from abc import ABC, abstractmethod

# A dumb mistake on my part which needs to be factored out
def softplus(x):
     return tf.math.softplus(x)

class NoisyAdam(optimizer.Optimizer):
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

        self.m = [0.0 for i in range(len(self.posterior_mean))]

        return self

    def step(self, features, labels, lrate):
        # OPTIMIZATION PARAMETERS:
        alpha = lrate #self.alpha
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        lam = self.lam

        N = self.N #60000 
        #N = float(self.batch_size) # batch size

        self.posterior_mean = self.model.get_weights()

        v1 = tf.one_hot(labels, depth=10)
        v2 = 1 - tf.one_hot(labels, depth=10)

        init_weights = []
        for i in range(len(self.posterior_mean)):
            var = tf.math.add(tf.math.sqrt(N*self.posterior_var[i]), lam)
            var = tf.math.reciprocal(var)
            sample = tf.random.normal(shape=self.posterior_var[i].shape, mean=0, stddev=1.0)
            sample = tf.math.multiply(var, sample)
            sample = tf.math.add(self.posterior_mean[i], sample)
            init_weights.append(sample)
        
        self.model.set_weights(init_weights)
        
        with tf.GradientTape(persistent=True) as tape:
            # Get the probabilities
            predictions = self.model(features)
            # Calculate the loss
            if(int(self.robust_train) == 0):
                loss = self.loss_func(labels, predictions)

            elif(int(self.robust_train) == 1):
                logit_l, logit_u = analyzers.IBP(self, features, self.model.trainable_variables, eps=self.epsilon)
                worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
                worst_case = self.model.layers[-1].activation(worst_case)
                output = (self.robust_lambda * predictions) + ((1-self.robust_lambda) * worst_case)
                loss =  self.loss_func(labels, output)
       
            elif(int(self.robust_train) == 2):
                features_adv = analyzers.PGD(self, features, self.attack_loss, eps=self.epsilon, num_models=-1)
                # Get the probabilities
                worst_case = self.model(features_adv)
                # Calculate the loss
                output = (self.robust_lambda * predictions) + ((1-self.robust_lambda) * worst_case)
                loss =  self.loss_func(labels, output)

            elif(int(self.robust_train) == 3):
                output = tf.zeros(predictions.shape)
                self.epsilon = max(0.0001, self.epsilon)
                self.eps_dist = tfp.distributions.Exponential(1.0/self.epsilon)
                for _mc_ in range(self.loss_monte_carlo):
                    #eps = tfp.random.rayleigh([1], scale=self.epsilon)
                    eps = self.eps_dist.sample()
                    logit_l, logit_u = analyzers.IBP(self, features, self.model.trainable_variables, eps=eps)
                    v1 = tf.one_hot(labels, depth=10)
                    v2 = 1 - tf.one_hot(labels, depth=10)
                    v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
                    worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
                    worst_case = self.model.layers[-1].activation(worst_case)
                    one_hot_cls = tf.one_hot(labels, depth=10)
                    output += (1.0/self.loss_monte_carlo) * worst_case
                loss = self.loss_func(labels, output)

            elif(int(self.robust_train) == 4):
                output = tf.zeros(predictions.shape)
                self.epsilon = max(0.0001, self.epsilon)
                self.eps_dist = tfp.distributions.Exponential(1.0/self.epsilon)
                for _mc_ in range(self.loss_monte_carlo):
                    #eps = tfp.random.rayleigh([1], scale=self.epsilon)
                    eps = self.eps_dist.sample()
                    features_adv = analyzers.FGSM(self, features, self.attack_loss, eps=self.epsilon, num_models=-1)
                    worst_case = self.model(features_adv)
                    output += (1.0/self.loss_monte_carlo) * worst_case
                loss = self.loss_func(labels, output)

        weight_gradient = tape.gradient(loss, self.model.trainable_variables)
        g = np.asarray(weight_gradient)

        sq_grad = []
        for i in range(len(weight_gradient)):
            sq_grad.append(tf.math.multiply(weight_gradient[i],weight_gradient[i]))
            self.m[i] = (beta_1*self.m[i]) + ((1-beta_1)*(g[i]+((lam*self.posterior_mean[i])/N)))
            self.posterior_var[i] = (beta_2*self.posterior_var[i]) + ((1-beta_2)*(sq_grad[i]))
              
        #print("sq: ", sq_grad)            
        sq_grad = np.asarray(sq_grad); self.m = np.asarray(self.m)
        self.posterior_var = np.asarray(self.posterior_var) 
        
        for i in range(len(weight_gradient)):
            m_ =  self.m[i]/(1-beta_1) 
            s_ = np.sqrt(self.posterior_var[i]) + lam/N 
            self.posterior_mean[i] = self.posterior_mean[i] - (alpha*(m_/s_))

        self.model.set_weights(self.posterior_mean)
        self.train_loss(loss)
        self.train_metric(labels, predictions)
        #self.posterior_mean = posti_mean
        #self.posterior_var = posti_var
        return self.posterior_mean, self.posterior_var
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        super().train(X_train, y_train, X_test, y_test)
