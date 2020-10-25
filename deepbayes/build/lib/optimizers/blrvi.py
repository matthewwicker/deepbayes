#Author: Matthew Wicker
# Impliments the Bayes Learning Rule version of VOGN optimizer for deepbayes

import os
import math
import copy
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
        self.beta_1 = kwargs.get('beta_1', 0.999)
        self.beta_2 = kwargs.get('beta_2', 0.9999)
        self.lam = kwargs.get('lam', 1.0)

        self.m = [0.0 for i in range(len(self.posterior_mean))]

        # To be implimented
        return self

    def step(self, features, labels, lrate):
        # OPTIMIZATION PARAMETERS:
        alpha = lrate #self.alpha
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        lam = self.lam

        N = 60000 #float(self.batch_size) # batch size

        self.posterior_mean = self.model.get_weights()

        init_weights = []
        for i in range(len(self.posterior_mean)):
            var = tf.math.add(tf.math.sqrt(N*self.posterior_var[i]), lam)
            var = tf.math.reciprocal(var)
            sample = tf.random.normal(shape=self.posterior_var[i].shape, mean=0, stddev=1.0)
            sample = tf.math.multiply(var, sample)
            sample = tf.math.add(self.posterior_mean[i], sample)
            init_weights.append(sample)
        
        self.model.set_weights(np.asarray(init_weights))
        
        with tf.GradientTape(persistent=True) as tape:
            # Get the probabilities
            predictions = self.model(features)
            # Calculate the loss
            if(int(self.robust_train) == 0):
                loss = self.loss_func(labels, predictions)

            elif(int(self.robust_train) == 1):
                logit_l, logit_u = analyzers.IBP(self, features, self.model.trainable_variables, eps=self.epsilon)
                v1 = tf.one_hot(labels, depth=10)
                v2 = 1 - tf.one_hot(labels, depth=10)
                worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
                worst_case = self.model.layers[-1].activation(worst_case)
                output = (self.robust_lambda * predictions) + ((1-self.robust_lambda) * worst_case)
                loss =  self.loss_func(labels, output)

            elif(int(self.robust_train) == 2):
                features_adv = analyzers.PGD(self, features, self.attack_loss, eps=self.epsilon, num_models=-1)
                worst_case = self.model(features_adv)
                output = (self.robust_lambda * predictions) + ((1-self.robust_lambda) * worst_case)
                loss =  self.loss_func(labels, output)

            elif(int(self.robust_train) == 3):
                output = tf.zeros(predictions.shape)
                self.epsilon = max(0.0001, self.epsilon)
                self.eps_dist = tfp.distributions.Exponential(1.0/self.epsilon)
                for _mc_ in range(self.loss_monte_carlo):
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
       
        #if(int(self.robust_train) == 1):
#        print(g)
        # We need to process the gradient according to the reparameterization given by Khan (2002.10060)
        g_mu = []
        g_s = []
        m_hat = []
        s_hat = []
        t = self.learning_rate
        for i in range(len(g)):
            # Appropriately scaled updates to the gradients Khan (2002.10060)[ICLR2020]
            g_mu.append((self.lam/60000)*self.posterior_mean[i] + g[i])
            g_s_comp2 = tf.math.multiply((60000*self.posterior_var[i]), (init_weights[i] - self.posterior_mean[i]))
            g_s_comp2 = tf.math.multiply(g_s_comp2, g[i])
            g_s.append((self.lam/60000) - self.posterior_var[i] + g_s_comp2)
            # Standard momentum updtae
            self.m[i] = (beta_1*self.m[i]) + ((1-beta_1)*(g_mu[i]))
            m_hat.append(self.m[i]/(1-beta_1))
            s_hat.append(self.posterior_var[i]/(1-beta_2))

        # Apply the effects from the updates
        for i in range(len(g)):
            self.posterior_mean[i] = self.posterior_mean[i] - t*(m_hat[i]/s_hat[i])
            comp_1 = (0.5 * ((1-beta_2)**2) * g_s[i])
            recip = tf.math.multiply(tf.math.reciprocal(self.posterior_var[i]), g_s[i])
            self.posterior_var[i] = self.posterior_var[i] + tf.math.multiply(comp_1, recip) 

        self.model.set_weights(self.posterior_mean)
        self.train_loss(loss)
        self.train_metric(labels, predictions)
        return self.posterior_mean, self.posterior_var
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        super().train(X_train, y_train, X_test, y_test)

    def save(self, path):
        save_var = []
        for i in range(len(self.posterior_var)):
            var = tf.math.reciprocal(tf.math.sqrt(self.N*self.posterior_var[i]))
            save_var.append(var)
        temp_var = copy.deepcopy(self.posterior_var)
        self.posterior_var = save_var
        super().save(path)
        self.posterior_var = temp_var

    def sample(self):
        sampled_weights = []
        for i in range(len(self.posterior_mean)):
            var = tf.math.reciprocal(tf.math.sqrt(self.N*self.posterior_var[i]))
            sampled_weights.append(np.random.normal(loc=self.posterior_mean[i], 
                                                    scale=var[i]))
        return sampled_weights
