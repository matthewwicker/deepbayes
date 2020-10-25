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

class Adam(optimizer.Optimizer):
    def __init__(self):
        super().__init__()

    # I set default params for each sub-optimizer but none for the super class for
    # pretty obvious reasons
    def compile(self, keras_model, loss_fn, batch_size=64, learning_rate=0.15, decay=0.0,
                      epochs=10, prior_mean=-1, prior_var=-1, **kwargs):
        super().compile(keras_model, loss_fn, batch_size, learning_rate, decay,
                      epochs, prior_mean, prior_var, **kwargs)


        # Now we get into the NoisyAdam specific enrichments to the class
        self.beta_1 = kwargs.get('beta_1', 0.99)
        self.beta_2 = kwargs.get('beta_2', 0.9999)
        self.lam = kwargs.get('lam', 0.5)
        self.m = [0.0 for i in range(len(self.posterior_mean))]
        self.posterior_var = [tf.zeros(i.shape) for i in self.posterior_mean]

        return self

    def step(self, features, labels, lrate):
        alpha = lrate
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        lam = self.lam

        posti_var = self.posterior_var
        posti_mean = self.posterior_mean

        N = float(self.batch_size) # batch size
        
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
                loss =  self.loss_func(labels, predictions, worst_case, self.robust_lambda)
                #self.train_rob(labels, worst_case)
            elif(int(self.robust_train) == 2):
                features_adv = analyzers.FGSM(self, features, self.attack_loss, eps=self.epsilon, num_models=-1)
                # Get the probabilities
                worst_case = self.model(features_adv)
                # Calculate the loss
                loss = self.loss_func(labels, predictions, worst_case, self.robust_lambda)

        weight_gradient = tape.gradient(loss, self.model.trainable_variables)
        g = np.asarray(weight_gradient)
        
        sq_grad = []
        for i in range(len(weight_gradient)):
            sq_grad.append(tf.math.multiply(weight_gradient[i],weight_gradient[i]))
            self.m[i] = (beta_1*self.m[i]) + ((1-beta_1)*(g[i]+((lam*posti_mean[i])/N)))
            posti_var[i] = (beta_2*posti_var[i]) + ((1-beta_2)*(sq_grad[i]))
            
        sq_grad = np.asarray(sq_grad); self.m = np.asarray(self.m)
        posti_var = np.asarray(posti_var) 
        
        for i in range(len(weight_gradient)):
            m_ =  self.m[i]/(1-beta_1) 
            s_ = np.sqrt(posti_var[i]) + lam/N 
            posti_mean[i] = posti_mean[i] - (alpha*(m_/s_))
        
        self.model.set_weights(posti_mean)
        self.train_loss(loss)
        self.train_metric(labels, predictions)
        return posti_mean, posti_var

    def old_step(self, features, labels, lrate):
        # OPTIMIZATION PARAMETERS:
        alpha = lrate #self.alpha
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        lam = self.lam

        posti_mean = self.model.get_weights()
        self.model.set_weights(posti_mean)
        
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
                loss =  self.loss_func(labels, predictions, worst_case, self.robust_lambda)
                #self.train_rob(labels, worst_case)
            elif(int(self.robust_train) == 2):
                features_adv = analyzers.FGSM(self, features, self.attack_loss, eps=self.epsilon, num_models=-1)
                # Get the probabilities
                worst_case = self.model(features_adv)
                # Calculate the loss
                loss = self.loss_func(labels, predictions, worst_case, self.robust_lambda)
                
        weight_gradient = tape.gradient(loss, self.model.trainable_variables)
        g = np.asarray(weight_gradient)
        #print(g)
        sq_grad = []
        for i in range(len(weight_gradient)):
            sq_grad.append(tf.math.multiply(weight_gradient[i],weight_gradient[i]))
            self.m[i] = (beta_1*self.m[i]) + ((1-beta_1)*(g[i]))
            self.posterior_var[i] = (beta_2*self.posterior_var[i]) + ((1-beta_2)*(sq_grad[i]))
              
        #print("sq: ", sq_grad)            
        sq_grad = np.asarray(sq_grad); self.m = np.asarray(self.m)
        self.posterior_var = np.asarray(self.posterior_var) 
        
        for i in range(len(weight_gradient)):
            m_ =  self.m[i]/(1-beta_1) 
            s_ = np.sqrt(self.posterior_var[i])
            #print(alpha*(m_/s_))
            self.posterior_mean[i] = self.posterior_mean[i] - (alpha*(m_/s_))

        #self.model.set_weights(self.posterior_mean)
        self.train_loss(loss)
        self.train_metric(labels, predictions)
 
        return self.posterior_mean, self.posterior_var
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        super().train(X_train, y_train, X_test, y_test)

    def sample(self):
        return self.model.get_weights()
