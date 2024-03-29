#Author: Matthew Wicker
# Impliments the DeepEnsembles optimizer for BayesKeras

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

class DeepEnsemble(optimizer.Optimizer):
    def __init__(self):
        super().__init__()

    # I set default params for each sub-optimizer but none for the super class for
    # pretty obvious reasons
    def compile(self, keras_model, loss_fn, batch_size=64, learning_rate=0.15, decay=0.0,
                      epochs=10, prior_mean=-1, prior_var=-1, **kwargs):
        super().compile(keras_model, loss_fn, batch_size, learning_rate, decay,
                      epochs, prior_mean, prior_var, **kwargs)
        # Now we get into the DeepEnsembles specific enrichments to the class
        self.num_models =  kwargs.get('num_models', 10)
        #self.posterior_var = [tf.zeros(self.posterior_var[i].shape) for i in range(len(self.posterior_var))]
        
        return self

    def step(self, features, labels, lrate):
        # Define the GradientTape context
        with tf.GradientTape(persistent=True) as tape:   # Below we add an extra variable for IBP
            tape.watch(self.posterior_mean) 
            predictions = self.model(features)
            if(self.robust_train == 0):
                worst_case = predictions 
                loss = self.loss_func(labels, predictions)

            elif(int(self.robust_train) == 1):
                predictions = self.model(features)
                logit_l, logit_u = analyzers.IBP(self, features, self.model.trainable_variables, eps=self.epsilon)
                v1 = tf.one_hot(labels, depth=self.classes); v1 = tf.cast(v1, dtype=tf.float32)
                v2 = 1 - tf.one_hot(labels, depth=self.classes); v2 = tf.cast(v2, dtype=tf.float32)
                logit_l, logit_u = tf.cast(logit_l, dtype=tf.float32), tf.cast(logit_u, dtype=tf.float32) 
                worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
                worst_case = self.model.layers[-1].activation(worst_case)
                output = (self.robust_lambda * predictions) + ((1-self.robust_lambda) * worst_case)
                loss =  self.loss_func(labels, output)

            elif(int(self.robust_train) == 2):
                predictions = self.model(features)
                features_adv = analyzers.FGSM(self, features, self.attack_loss, eps=self.epsilon, num_models=-1)
                worst_case = self.model(features_adv)
                output = (self.robust_lambda * predictions) + ((1-self.robust_lambda) * worst_case)
                loss =  self.loss_func(labels, output)
                #self.train_rob(labels, worst_case)

            elif(int(self.robust_train) == 3):
                output = tf.zeros(predictions.shape)
                self.epsilon = max(0.0001, self.epsilon)
                self.eps_dist = tfp.distributions.Exponential(1.0/self.epsilon)
                for _mc_ in range(self.loss_monte_carlo):
                    eps = tfp.random.rayleigh([1], scale=self.epsilon/2.0)
                    logit_l, logit_u = analyzers.IBP(self, features, self.model.trainable_variables, eps=self.epsilon)
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
                self.eps_dist = tfp.distributions.Exponential(1.0/float(self.epsilon))
                for _mc_ in range(self.loss_monte_carlo):
                    #eps = tfp.random.rayleigh([1], scale=self.epsilon)
                    eps = self.eps_dist.sample()
                    features_adv = analyzers.FGSM(self, features, self.attack_loss, eps=self.epsilon, num_models=-1)
                    worst_case = self.model(features_adv)
                    output += (1.0/self.loss_monte_carlo) * worst_case
                loss = self.loss_func(labels, output)

        # Get the gradients
        weight_gradient = tape.gradient(loss, self.model.trainable_variables)
#        print(weight_gradient)
        weights = self.model.get_weights()
        new_weights = []
        for i in range(len(weight_gradient)):
            wg = tf.math.multiply(weight_gradient[i], lrate)
            m = tf.math.subtract(weights[i], wg)
            new_weights.append(m)

        self.model.set_weights(new_weights)
        self.posterior_mean = new_weights

        self.train_loss(loss)
        self.train_metric(labels, predictions)
        #self.train_rob(labels, worst_case)
        return self.posterior_mean, self.posterior_var

    def reset_weights(self, model):
        sampled_weights = []
        for i in range(len(self.posterior_mean)):
            sampled_weights.append(np.random.normal(loc=self.posterior_mean[i],
                                                scale=inflate*self.posterior_var[i]))
        self.set_weights(sampled_weights)      
        
    def train(self, X_train, y_train, X_test=None, y_test=None):
        self.posterior_samples = []
        for i in range(self.num_models):
            super().train(X_train, y_train, X_test, y_test)
            self.posterior_samples.append(self.model.get_weights())
            self.reset_weights(self.model)
            
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path+"/samples"):
            os.makedirs(path+"/samples")
        np.save(path+"/mean", np.asarray(self.posterior_mean))
        for i in range(len(self.posterior_samples)):
            np.save(path+"/samples/sample_%s"%(i), np.asarray(self.posterior_samples[i]))
        self.model.save(path+'/model.h5')
        freq =  np.save(np.asarray([1 for i in range(self.num_models)]))
        np.save(path+"/freq",np.asarray(freq))
        model_json = self.model.to_json()
        with open(path+"/arch.json", "w") as json_file:
            json_file.write(model_json)
            
            