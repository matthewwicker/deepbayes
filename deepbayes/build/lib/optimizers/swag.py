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

from deepbayes import analyzers
from deepbayes.optimizers import optimizer 
from deepbayes.optimizers import losses
from abc import ABC, abstractmethod

# A dumb mistake on my part which needs to be factored out
def softplus(x):
     return tf.math.softplus(x)

class StochasticWeightAveragingGaussian(optimizer.Optimizer):
    def __init__(self):
        super().__init__()

    # I set default params for each sub-optimizer but none for the super class for
    # pretty obvious reasons
    def compile(self, keras_model, loss_fn, batch_size=64, learning_rate=0.15, decay=0.0,
                      epochs=10, prior_mean=-1, prior_var=-1, **kwargs):
        super().compile(keras_model, loss_fn, batch_size, learning_rate, decay,
                      epochs, prior_mean, prior_var, **kwargs)


        # Now we get into the SGD specific enrichments to the class
        self.posterior_var = [tf.zeros(self.posterior_var[i].shape) for i in range(len(self.posterior_var))]
        self.record_epochs = int(kwargs.get('record_epochs', int(2)))
        self.full_covar = int(kwargs.get('full_covar', False))
        self.inflate_prior /= 100
        self.expl_lr = float(kwargs.get('expl_lr', self.inflate_prior*(learning_rate/5.0)))
        
        self.weights_stack = []
        self.record = False
    
        return self

    def step(self, features, labels, lrate):
        # Define the GradientTape context
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.posterior_mean) 

            predictions = self.model(features)

            if(not self.robust_train):
                worst_case = predictions
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
                #self.train_rob(labels, worst_case)

            elif(int(self.robust_train) == 3):
                output = tf.zeros(predictions.shape)
                self.epsilon = max(0.0001, self.epsilon)
                self.eps_dist = tfp.distributions.Exponential(1.0/float(self.epsilon))
                for _mc_ in range(self.loss_monte_carlo):
                    eps = self.eps_dist.sample()
                    #eps = tfp.random.rayleigh([1], scale=self.epsilon)
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
        weights = self.model.get_weights()
        new_weights = []
        for i in range(len(weight_gradient)):
            wg = tf.math.multiply(weight_gradient[i], lrate)
            m = tf.math.subtract(weights[i], wg)
            new_weights.append(m)

        if(self.record == True):
            self.weights_stack.append(new_weights)

        self.model.set_weights(new_weights)
        self.posterior_mean = new_weights

        self.train_loss(loss)
        self.train_metric(labels, predictions)
        #self.train_rob(labels, worst_case)
        return self.posterior_mean, self.posterior_var
    
    
    def train(self, X_train, y_train, X_test=None, y_test=None, **kwargs):

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(100).batch(self.batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(self.batch_size)

        if(self.robust_linear):
            self.max_eps = self.epsilon
            self.epsilon = 0.0

        lr = self.learning_rate; decay = self.decay
        for epoch in range(self.epochs + self.record_epochs):
            lrate = self.learning_rate * (1 / (1 + self.decay * epoch))
            if(epoch > self.epochs):
                self.record = True
                lrate = self.expl_lr

            # Run the model through train and test sets respectively
            for (features, labels) in tqdm(train_ds):
                self.posterior, self.posterior_var = self.step(features, labels, lrate)
            for test_features, test_labels in test_ds:
                self.model_validate(test_features, test_labels)

            # Grab the results
            (loss, acc) = self.train_loss.result(), self.train_metric.result()
            (val_loss, val_acc) = self.valid_loss.result(), self.valid_metric.result()

            # Clear the current state of the metrics
            self.train_loss.reset_states(), self.train_metric.reset_states()
            self.valid_loss.reset_states(), self.valid_metric.reset_states()

            self.logging(loss, acc, val_loss, val_acc, epoch)

            if(self.robust_linear):
                self.epsilon += self.max_eps/self.epochs

    def get_posterior(self):
        ws = self.weights_stack[0:250]
        ws = np.swapaxes(ws, 0, 1)
        mean, var = [], []
        for i in ws:
            #print(type(i), i[0].shape)
            mean.append(tf.math.reduce_mean(tf.stack(i), axis=0))
            if(not self.full_covar):
                var.append(tf.math.reduce_std(tf.stack(i), axis=0))
            else:
                var.append(tfp.stats.covariance(tf.stack(i)))
        self.posterior_var = var
        self.posterior_mean = mean

    def save(self, path):
        self.get_posterior()
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path+"/mean", np.asarray(self.posterior_mean, dtype=object))
        np.save(path+"/var", np.asarray(self.posterior_var, dtype=object))
        self.model.save(path+'/model.h5')
        model_json = self.model.to_json()
        with open(path+"/arch.json", "w") as json_file:
            json_file.write(model_json)
