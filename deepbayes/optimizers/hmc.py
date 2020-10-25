#Author: Matthew Wicker
# Impliments the HamiltonianMonteCarlo optimizer for BayesKeras

import os
import sys
import math
import random
import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import copy
from tqdm import tqdm
from tqdm import trange

from deepbayes.optimizers import optimizer
from deepbayes.optimizers import losses
from deepbayes import analyzers
from abc import ABC, abstractmethod

class HamiltonianMonteCarlo(optimizer.Optimizer):
    def __init__(self):
        super().__init__()

    # I set default params for each sub-optimizer but none for the super class for
    # pretty obvious reasons
    def compile(self, keras_model, loss_fn, batch_size=64, learning_rate=0.15, decay=0.0,
                      epochs=10, prior_mean=-1, prior_var=-1, **kwargs):
        super().compile(keras_model, loss_fn, batch_size, learning_rate, decay,
                      epochs, prior_mean, prior_var, **kwargs)
        # Now we get into the HMC specific enrichments to the class
        self.burn_in =  kwargs.get('burn_in', 10)
        self.mh_burn =  kwargs.get('mh_burn', False)
        self.m_burn =  kwargs.get('b_m', 0.05)
        self.b_steps = kwargs.get('b_steps', 1)
        self.steps = int(kwargs.get('steps', 5))
        print(type(self.posterior_mean))
        for i in range(len(self.posterior_mean)):
            print(type(self.posterior_mean[i]))

        self.preload = kwargs.get('preload', -1)
        if(type(self.preload) != int):
#        if(False):
            print("Preloaded posterior mean weights: %s"%(self.preload))
            self.posterior_mean = np.load(self.preload + "/mean.npy", allow_pickle=True)
            self.posterior_mean = list(self.posterior_mean)
            self.prior_mean = np.load(self.preload + "/mean.npy", allow_pickle=True)
            self.prior_mean = list(self.posterior_mean)


        self.U_metric = tf.keras.metrics.Mean(name="U_metric")
        self.q = self.posterior_mean
        self.current_q = copy.deepcopy(self.q)
        print(self.q)
        self.m = kwargs.get('m', 0.1) #2.0 is optimal for normal training
        self.num_rets = [0] # number of times each iterate in the chain has appeared 
        self.iterate = 0
        self.posterior_samples = []
        return self

    def kinetic_energy(self, p):
        retval = 0.0 # The kinetic energy 
        for i in range(len(p)):
            m_i = (self.learning_rate * self.m) * tf.reduce_sum(tf.ones(p[i].shape))
            retval += tf.math.reduce_sum((p[i]**2)/(2.0*m_i)) #!*! Come back to this
        return retval

    def sample(self, features, labels, lrate):

        self.p = []
        for i in range(len(self.posterior_mean)):
            p_comp = tf.random.normal(shape=self.posterior_mean[i].shape, mean=0, stddev=lrate*self.m)
            self.p.append(p_comp)
        self.current_p = copy.deepcopy(self.p)
        self.current_K = self.kinetic_energy(self.p)

        # half step for the momentum
        self.step(features, labels, lrate/2.0)
        steps = self.b_steps if self.burning_in_chain else self.steps
        for i in trange(steps, desc="Numerical Integration"):
            # step for the position
            for i in range(len(self.q)):
                self.q[i] = self.q[i] - ((lrate/self.m) * self.p[i])
            self.model.set_weights(self.q)

            # step for the momentum
            if(i == steps):
                break
            self.step(features, labels, lrate)
            #self.epsilon += self.max_eps/self.steps

        # half step for the momentum
        self.step(features, labels, lrate/2.0)

        for i in range(len(self.posterior_mean)):
            self.p[i] = tf.math.multiply(self.p[i], -1)
        self.proposed_K = self.kinetic_energy(self.p)
        self.proposed_U = self.evaluate_U(features, labels)


        met_const = tf.math.exp(self.current_U-self.proposed_U+self.current_K-self.proposed_K)
        print("Current  U: ", self.current_U)
        print("Proposed U: ", self.proposed_U)
        print("Current  K: ", self.current_K)
        print("Proposed K: ", self.proposed_K)
        print("METROPOLIS CORRECTION CONSTANT: ", met_const)
        if(self.burning_in_chain==True and self.mh_burn==False):
            met_const = 2.0
        if(random.uniform(0,1) < met_const):
            print("ACCEPTED")
            self.num_rets.append(1); self.iterate += 1
            self.posterior_samples.append(self.q) 
            self.current_q = self.q
            self.current_p = self.p
            self.current_U = self.proposed_U
            #self.current_K = self.proposed_K
        else:
            print("REJECTED")
            self.num_rets[self.iterate] += 1
            self.model.set_weights(self.current_q)
        print("Debug info:")
        print(self.num_rets)

    """
    Step here no longer means what it normally means (in the variational setting). Instead it 
    is a method that updates the momentum of the HMC iterate.
    """
    def step(self, features, labels, lrate):

        # Define the GradientTape context
        with tf.GradientTape(persistent=True) as tape:   # Below we add an extra variable for IBP
            tape.watch(self.posterior_mean) 
            predictions = self.model(features)
            if(self.robust_train == 0):
                loss = losses.normal_potential_energy(labels, predictions, self.prior_mean,
                                               self.prior_var, self.q, self.loss_func)

            elif(int(self.robust_train) == 1):
                # Get the probabilities
                predictions = self.model(features)
                logit_l, logit_u = analyzers.IBP(self, features, self.model.trainable_variables, eps=self.epsilon)
                #!*! TODO: Undo the hardcoding of depth in this function
                v1 = tf.one_hot(labels, depth=10)
                v2 = 1 - tf.one_hot(labels, depth=10)
                worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
                # Now we have the worst case softmax probabilities
                worst_case = self.model.layers[-1].activation(worst_case)
                output = (self.robust_lambda * predictions) + ((1-self.robust_lambda) * worst_case)
                #loss =  self.loss_func(labels, output)
                loss = losses.normal_potential_energy(labels, predictions, self.prior_mean,
                                               self.prior_var, self.q, self.loss_func)

            elif(int(self.robust_train) == 2):
                predictions = self.model(features)
                features_adv = analyzers.FGSM(self, features, self.attack_loss, eps=self.epsilon, num_models=-1)
                # Get the probabilities
                worst_case = self.model(features_adv)
                # Calculate the loss
                output = (self.robust_lambda * predictions) + ((1-self.robust_lambda) * worst_case)
                #loss =  self.loss_func(labels, output)
                loss = losses.normal_potential_energy(labels, predictions, self.prior_mean,
                                               self.prior_var, self.q, self.loss_func)

            elif(int(self.robust_train) == 3):
                output = tf.zeros(predictions.shape)
                self.epsilon = max(0.0001, self.epsilon )
                self.eps_dist= tfp.distributions.Exponential(1.0/self.epsilon)
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
                loss = losses.normal_potential_energy(labels, predictions, self.prior_mean,
                                               self.prior_var, self.q, self.loss_func)     
            elif(int(self.robust_train) == 4):
                predictions = self.model(features)
                self.epsilon = max(0.0001, self.epsilon)
                self.eps_dist = tfp.distributions.Exponential(1.0/self.epsilon)
                output = tf.zeros(predictions.shape)
                for _mc_ in range(self.loss_monte_carlo):
                    #eps = tfp.random.rayleigh([1], scale=self.epsilon)
                    eps = self.eps_dist.sample()
                    features_adv = analyzers.FGSM(self, features, self.attack_loss, eps=self.epsilon, num_models=-1)
                    worst_case = self.model(features_adv)
                    output += (1.0/self.loss_monte_carlo) * worst_case
                loss = losses.normal_potential_energy(labels, predictions, self.prior_mean,
                                               self.prior_var, self.q, self.loss_func)     

        # Get the gradients
        weight_gradient = tape.gradient(loss, self.model.trainable_variables)
        temp_p = []
        for i in range(len(weight_gradient)):
            wg = tf.math.multiply(weight_gradient[i], lrate)
            temp_p.append(tf.math.add(self.p[i], wg)) # maybe come back and make this subtraction
        self.p = np.asarray(temp_p)

        self.train_loss(loss)
        self.train_metric(labels, predictions)
        #self.train_rob(labels, worst_case)
        return self.posterior_mean, self.posterior_var

    def evaluate_U(self, features, labels):
        predictions = self.model(features)
        if(self.robust_train == 1): # We only check with IBP if we need to
            logit_l, logit_u = analyzers.IBP(self, features, self.model.get_weights(), self.epsilon)
            #logit_l, logit_u = analyzers.IBP(self, features, self.model.trainable_variables, 0.0)
            v1 = tf.one_hot(labels, depth=10)
            v2 = 1 - tf.one_hot(labels, depth=10)
            worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
            worst_case = self.model.layers[-1].activation(worst_case)
            v_loss = losses.robust_potential_energy(labels, predictions, self.prior_mean,
                                               self.prior_var, self.q, self.loss_func,
                                               worst_case, self.robust_lambda)
            #v_loss = self.loss_func(labels, predictions, worst_case, self.robust_lambda)
            self.extra_metric(labels, worst_case)
        elif(self.robust_train == 2):
            v_loss = self.loss_func(labels, predictions, predictions, self.robust_lambda)
            worst_case = predictions
        else:
            v_loss = losses.normal_potential_energy(labels, predictions, self.prior_mean,
                                               self.prior_var, self.q, self.loss_func)
            #v_loss = self.loss_func(labels, predictions)
            worst_case = predictions
        self.U_metric(v_loss)
        res = self.U_metric.result()
        self.U_metric.reset_states()
        return res

    def train(self, X_train, y_train, X_test=None, y_test=None):
        # We generate this only to speed up parallel computations of the test statistics
        # if there are any to compute
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(256)
    
        # Open with a user warning about potential memory constraints of their process:
        s = (self.burn_in + self.epochs) * (sys.getsizeof(self.model.get_weights())/1000000)
        warn = """BayesKeras Warning: HMC is a memory hungry optimizer. 
         Given you system and parameters of this training run,
         we expect your system to need %s MB of available memory"""%(s) 
        print(warn)

        if(self.robust_linear):
            self.max_eps = self.epsilon
            self.epsilon = 0.0

        self.current_U = self.evaluate_U(X_train, y_train)
        self.burning_in_chain = True
        temp_m = self.m; self.m = self.m_burn
        for iter in range(self.burn_in):
            self.sample(X_train, y_train, self.learning_rate)
            for test_features, test_labels in test_ds:
                self.model_validate(test_features, test_labels)
            # Grab the results
            (loss, acc) = self.train_loss.result(), self.train_metric.result()
            (val_loss, val_acc) = self.valid_loss.result(), self.valid_metric.result()
            self.logging(loss, acc, val_loss, val_acc, iter)
            print("Still in Burn-In state of Markov chain.")

            # Clear the current state of the metrics
            self.train_loss.reset_states(), self.train_metric.reset_states()
            self.valid_loss.reset_states(), self.valid_metric.reset_states()
            self.extra_metric.reset_states()
            if(self.robust_train != 0 and self.robust_linear == True):
                self.epsilon += self.max_eps/self.burn_in
            if(not self.mh_burn):
                self.num_rets = [0] # number of times each iterate in the chain has appeared
                self.iterate = 0
                self.posterior_samples = []
        self.burning_in_chain = False
        self.m = temp_m
        if(self.robust_linear):
            self.epsilon = self.max_eps

        # We use that to update the chain and start the burn in but then
        # we reset all of the parameters that keep track of our approximate
        # posterior samples
        self.num_rets = [0] # number of times each iterate in the chain has appeared
        self.iterate = 0
        self.posterior_samples = []
        self._learning_rate = self.learning_rate
        for iter in range(self.epochs):
            self.learning_rate = self._learning_rate * (1 / (1 + self.decay * iter))
            self.sample(X_train, y_train, self.learning_rate)
            for test_features, test_labels in test_ds:
                self.model_validate(test_features, test_labels)
            # Grab the results
            (loss, acc) = self.train_loss.result(), self.train_metric.result()
            (val_loss, val_acc) = self.valid_loss.result(), self.valid_metric.result()
            self.logging(loss, acc, val_loss, val_acc, iter)
            print("Out of Burn-In state. Generating samples from the chain")

            # Clear the current state of the metrics
            self.train_loss.reset_states(), self.train_metric.reset_states()
            self.valid_loss.reset_states(), self.valid_metric.reset_states()
            self.extra_metric.reset_states()
            #self.epsilon += self.max_eps/self.epochs

        # Post processing for the approximate posterior distribution
    def save(self, path):
        if(self.num_rets[0] == 0):
            self.num_rets = self.num_rets[1:]
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path+"/samples"):
            os.makedirs(path+"/samples")
        np.save(path+"/mean", np.asarray(self.posterior_mean))
        for i in range(len(self.posterior_samples)):
            np.save(path+"/samples/sample_%s"%(i), np.asarray(self.posterior_samples[i]))
        self.model.save(path+'/model.h5')
        np.save(path+"/freq",np.asarray(self.num_rets))
        model_json = self.model.to_json()
        with open(path+"/arch.json", "w") as json_file:
            json_file.write(model_json)

            
