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

class SGHamiltonianMonteCarlo(optimizer.Optimizer):
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
        self.m_burn =  kwargs.get('b_m', self.learning_rate)
        self.b_steps = kwargs.get('b_steps', 1)
        self.steps = int(kwargs.get('steps', 5))
        self.preload = kwargs.get('preload', -1)
        self.max_batches = kwargs.get('max_batches', 20)
        if(type(self.preload) != int):
            print("Preloaded posterior mean weights: %s"%(self.preload))
            self.posterior_mean = np.load(self.preload + "/mean.npy", allow_pickle=True)
            self.posterior_mean = list(self.posterior_mean)
            self.prior_mean = np.load(self.preload + "/mean.npy", allow_pickle=True)
            self.prior_mean = list(self.posterior_mean)


        self.U_metric = tf.keras.metrics.Mean(name="U_metric")
        self.q = self.posterior_mean
        self.current_q = copy.deepcopy(self.q)
        self.m = kwargs.get('m', 0.1) 
        
        # Maintain chain samples
        self.num_rets = [0] 
        self.iterate = 0
        self.posterior_samples = []
        
        # In case we restart the chain 
        self.global_rets = [0]
        self.global_iterate = 0
        self.global_poseterior_samples = []
        return self

    def kinetic_energy(self, p):
        retval = 0.0 # The kinetic energy 
        for i in range(len(p)):
            m_i = (self.learning_rate * self.m) * tf.reduce_sum(tf.ones(p[i].shape))
            retval += tf.math.reduce_sum((p[i]**2)/(2.0*m_i)) #!*! Come back to this
        return retval

    def sample(self, train_ds, lrate, features_l=None, features_u=None, constraint=False):

        self.p = []
        for i in range(len(self.posterior_mean)):
            p_comp = tf.random.normal(shape=self.posterior_mean[i].shape, mean=0, stddev=lrate*self.m)
            self.p.append(p_comp)
        self.current_p = copy.deepcopy(self.p)
        self.current_K = self.kinetic_energy(self.p)

        # half step for the momentum
        self.step(train_ds, lrate/2.0)
        
        steps = self.b_steps if self.burning_in_chain else self.steps
        for i in trange(steps, desc="Numerical Integration"):
            # step for the position
            for i in range(len(self.q)):
                self.q[i] = self.q[i] - ((lrate/self.m) * self.p[i])
            self.model.set_weights(self.q)

            # step for the momentum
            if(i == steps):
                break
                
            self.step(train_ds, lrate)

        # half step for the momentum
        self.step(train_ds, lrate/2.0)

        for i in range(len(self.posterior_mean)):
            self.p[i] = tf.math.multiply(self.p[i], -1)
            
        self.proposed_K = self.kinetic_energy(self.p)
        self.proposed_U = self.evaluate_U(self.X_train, self.y_train)

        met_const = tf.math.exp(self.current_U-self.proposed_U+self.current_K-self.proposed_K)
        print("Current  U: ", self.current_U)
        print("Proposed U: ", self.proposed_U)
        print("METROPOLIS CORRECTION CONSTANT: ", met_const)
        if(self.burning_in_chain==True and self.mh_burn==False):
            met_const = 2.0
        if(random.uniform(0,1) < met_const):
            print("ACCEPTED")
            self.num_rets.append(1); self.iterate += 1
            self.posterior_samples.append(copy.deepcopy(self.q))
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
    def step(self, train_ds, lrate):
        num_complete = 0
        for features, labels in train_ds:
            # Define the GradientTape context
            with tf.GradientTape(persistent=True) as tape:   # Below we add an extra variable for IBP
                tape.watch(self.posterior_mean) 
                predictions = self.model(features)
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
            try:
                self.train_metric(labels, predictions)
            except:
                self.train_metric(labels, np.argmax(predictions, axis=1))
            num_complete += 1
            if(num_complete > self.max_batches):
                break
        return self.posterior_mean, self.posterior_var
    
    
        """
    Step here no longer means what it normally means (in the variational setting). Instead it 
    is a method that updates the momentum of the HMC iterate.
    """
    def constraint_step(self, features, labels, lrate, features_l, features_u):

        # Define the GradientTape context
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.posterior_mean) 
            # Standard forward pass through the neural network
            predictions = self.model(features)
            
            # Convex relaxation pass of the whole data through the neural network model
            logit_l, logit_u = analyzers.pIBP(self, features_l, features_u, self.model.trainable_variables)
            #try:
            #    v1 = tf.one_hot(labels, depth=self.classes); v1 = tf.cast(v1, dtype=tf.float32)
            #    v2 = 1 - tf.one_hot(labels, depth=self.classes); v2 = tf.cast(v2, dtype=tf.float32)
            #except:
            v1 = labels; v1 = tf.cast(v1, dtype=tf.float32)
            v2 = 1 - labels; v2 = tf.cast(v2, dtype=tf.float32)
            logit_l, logit_u = tf.cast(logit_l, dtype=tf.float32), tf.cast(logit_u, dtype=tf.float32) 
            worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
            predictions = self.model.layers[-1].activation(worst_case)
            output = (self.robust_lambda * predictions) + ((1-self.robust_lambda) * worst_case)
            loss = losses.normal_potential_energy(labels, output, self.prior_mean,
                                               self.prior_var, self.q, self.loss_func)
            
        # Get the gradients
        weight_gradient = tape.gradient(loss, self.model.trainable_variables)
        temp_p = []
        for i in range(len(weight_gradient)):
            wg = tf.math.multiply(weight_gradient[i], lrate)
            temp_p.append(tf.math.add(self.p[i], wg)) # maybe come back and make this subtraction
        self.p = np.asarray(temp_p)

        self.train_loss(loss)
        try:
            self.train_metric(labels, predictions)
        except:
            self.train_metric(labels, np.argmax(predictions, axis=1))
            
        return self.posterior_mean, self.posterior_var

    def evaluate_U(self, features, labels):
        predictions = self.model(features)
        v_loss = losses.normal_potential_energy(labels, predictions, self.prior_mean,
                                               self.prior_var, self.q, self.loss_func)
        worst_case = predictions
        self.U_metric(v_loss)
        res = self.U_metric.result()
        self.U_metric.reset_states()
        return res
    
    def clear_metrics(self):
        self.train_loss.reset_states(), self.train_metric.reset_states()
        self.valid_loss.reset_states(), self.valid_metric.reset_states()
        self.extra_metric.reset_states()
        
    def train(self, X_train, y_train, X_test=None, y_test=None, X_l=None, X_u=None, constraint=False): 
        # We generate this only to speed up parallel computations of the test statistics
        # if there are any to compute
        self.X_train = X_train
        self.y_train = y_train
        print(np.shape(X_train), np.shape(y_train))
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(100).batch(self.batch_size)
        if(self.robust_linear):
            self.max_eps = self.epsilon
            self.epsilon = 0.0
            
        # Evaluate the potential energy of the 
        self.current_U = self.evaluate_U(X_train, y_train)
        
        # Perform the burn-in of the chain
        self.burning_in_chain = True
        temp_m = self.m; self.m = self.m_burn
        for iter in range(self.burn_in):
            print("Still in Burn-In state of Markov chain.")
            if(constraint):
                self.sample(X_train, y_train, self.learning_rate, features_l=X_l, features_u=X_u, constraint=True)
            else:
                self.sample(train_ds, self.learning_rate)
            for test_features, test_labels in test_ds:
                test_features = np.asarray([test_features])
                test_labels = np.asarray([test_labels])
                self.model_validate(test_features, test_labels)
                
            # Grab the results, print metrics
            (loss, acc) = self.train_loss.result(), self.train_metric.result()
            (val_loss, val_acc) = self.valid_loss.result(), self.valid_metric.result()
            self.logging(loss, acc, val_loss, val_acc, iter)
      
            # Clear the current state of the metrics
            self.clear_metrics()
            
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
        self._learning_rate = self.learning_rate
        for it in range(self.epochs):
            self.learning_rate = self._learning_rate * (1 / (1 + self.decay * it))
            if(constraint):
                self.sample(X_train, y_train, self.learning_rate, features_l=X_l, features_u=X_u, constraint=True)
            else:
                self.sample(X_train, y_train, self.learning_rate)
            
            for test_features, test_labels in test_ds:
                test_features = np.asarray([test_features])
                test_labels = np.asarray([test_labels])
                self.model_validate(test_features, test_labels)
                
            # Grab the results
            (loss, acc) = self.train_loss.result(), self.train_metric.result()
            (val_loss, val_acc) = self.valid_loss.result(), self.valid_metric.result()
            self.logging(loss, acc, val_loss, val_acc, iter)
            print("Out of Burn-In state. Generating samples from the chain")

            # Clear the current state of the metrics
            self.clear_metrics()        
            
    def train_chain(self, X_train, y_train, X_test=None, y_test=None):
        self.train(X_train, y_train, X_test, y_test)
        self.global_rets.extend(self.num_rets[1:])
        self.global_iterate += self.iterate
        self.global_poseterior_samples.extend(self.posterior_samples)
        
    def constraint_train(self, X_train, y_train, X_l, X_u, X_test=None, y_test=None):
        self.train(X_train, y_train, X_test, y_test, X_l=X_l, X_u=X_u, constraint=True)
        self.global_rets.extend(self.num_rets[1:])
        self.global_iterate += self.iterate
        self.global_poseterior_samples.extend(self.posterior_samples)
        print("Must call finlize_chain() after this to save/use posterior samples")

    def finalize_chain(self):
        self.num_rets = self.global_rets # number of times each iterate in the chain has appeared
        self.iterate = self.global_iterate
        self.posterior_samples = self.global_poseterior_samples
        print("Finalized chain")
        print(len(self.num_rets), len(self.posterior_samples))
            
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

            

