# By: Matthew Wicker
# Abstract base class for a Bayesian Optimizer which takes a keras class.
# Care has been taken to support robust training schemes.

# NB to developers who may use this: @abstractmethod means the method 
# must be overridden in all subclasses. This is just from my habits as 
# a C++/Java dev, but it seems to be good practice.

from abc import ABC, abstractmethod

import os
import copy
import math
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tqdm import trange

import tensorflow_probability as tfp
from deepbayes import analyzers

# A dumb mistake on my part which needs to be factored out
def softplus(x):
     return tf.math.softplus(x)

class Optimizer(ABC):
    def __init__(self):
        print("This optimizer does not have a default compilation method. Please make sure to call the correct .compile method before use.")

    @abstractmethod
    def compile(self, keras_model, loss_fn, batch_size, learning_rate, decay,
                      epochs, prior_mean, prior_var, **kwargs):
        self.model = keras_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay = decay
        self.epochs = epochs
        self.loss_func = loss_fn

        self.log_dir = kwargs.get('log_file', '/tmp/BayesKeras.log')

        self.det = kwargs.get('deterministic', False)
        self.inflate_prior = kwargs.get('inflate_prior', 1)
        self.input_noise = kwargs.get('input_noise', 0.0)
        #self.prior_mean, self.prior_var = self.prior_generator(keras_model, prior_mean, prior_var, verbose=True)
        self.prior_mean, self.prior_var = self.prior_generator(prior_mean, prior_var)
        self.posterior_mean, self.posterior_var = self.prior_generator(prior_mean, prior_var)
        #self.posterior_mean = copy.deepcopy(self.prior_mean)
        #self.posterior_var = copy.deepcopy(self.prior_var)

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.valid_loss = tf.keras.metrics.Mean(name="valid_loss")

        # Right now I only have one accessory metric. Will come back and add 
        # many later. 
        self.train_metric = kwargs.get('metric', tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc"))
        self.valid_metric = kwargs.get('metric', tf.keras.metrics.SparseCategoricalAccuracy(name="valid_acc"))
        self.extra_metric = kwargs.get('metric', tf.keras.metrics.SparseCategoricalAccuracy(name="extra_acc"))

        self.robust_train = kwargs.get('robust_train', 0)
        if(self.robust_train != 0):
            print("BayesKeras: Detected robust training at compilation. Please ensure you have selected a robust-compatible loss")
            self.epochs += 1
        self.epsilon = kwargs.get('epsilon', 0.1) 
        self.robust_lambda = kwargs.get('rob_lam', 0.5)
        self.robust_linear = kwargs.get('linear_schedule', True)        

        self.attack_loss = tf.keras.losses.SparseCategoricalCrossentropy()

        self.loss_monte_carlo = kwargs.get('loss_mc', 2)
        self.eps_dist = tfp.distributions.Exponential(rate = 1.0/float(self.epsilon))

        self.acc_log = []
        self.rob_log = []
        self.loss_log = []
        return self

    @abstractmethod
    def train(self, X_train, y_train, X_test=None, y_test=None, **kwargs):
        self.N = len(X_train)
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(100).batch(self.batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(self.batch_size)

        if(self.robust_linear):
            self.max_eps = self.epsilon
            self.epsilon = 0.0
            self.max_robust_lambda = self.robust_lambda

        lr = self.learning_rate; decay = self.decay
        for epoch in range(self.epochs):
            lrate = self.learning_rate * (1 / (1 + self.decay * epoch))

            # Run the model through train and test sets respectively
            for (features, labels) in tqdm(train_ds):
                features += np.random.normal(loc=0.0, scale=self.input_noise, size=features.shape)
                self.posterior, self.posterior_var = self.step(features, labels, lrate)
            for test_features, test_labels in test_ds:
                self.model_validate(test_features, test_labels)
                    
            # Grab the results
            (loss, acc) = self.train_loss.result(), self.train_metric.result()
            (val_loss, val_acc) = self.valid_loss.result(), self.valid_metric.result()
            self.logging(loss, acc, val_loss, val_acc, epoch)
            
            # Clear the current state of the metrics
            self.train_loss.reset_states(), self.train_metric.reset_states()
            self.valid_loss.reset_states(), self.valid_metric.reset_states()
            self.extra_metric.reset_states()
            
            if(self.robust_linear):
                self.epsilon += self.max_eps/self.epochs
#                self.robust_lambda -= self.max_robust_lambda/(self.epochs*2)
    @abstractmethod
    def step(self, features, labels, learning_rate):
        pass

    def model_validate(self, features, labels):
        #self.model.set_weights(self.sample())
        predictions = self.model(features)
        if(self.robust_train == 1): # We only check with IBP if we need to 
            logit_l, logit_u = analyzers.IBP(self, features, self.model.get_weights(), self.epsilon)
            #logit_l, logit_u = analyzers.IBP(self, features, self.model.trainable_variables, 0.0)
            v1 = tf.one_hot(labels, depth=10)
            v2 = 1 - tf.one_hot(labels, depth=10)
            worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
            worst_case = self.model.layers[-1].activation(worst_case)
            v_loss = self.loss_func(labels, predictions, worst_case, self.robust_lambda)
            self.extra_metric(labels, worst_case)
        elif(self.robust_train == 2):
            v_loss = self.loss_func(labels, predictions, predictions, self.robust_lambda)
            worst_case = predictions
        elif(self.robust_train == 3 or self.robust_train == 5): # We only check with IBP if we need to 
            logit_l, logit_u = analyzers.IBP(self, features, self.model.get_weights(), self.epsilon)
            #logit_l, logit_u = analyzers.IBP(self, features, self.model.trainable_variables, 0.0)
            #print(logit_l.shape)
            v1 = tf.squeeze(tf.one_hot(labels, depth=10))
            v2 = tf.squeeze(1 - tf.one_hot(labels, depth=10))
            #print(v1.shape, v2.shape)
            worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
            #print(worst_case.shape)
            worst_case = self.model.layers[-1].activation(worst_case)
            #print(worst_case.shape)
            v_loss = self.loss_func(labels, worst_case)
            self.extra_metric(labels, worst_case)
        else:
            v_loss = self.loss_func(labels, predictions)
            worst_case = predictions
        self.valid_metric(labels, predictions)
        self.valid_loss(v_loss)
        #self.valid_rob(labels, worst_case)

    def logging(self, loss, acc, val_loss, val_acc, epoch):
        # Local logging
        if(self.robust_train == 0):
            template = "Epoch {}, loss: {:.3f}, acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}" 
            print (template.format(epoch+1, loss,
                         acc,
                         val_loss,
                         val_acc))
        else:
            rob = self.extra_metric.result()
            template = "Epoch {}, loss: {:.3f}, acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}, rob: {:.3f}, (eps = {:.6f})" 
            print (template.format(epoch+1, loss,
                         acc,
                         val_loss,
                         val_acc, rob, self.epsilon))
        log_template = "Epoch: {}, Train: [Loss: {:.3f}, Acc: {:.3f}], Test: [Loss: {:.3f}, Acc: {:.3f}]"
        logging.basicConfig(filename=self.log_dir, level=logging.DEBUG)
        logging.info(log_template.format(epoch+1, loss, acc, val_loss, val_acc))


    def sample(self):
        sampled_weights = []
        for i in range(len(self.posterior_mean)):
            sampled_weights.append(np.random.normal(loc=self.posterior_mean[i], 
                                                    scale=self.posterior_var[i]))
        return sampled_weights

    def _gen_implicit_prior(self):
        print("BayesKeras: Using implicit prior")
        prior_mean = []
        prior_var = []
        for i in range(len(self.model.layers)):
            try:
                sha = self.model.layers[i].get_weights()[0].shape
                b_sha = self.model.layers[i].get_weights()[1].shape
                if(len(sha) > 2):
                    nin = 1
                    for i in range(len(sha)-1):
                        nin*=sha[i]
                else:
                    nin = sha[0]
                std = math.sqrt(self.inflate_prior/(nin))
                print(sha, std)
                mean_w = tf.zeros(sha)
                var_w = tf.ones(sha) * std
                mean_b = tf.zeros(b_sha)
                var_b = tf.ones(b_sha) * std
                prior_mean.append(mean_w); prior_mean.append(mean_b)
                prior_var.append(var_w); prior_var.append(var_b)
            except:
                pass
        return prior_mean, prior_var

    def prior_generator(self, means, vars):
        if(type(means) == int and type(vars) == int):
            if(means < 0 or vars < 0):
                model_mean, model_var = self._gen_implicit_prior()
                #for i in range(len(model_var)):
                #    model_var[i] = tf.math.log(tf.math.exp(self.model_var[i])-1)
                return model_mean, model_var
        if(type(means) == int or type(means) == float):
            if(means == -1):
                means = 0.0
            mean_params = [means for i in range(len(self.model.weights))]
            means = mean_params
        if(type(vars) == int or type(vars) == float):
            if(vars == -1):
                vars = 0.0
            var_params = [vars for i in range(len(self.model.weights))]
            vars = var_params
        model_mean = []
        model_var  = []
        index = 0.0
        for weight in self.model.weights:
            param_index = math.floor(index/2.0)
            mean_i = tf.math.multiply(tf.ones(weight.shape), means[param_index]) 
            vari_i = tf.math.multiply(tf.ones(weight.shape), vars[param_index]) 
            model_mean.append(mean_i)
            model_var.append(vari_i)
            index += 1
        return model_mean, model_var

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path+"/mean", np.asarray(self.posterior_mean))
        np.save(path+"/var", np.asarray(self.posterior_var))
        self.model.save(path+'/model.h5')
        model_json = self.model.to_json()
        with open(path+"/arch.json", "w") as json_file:
            json_file.write(model_json)


    def load(self, path):
        posti_mean = np.load(path + "/mean.npy", allow_pickle=True)
        posti_var = np.load(path + "/var.npy", allow_pickle=True)
        v, m = [], []
        for i in range(len(posti_var)):
            posti_var[i] = tf.convert_to_tensor(posti_var[i], dtype=tf.float32)
            posti_mean[i] = tf.convert_to_tensor(posti_mean[i], dtype=tf.float32)
            posti_var[i] =  tf.math.log(tf.math.exp(posti_var[i])-1)
            v.append(posti_var[i])
            m.append(posti_mean[i])
        self.posti_var = v
        self.posti_mean = m


    def predict(self, input, n=1):
        # The below is only a single sample predict without any sampling because
        # basically during training we want to evaluate the gradient at the point
        # estimate so we dont mess up our scaling during reparamterized gradient
        # computations. Override this if you want different behaviour
        return self.model(input)
        """
        out = -1
        for i in range(n):
            self.model.set_weights(self.sample())
            if(type(out) == int):
                out = self.model(input)
            else:
                out += self.model(input)
        return out/float(n)
        """
