import numpy as np
import math
import tensorflow_probability as tfp
import tensorflow as tf




"""
Computes log probability of x coming from Guassian 
parameterized by mu and rho
"""
def log_q(x, mu, rho):
    logq_sum = 0.0
    for i in range(len(x)):
        sigma = tf.math.softplus(rho[i])
        posti_dist = tfp.distributions.Normal(mu[i], sigma) #!*! is this variance or std?
        logq_sum += tf.math.reduce_mean(posti_dist.log_prob(x[i]))
    return tf.cast(logq_sum, 'float32')

"""
KL loss function for training by BBB. We do a bit of
algebra to arrive at the formulation that we return, 
but it is straight-forwars to get to there from the 
normal BBB loss. [Added algebra below for transparency ;)]
@Variable labels - ground truth labels (should be single class numbers not one hot)
@Variable predictions - predictions from the BNN (full probability output)
@Variable weight - weight used for this sample
@Variable prior_mean - mean of the prior distribution
@Variable prior_var - variance of the prior distribution (here encoded as rho)
@Variable q_mean - mean of the current variational posterior
@Variable q_var - variance of the current variational posterior (here encoded as rho)
@Variable loss_func - data liklihood function (robust version of categorical crossentropy
@Variable kl_weight - weight on the contribution to the loss of the kl divergence 
"""
def robust_KL_Loss(labels, predictions, weight, prior_mean, prior_var, q_mean, q_var, loss_func, kl_weight, worst_case, robust_lambda):
    posti_likli = log_q(weight, q_mean, q_var)
    prior_likli = log_q(weight, prior_mean, prior_var)
    data_likli = loss_func(labels, predictions, worst_case, robust_lambda)
    data_likli = tf.cast(data_likli, 'float32')
    return data_likli + (kl_weight*(posti_likli-prior_likli)), posti_likli-prior_likli

def KL_Loss(labels, predictions, weight, prior_mean, prior_var, q_mean, q_var, loss_func, kl_weight):
    posti_likli = log_q(weight, q_mean, q_var)
    prior_likli = log_q(weight, prior_mean, prior_var)
    data_likli = loss_func(labels, predictions)
    data_likli = tf.cast(data_likli, 'float32')
    return data_likli + (kl_weight*(posti_likli-prior_likli)), posti_likli-prior_likli


def crossentropy_loss(target, output):
    _EPSILON = 0.00001
    one_hot_cls = tf.one_hot(target, depth=10)
    new_out = tf.math.reduce_max(output*one_hot_cls)
    #new_out = tf.math.reduce_max((robust_lambda*(output*one_hot_cls)) + ((1-robust_lambda)*(output_worst*one_hot_cls)), axis=1)
    output = tf.convert_to_tensor(new_out)
    epsilon = tf.convert_to_tensor(_EPSILON, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    return - tf.reduce_mean(tf.math.log(output))

"""
def crossentropy_loss(target, output):
    output = tf.reshape(tf.math.reduce_max(output, axis=1), (-1, 1))
    #output = tf.cast(output, tf.float32)
    #target = tf.cast(target, tf.float32)
    return tf.keras.losses.sparse_categorical_crossentropy(target, output)
"""

def robust_crossentropy_loss(target, output, output_worst, robust_lambda):
    _EPSILON = 0.000001
    # There HAS to be a better way to do this that I need to look for, smh
#    new_out = []
#    for i in range(len(target)):
#        index = int(target[i])
#        new_out.append((robust_lambda*output[i][index]) + ((1-robust_lambda)*output_worst[i][index]))
    one_hot_cls = tf.one_hot(target, depth=10)
    new_out = tf.math.reduce_max((robust_lambda*(output*one_hot_cls)) + ((1-robust_lambda)*(output_worst*one_hot_cls)), axis=1)
    output = tf.convert_to_tensor(new_out)
    epsilon = tf.convert_to_tensor(_EPSILON, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    return - tf.reduce_mean(tf.math.log(output))
"""
def robust_crossentropy_loss(target, output, output_worst, robust_lambda):
    _EPSILON = 0.000001
    # There HAS to be a better way to do this that I need to look for, smh
    new_out = []
    for i in range(len(target)):
        index = int(target[i])
        c1 = tf.math.multiply(robust_lambda, output[i][index])
        c2 = tf.math.multiply(1-robust_lambda, output_worst[i][index])
        new_out.append(tf.math.add(c1,c2))
    # Attempt at something better
    #one_hot_cls = tf.one_hot(target, depth=10)
    #new_out = tf.math.reduce_max((robust_lambda*(output*one_hot_cls)) + ((1-robust_lambda)*(output_worst*one_hot_cls)), axis=1)
    output = tf.convert_to_tensor(new_out)
    epsilon = tf.convert_to_tensor(_EPSILON, tf.dtypes.float32)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    return - tf.reduce_mean(tf.math.log(output))
"""

def adv_crossentropy_loss(target, output, output_worst, robust_lambda):
    #print(target)
    _EPSILON = 0.001
    epsilon = tf.convert_to_tensor(_EPSILON, output.dtype.base_dtype)
    output_worst = tf.clip_by_value(output_worst, epsilon, 1. - epsilon)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    #print(output_worst, output)
    target = tf.one_hot(target, depth=10)
    comp1 = tf.keras.losses.categorical_crossentropy(output, target)
    comp2 = tf.keras.losses.categorical_crossentropy(output_worst, target)
    return (((robust_lambda)*comp1) + ((1-robust_lambda)*comp2))

"""
Computes log probability of x coming from Guassian
parameterized by mu and rho
"""
def log_pdf(x, mu, sigma):
    logq_sum = 0.0
    for i in range(len(x)):
        posti_dist = tfp.distributions.Normal(mu[i], sigma[i]) #!*! is this variance or std?
        logq_sum += tf.math.reduce_mean(posti_dist.log_prob(x[i])) # This should be the log but it gets so small!!!
    return tf.cast(logq_sum, 'float32')

def normal_potential_energy(target, output, prior_mean, prior_var, q, loss_func):
    pw_d = loss_func(target, output)
    pw = log_pdf(q, prior_mean, prior_var)
    #print("DATA: ", pw_d)
    #print("PRIOR: ", pw)
    return pw_d - pw

def robust_potential_energy(target, output, prior_mean, prior_var, q, loss_func, output_worst, robust_lambda):
    pw_d = loss_func(target, output, output_worst, robust_lambda)
    pw = log_pdf(q, prior_mean, prior_var)
    #print("DATA: ", pw_d)
    #print("PRIOR: ", pw)
    return pw_d - pw
