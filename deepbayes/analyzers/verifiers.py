# Author: Matthew Wicker


from statsmodels.stats.proportion import proportion_confint
import math
import numpy as np
import tensorflow as tf
"""
NOW TWO STAGES IN THIS FUNCTION.
Interval propogation from the input space (only).
NB - This is different from UAI as we do not consider weight space here
+ this implimentation is much faster than the previous version
NB - There is an odd diffentiation thing where if eps == 0.0 I think I
     am getting zero gradients. Need to look into it more, but this is
     edge behavior which hopefully does not impact the rest of the results
@Variable model - Bayesian posterior (Using the compiled BayesKeras class)
@Variable inp - the tensor of inputs (Usually a full batch of inputs)
@Variable weights - the sampled weights (Can iterate this function to get expectation)
@Variable eps - the $l_\infty$ epsilon radius considered
"""
def propagate_conv2d(W, b, x_l, x_u):
    w_pos = tf.maximum(W, 0)
    w_neg = tf.minimum(W, 0)
    h_l = (tf.nn.convolution(x_l, w_pos) +
          tf.nn.convolution(x_u, w_neg))
    h_u = (tf.nn.convolution(x_u, w_pos) +
          tf.nn.convolution(x_l, w_neg))
    nom = tf.nn.convolution((x_l+x_u)/2, W) + b
    h_l = nom + h_l
    h_u = nom + h_u

#    mu = tf.divide(tf.math.add(x_u, x_l), 2)
#    r = tf.divide(tf.math.subtract(x_u, x_l), 2)
#    mu_new = tf.math.add(tf.nn.convolution(mu, W), b) 
#    rad = tf.nn.convolution(r, tf.math.abs(W)) 
#    h_u = tf.math.add(mu_new, rad)
#    h_l = tf.math.subtract(mu_new, rad)
    return h_l, h_u

def propagate_interval(W, b, x_l, x_u):
    mu = tf.divide(tf.math.add(x_u, x_l), 2)
    r = tf.divide(tf.math.subtract(x_u, x_l), 2)
    mu_new = tf.math.add(tf.matmul(mu, W), b) 
    rad = tf.matmul(r, tf.math.abs(W)) 
    h_u = tf.math.add(mu_new, rad)
    h_l = tf.math.subtract(mu_new, rad)
    return h_l, h_u

def IBP(model, inp, weights, eps, predict=False):
    h_u = tf.clip_by_value(tf.math.add(inp, eps), 0.0, 1.0)
    h_l = tf.clip_by_value(tf.math.subtract(inp, eps), 0.0, 1.0)
    layers = model.model.layers
    offset = 0
    for i in range(len(layers)):
        #print(len(layers[i].get_weights()))
        #print(i, offset, weights[2*(i-offset)].shape, len(layers[i].get_weights()))
        if(len(layers[i].get_weights()) == 0):
            h_u = model.model.layers[i](h_u)
            h_l = model.model.layers[i](h_l)
            offset += 1
            continue
        w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
        #print(w.shape)
        if(len(w.shape) == 2):
            h_l, h_u = propagate_interval(w, b, h_l, h_u)
            activate = True
        elif(len(w.shape) == 4):
            h_l, h_u = propagate_conv2d(w, b, h_l, h_u)
            activate = True
        #else:
        #    h_u = model.layers[i](h_u)
        #    h_l = model.layers[i](h_l)
        #    activate = False
        if(i < len(layers)-1 and activate):
            h_l = model.model.layers[i].activation(h_l)
            h_u = model.model.layers[i].activation(h_u)
    return h_l, h_u




# also known as the chernoff bound
def okamoto_bound(epsilon, delta):
    return (-1*.5) * math.log(float(delta)/2) * (1.0/(epsilon**2))

# This is h_a in the paper
def absolute_massart_halting(succ, trials, I, epsilon, delta, alpha):
    gamma = float(succ)/trials
    if(I[0] < 0.5 and I[1] > 0.5):
        return -1
    elif(I[1] < 0.5):
        val = I[1]
        h = (9/2.0)*(((3*val + epsilon)*(3*(1-val)-epsilon))**(-1))
        return math.ceil((h*(epsilon**2))**(-1) * math.log((delta - alpha)**(-1)))
    elif(I[0] >= 0.5):
        val = I[0]
        h = (9/2.0)*(((3*(1-val) + epsilon)*((3*val)+epsilon))**(-1))
        return math.ceil((h*(epsilon**2))**(-1) * math.log((delta - alpha)**(-1)))

"""

"""
def chernoff_bound_verification(model, inp, eps, cls, **kwargs):
    from tqdm import trange
    delta = kwargs.get('delta', 0.3)
    alpha = kwargs.get('alpha', 0.05)
    confidence = kwargs.get('confidence', 0.95)
    verbose = kwargs.get('verbose', False)
    epsilon = 1-confidence
    chernoff_bound = math.ceil( (1/(2*epsilon**2)) * math.log(2/delta) )
    softmax = 0
    for i in trange(chernoff_bound, desc="Sampling for Chernoff Bound Satisfaction"):
        model.set_weights(model.sample())
        logit_l, logit_u = IBP(model, inp, model.model.get_weights(), eps, predict=False)
        v1 = tf.one_hot(cls, depth=10)
        v2 = 1 - tf.one_hot(cls, depth=10)
        v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
        worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
        if(type(softmax) == int):
            softmax = model.model.layers[-1].activation(worst_case)
        else:
            softmax += model.model.layers[-1].activation(worst_case)
    return softmax
    #print("Not yet implimented")

"""
property - a function that takes a vector and returns a boolean if it was successful
"""
def massart_bound_check(model, inp, eps, cls, **kwargs):
    delta = kwargs.get('delta', 0.3)
    alpha = kwargs.get('alpha', 0.05)
    confidence = kwargs.get('confidence', 0.95)
    verbose = kwargs.get('verbose', False)
    
    atk_locs = []
    epsilon = 1-confidence
    chernoff_bound = math.ceil( (1/(2*epsilon**2)) * math.log(2/delta) )
    print("BayesKeras. Maximum sample bound = %s"%(chernoff_bound))
    successes, iterations, misses = 0.0, 0.0, 0.0
    halting_bound = chernoff_bound
    I = [0,1]
    while(iterations <= halting_bound):
        if(iterations > 0 and verbose):
            print("Working on iteration: %s \t Bound: %s \t Param: %s"%(iterations, halting_bound, successes/iterations))  
        model.set_weights(model.sample())
        logit_l, logit_u = IBP(model, inp, model.model.get_weights(), eps, predict=False)
        v1 = tf.one_hot(cls, depth=10)
        v2 = 1 - tf.one_hot(cls, depth=10)
        worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
        if(np.argmax(np.squeeze(worst_case)) != cls):
            misses += 1
            result = 0
        else:
            result = 1
        successes += result
        iterations += 1
        # Final bounds computation below
        lb, ub = proportion_confint(successes, iterations, method='beta')
        if(math.isnan(lb)):
            lb = 0.0 # Setting lb to zero if it is Nans
        if(math.isnan(ub)):
            ub = 1.0 # Setting ub to one if it is Nans
        I = [lb, ub]
        hb = absolute_massart_halting(successes, iterations, I, epsilon, delta, alpha)
        if(hb == -1):
            halting_bound = chernoff_bound
        else:
            halting_bound = min(hb, chernoff_bound)
    if(verbose):
        print("Exited becuase %s >= %s"%(iterations, halting_bound))
    return successes/iterations
    #return None
