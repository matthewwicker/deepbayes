# Author: Matthew Wicker
# Probabilistic Verification script for Bayesian Neural Networks

import copy
import math
import tqdm
import itertools
import numpy as np
from . import attacks
from tqdm import trange
import tensorflow as tf
from multiprocessing import Pool
from statsmodels.stats.proportion import proportion_confint


def propagate_conv2d(W, b, x_l, x_u, marg=0, b_marg=0):
    marg = tf.divide(marg, 2)
    b_marg = tf.divide(b_marg, 2)
    w_pos = tf.maximum(W+marg, 0)
    w_neg = tf.minimum(W-marg, 0)
    h_l = (tf.nn.convolution(x_l, w_pos) +
          tf.nn.convolution(x_u, w_neg))
    h_u = (tf.nn.convolution(x_u, w_pos) +
          tf.nn.convolution(x_l, w_neg))
    nom = tf.nn.convolution((x_l+x_u)/2, W)
    h_l = nom + h_l + (b - b_marg)
    h_u = nom + h_u + (b + b_marg)
    return h_l, h_u


def propagate_interval(W, b, x_l, x_u, marg=0, b_marg=0):
    marg = tf.divide(marg, 2)
    b_marg = tf.divide(b_marg, 2)
    x_mu = tf.cast(tf.divide(tf.math.add(x_u, x_l), 2), dtype=tf.float64)
    x_r =  tf.cast(tf.divide(tf.math.subtract(x_u, x_l), 2), dtype=tf.float64)
    if(type(marg) == int):
        W_r = 0.0 * W_mu
    W_mu = tf.cast(W, dtype=tf.float64)
    W_r =  tf.cast(marg, dtype=tf.float64)
    b = tf.cast(b, dtype=tf.float64)
    b_marg = tf.cast(b_marg, dtype=tf.float64)
    b_u =  tf.cast(b + b_marg, dtype=tf.float64)
    b_l =  tf.cast(b - b_marg, dtype=tf.float64)
    #h_mu = tf.math.add(tf.matmul(x_mu, W_mu), b_mu)
    h_mu = tf.matmul(x_mu, W_mu)
    x_rad = tf.matmul(x_r, tf.math.abs(W_mu))
    try:
        W_rad = tf.matmul(tf.abs(x_mu), W_r)
        Quad = tf.matmul(tf.abs(x_r), tf.abs(W_r))
    except:
        W_rad = 0.0
        Quad = 0.0   
    h_u = tf.add(tf.add(tf.add(tf.add(h_mu, x_rad), W_rad), Quad), b_u)
    h_l = tf.add(tf.subtract(tf.subtract(tf.subtract(h_mu, x_rad), W_rad), Quad), b_l)
    return h_l, h_u

def propagate_interval_exact(W, b, x_l, x_u, marg=0, b_marg=0):
    """
    Function which does matrix multiplication but with weight and
    input intervals.
    """
    x_l = tf.cast(x_l, dtype=tf.float32);x_u = tf.cast(x_u, dtype=tf.float32)
    W = tf.cast(W, dtype=tf.float32); b = tf.cast(b, dtype=tf.float32)
    marg = tf.cast(marg, dtype=tf.float32); b_marg = tf.cast(b_marg, dtype=tf.float32)
    x_l = tf.squeeze(x_l); x_u = tf.squeeze(x_u)
    W_l, W_u = W-marg, W+marg           #Use eps as small symetric difference about the mean
    b_l, b_u = b-b_marg, b+b_marg       #Use eps as small symetric difference about the mean
    h_max = np.zeros(len(W[0]))         #Placeholder variable for return value
    h_min = np.zeros(len(W[0]))         #Placeholder variable for return value
    for i in range(len(W)):             #This is literally just a step-by-step matrix multiplication
        for j in range(len(W[0])):      # where we are taking the min and max of the possibilities
            out_arr = [W_l[i][j]*x_l[i], W_l[i][j]*x_u[i],
                       W_u[i][j]*x_l[i], W_u[i][j]*x_u[i]]
            h_min[j] += min(out_arr)
            h_max[j] += max(out_arr)
    h_min = h_min + b_l
    h_max = h_max + b_u
    return h_min, h_max         #Return the min and max of the intervals.
                                #(dont forget to apply activation function after)

"""
Version of IBP which takes an explicit upper and lower bound.
"""
def IBP_prob(model, s0, s1, weights, weight_margin=0, logits=True):
    h_l = s0
    h_u = s1
    layers = model.model.layers
    offset = 0
    for i in range(len(layers)):
        if(len(layers[i].get_weights()) == 0):
            #print("FLATTENED")
            h_u = model.model.layers[i](h_u)
            h_l = model.model.layers[i](h_l)
            offset += 1
            continue
        w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
        sigma = model.posterior_var[2*(i-offset)]
        b_sigma = model.posterior_var[2*(i-offset)+1]
        marg = weight_margin*sigma
        b_marg = weight_margin*b_sigma
        if(len(w.shape) == 2):
            h_l, h_u = propagate_interval(w, b, h_l, h_u, marg=marg, b_marg=b_marg)
            activate = True
        elif(len(w.shape) == 4):
            h_l, h_u = propagate_conv2d(w, b, h_l, h_u, marg=marg, b_marg=b_marg)
            activate = True
        if(i < len(layers)-1):
            h_l = model.model.layers[i].activation(h_l)
            h_u = model.model.layers[i].activation(h_u)
    return h_l, h_u


def IBP_samp(model, s0, s1, weights, weight_margin=0, logits=True):
    h_l = s0
    h_u = s1
    layers = model.model.layers
    offset = 0
    for i in range(len(layers)):
        if(len(layers[i].get_weights()) == 0):
            #print("FLATTENED")
            h_u = model.model.layers[i](h_u)
            h_l = model.model.layers[i](h_l)
            offset += 1
            continue
        w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
        sigma = weights[2*(i-offset)] #model.posterior_var[2*(i-offset)]
        b_sigma = weights[2*(i-offset)+1] #model.posterior_var[2*(i-offset)+1]
        marg = weight_margin*sigma
        b_marg = weight_margin*b_sigma
        if(len(w.shape) == 2):
            h_l, h_u = propagate_interval(w, b, h_l, h_u, marg=marg, b_marg=b_marg)
            activate = True
        elif(len(w.shape) == 4):
            h_l, h_u = propagate_conv2d(w, b, h_l, h_u, marg=marg, b_marg=b_marg)
            activate = True
        if(i < len(layers)-1):
            h_l = model.model.layers[i].activation(h_l)
            h_u = model.model.layers[i].activation(h_u)
    if(not logits):
        h_l = model.model.layers[-1].activation(h_l)
        h_u = model.model.layers[-1].activation(h_u)
        return h_l, h_u
    return h_l, h_u

def pIBP(model, inp_l, inp_u, weights, predict=False):
    #if(predict == False):
    #    h_u = tf.clip_by_value(tf.math.add(inp, eps), 0.0, 1.0)
    #    h_l = tf.clip_by_value(tf.math.subtract(inp, eps), 0.0, 1.0)
    #else:
    h_u = inp_u
    h_l = inp_l
    layers = model.model.layers
    offset = 0
    for i in range(len(layers)):
        if(len(layers[i].get_weights()) == 0):
            h_u = model.model.layers[i](h_u)
            h_l = model.model.layers[i](h_l)
            offset += 1
            continue
        w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
        if(len(w.shape) == 2):
            h_l, h_u = propagate_interval_exact(w, b, h_l, h_u)
            activate = True
        #elif(len(w.shape) == 4):
        #    h_l, h_u = propagate_conv2d(w, b, h_l, h_u)
        #    activate = True
        if(predict == False and i >= len(layers)-1):
            continue
        else:
            h_l = model.model.layers[i].activation(h_l)
            h_u = model.model.layers[i].activation(h_u)
    return h_l, h_u



"""
Given a set intervals, compute the probability of a random
sample from a guassian falling in these intervals. (Taken from lemma)
of the document
"""
import math
from scipy.special import erf
def compute_erf_prob(intervals, mean, var):
    prob = 0.0
    for interval in intervals:
        val1 = erf((mean-interval[0])/(math.sqrt(2)*(var)))
        val2 = erf((mean-interval[1])/(math.sqrt(2)*(var)))
        prob += 0.5*(val1-val2)
    return prob


def intersect_intervals(wi_a, wi_b, margin, var):
    intersection_l = []
    intersection_u = []
    for l in range(len(wi_a)):
        wi_a_u = (wi_a[l] + (var[l]*margin)).numpy() # Upper bound for these variables
        wi_a_l = (wi_a[l] - (var[l]*margin)).numpy() # Lower bound for these variables
        wi_b_u = (wi_b[l] + (var[l]*margin)).numpy() # Upper bound for these variables
        wi_b_l = (wi_b[l] - (var[l]*margin)).numpy() # Lower bound for these variables
        intersect_l = np.maximum(wi_a_l, wi_b_l)
        intersect_u = np.minimum(wi_a_u, wi_b_u)
        intersect_l[(intersect_u - intersect_l) <= 0] = 0
        intersect_u[(intersect_u - intersect_l) <= 0] = 0
        intersection_l.append(np.array(intersect_l))
        intersection_u.append(np.array(intersect_u))
    return intersection_l, intersection_u

def intersection_bounds(wa_l, wa_u, wi_b, margin, var):
    intersection_l = []
    intersection_u = []
    for l in range(len(wa_l)):
        wi_a_u = wa_u[l] # Upper bound for these variables
        wi_a_l = wa_l[l] # Lower bound for these variables
        wi_b_u = (wi_b[l] + (var[l]*margin)).numpy() # Upper bound for these variables
        wi_b_l = (wi_b[l] - (var[l]*margin)).numpy() # Lower bound for these variables
        intersect_l = np.maximum(wi_a_l, wi_b_l)
        intersect_u = np.minimum(wi_a_u, wi_b_u)
        intersect_l[(intersect_u - intersect_l) <= 0] = 0
        intersect_u[(intersect_u - intersect_l) <= 0] = 0
        intersection_l.append(np.array(intersect_l))
        intersection_u.append(np.array(intersect_u))
    return intersection_l, intersection_u


def get_bounds(wi_a, margin, var):
    wi_l = []
    wi_u = []
    for l in range(len(wi_a)):
        wi_a_l = (wi_a[l] - (var[l]*margin)).numpy()
        wi_a_u = (wi_a[l] + (var[l]*margin)).numpy()
        wi_l.append(wi_a_l)
        wi_u.append(wi_a_u)
    return wi_l, wi_u

def compute_interval_probs_weight_std(arg):
    vector_intervals, marg, mean, var = arg
    means = mean; # vars = var
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in range(len(vector_intervals[0])):
        for j in range(len(vector_intervals[0][0])):
            intervals = []
            p = 0.0
            for num_found in range(len(vector_intervals)):
                interval = [vector_intervals[num_found][i][j]-(var[i][j]*marg), vector_intervals[num_found][i][j]+(var[i][j]*marg)]
                interval = [interval]
                p += compute_erf_prob(interval, means[i][j], var[i][j])
            prob_vec[i][j] = p #math.log(p)
    return np.asarray(prob_vec) # what is being returned here is the sum of the cumulative density for each entry in the weight vector

def compute_interval_probs_weight_int(arg):
    vector_l, vector_u, mean, var = arg
    means = mean; # vars = var
    prob_vec = np.zeros(vector_l[0].shape)
    for i in range(len(vector_l[0])):
        for j in range(len(vector_l[0][0])):
            intervals = []
            p = 0.0
            for num_found in range(len(vector_l)):
                interval = [vector_l[num_found][i][j], vector_u[num_found][i][j]]
                interval = [interval]
                p += compute_erf_prob(interval, means[i][j], var[i][j])
            prob_vec[i][j] = p #math.log(p)
    return np.asarray(prob_vec) # what is being returned here is the sum of the cumulative density for each entry in the weight vector


def compute_interval_probs_bias_std(arg):
    vector_intervals, marg, mean, var = arg
    means = mean; #stds = var
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in range(len(vector_intervals[0])):
        intervals = []
        p = 0.0
        for num_found in range(len(vector_intervals)):
            interval = [vector_intervals[num_found][i]-(var[i]*marg), vector_intervals[num_found][i]+(var[i]*marg)]
            interval = [interval]
            #intervals.append(interval)
            p += compute_erf_prob(interval, means[i], var[i])
        prob_vec[i] = p #math.log(p)
    return np.asarray(prob_vec)

def compute_interval_probs_bias_int(arg):
    vector_l, vector_u, mean, var = arg
    means = mean; #stds = var
    prob_vec = np.zeros(vector_l[0].shape)
    for i in range(len(vector_l[0])):
        intervals = []
        p = 0.0
        for num_found in range(len(vector_l)):
            interval = [vector_l[num_found][i], vector_u[num_found][i]]
            interval = [interval]
            p += compute_erf_prob(interval, means[i], var[i])
        #assert(p != 0)
        prob_vec[i] = p #math.log(p)
    return np.asarray(prob_vec)


def compute_probability_subroutine_multiprocess(model, weight_intervals, margin, verbose=True, n_proc=1, correction=False):
    if(verbose == True):
        func = trange
    else:
        func = range

    # compute the probability of weight intervals
    if(correction == False):
        dimensionwise_intervals = weight_intervals
    else:
        dim_intervals_l = np.swapaxes(np.asarray(np.asarray([weight_intervals[0]])),1,0) #weight_intervals[0]
        dim_intervals_u = np.swapaxes(np.asarray(np.asarray([weight_intervals[1]])),1,0) #weight_intervals[1]

    args_bias = []
    args_weights = []
    for i in func(len(model.posterior_mean), desc="Comping in serial"):
        if(i % 2 == 0): # then its a weight vector
            if(correction):
                args_weights.append((dim_intervals_l[i], dim_intervals_u[i], model.posterior_mean[i], np.asarray(model.posterior_var[i])))
            else:
                args_weights.append((dimensionwise_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i])))
        else: # else it is a bias vector
            if(correction):
                args_bias.append((dim_intervals_l[i], dim_intervals_u[i], model.posterior_mean[i], np.asarray(model.posterior_var[i])))
            else:
                args_bias.append((dimensionwise_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i])))

    from multiprocessing import Pool
    #print("Computing for bias")
    proc_pool = Pool(n_proc)
    if(correction):
        ps_bias = proc_pool.map(compute_interval_probs_bias_int, args_bias)
    else:
        ps_bias = proc_pool.map(compute_interval_probs_bias_std, args_bias)
    proc_pool.close()
    proc_pool.join()


    #print("Computing for weight")
    proc_pool = Pool(n_proc)
    if(correction):
        ps_weight = proc_pool.map(compute_interval_probs_weight_int, args_weights)
    else:
        ps_weight = proc_pool.map(compute_interval_probs_weight_std, args_weights)
    proc_pool.close()
    proc_pool.join()

    import itertools
    ps_bias = np.concatenate(ps_bias).ravel()
    ps_weight = np.asarray(list(itertools.chain(*(itertools.chain(*ps_weight)))))

    full_p = 1.0 # converted to log probability
    full_p *= np.prod(ps_bias)
    full_p *= np.prod(ps_weight)
    if(full_p.size != 1):
#        full_p = np.prod(full_p)
#        full_p = np.prod(full_p)
        full_p = np.prod(full_p)
#        print("Interval Prob: ", full_p)
    return full_p


def compute_probability_subroutine(args): #(model, weight_intervals, margin, verbose=True, n_proc=1, correction=False):
    mean, var, weight_intervals, margin, verbose,  n_proc, correction = args
    func = range
    # compute the probability of weight intervals
    if(correction == False):
        dimensionwise_intervals = weight_intervals
    else:
        dim_intervals_l = np.swapaxes(np.asarray(np.asarray([weight_intervals[0]])),1,0) #weight_intervals[0]
        dim_intervals_u = np.swapaxes(np.asarray(np.asarray([weight_intervals[1]])),1,0) #weight_intervals[1]

    args_bias = []
    args_weights = []
    ps_weight = []
    ps_bias = []
    for i in func(len(mean)):
        if(i % 2 == 0): # then its a weight vector
            if(correction):
                ps_weight.append(compute_interval_probs_weight_int((dim_intervals_l[i], dim_intervals_u[i], mean[i], np.asarray(var[i]))))
            else:
                ps_weight.append(compute_interval_probs_weight_std((dimensionwise_intervals[i], margin, mean[i], np.asarray(var[i]))))
        else: # else it is a bias vector
            if(correction):
                ps_bias.append(compute_interval_probs_bias_int((dim_intervals_l[i], dim_intervals_u[i], mean[i], np.asarray(var[i]))))
            else:
                ps_bias.append(compute_interval_probs_bias_std((dimensionwise_intervals[i], margin, mean[i], np.asarray(var[i]))))

    import itertools
    ps_bias = np.concatenate(ps_bias).ravel()
    ps_weight = np.asarray(list(itertools.chain(*(itertools.chain(*ps_weight)))))
    full_p = 1.0
    full_p *= np.prod(ps_bias)
    full_p *= np.prod(ps_weight)
    #print("Prob: ", full_p)
    #print("FULL P", full_p)
    if(full_p.size != 1):
 #       full_p = np.prod(full_p)
 #       full_p = np.prod(full_p)
        full_p = np.prod(full_p)
 #       print("Interval Prob: ", full_p)
    return full_p

def compute_probability_bonferroni_n(model, weight_intervals, margin, depth, max_depth, current_approx, verbose=True, n_proc=30):
    probability_args = []
    for combination in itertools.combinations(range(len(weight_intervals)), depth):
        # intersection of first two
        int_l, int_u = intersect_intervals(weight_intervals[combination[0]], weight_intervals[combination[1]], margin, model.posterior_var)
        for c in range(2, len(combination)):
            # intersection of the rest
            int_l, int_u = intersection_bounds(int_l, int_u, weight_intervals[c], margin, model.posterior_var)
        probability_args.append((model.posterior_mean, model.posterior_var, [int_l, int_u], 0.0, verbose, n_proc, True))
    print("Depth %s has %s intersections"%(depth, len(probability_args)))
    proc_pool = Pool(n_proc)
    stage1p = []
    for result in tqdm.tqdm(proc_pool.imap_unordered(compute_probability_subroutine, probability_args), total=len(probability_args)):
        stage1p.append(result)
    proc_pool.close()
    proc_pool.join()
    p1 = sum(stage1p)

    print("Depth %s prob: %s"%(depth, p1*(-1)**(depth-1)))
    current_approx = current_approx + p1*(-1)**(depth-1)
    print("Current approximation: %s"%(current_approx))
    return current_approx

def compute_probability_bonferroni(model, weight_intervals, margin, max_depth=4, verbose=True, n_proc=30):
    print("About to compute intersection for this many intervals: ", len(weight_intervals))
    stage1_args = []
    stage2_args = []
    int_l, int_u = [], []
    for wi in trange(len(weight_intervals), desc="Computing intersection weights"):
        stage1_args.append((model.posterior_mean, model.posterior_var, np.swapaxes(np.asarray([weight_intervals[wi]]),1,0), margin, verbose, n_proc, False))

    print("Depth 1 has %s intersections"%(len(stage1_args)))
    proc_pool = Pool(n_proc)
    stage1p = []
    #stage1p = proc_pool.map(compute_probability_subroutine, stage1_args)
    for result in tqdm.tqdm(proc_pool.imap_unordered(compute_probability_subroutine, stage1_args), total=len(stage1_args)):
        stage1p.append(result)
    proc_pool.close()
    proc_pool.join()
    p1 = sum(stage1p)
    print("Depth 1 prob: ", p1)

    current_approx = compute_probability_bonferroni_n(model, weight_intervals, margin, 2, max_depth, p1, verbose, n_proc)
    print("Depth 2 prob:: ", current_approx)

    if(max_depth >= 3):
        approx = current_approx
        for i in range(3, max_depth+1):
            approx = compute_probability_bonferroni_n(model, weight_intervals, margin, i, max_depth, approx, verbose, n_proc)
            print("Got this approximation: ", approx)
        return approx
    else:
        return current_approx

def compute_decision_bonferroni_n(model, weight_intervals, values, margin, depth, max_depth, current_prob, current_dec, verbose=True, n_proc=30, y_inf=0.0):
    probability_args = []
    output_values = []
    for combination in itertools.combinations(range(len(weight_intervals)), depth):
        # intersection of first two
        if(y_inf == 1.0): # upperbounding
            val_l = min(values[combination[0]], values[combination[1]])
        else: # lowerbounding
            val_l = max(values[combination[0]], values[combination[1]])
        int_l, int_u = intersect_intervals(weight_intervals[combination[0]], weight_intervals[combination[1]], margin, model.posterior_var)
        for c in range(2, len(combination)):
            # intersection of the rest
            if(y_inf == 1.0):
                val_l = max(val_l, values[combination[c]])
            else:
                val_l = min(val_l, values[combination[c]])
            int_l, int_u = intersection_bounds(int_l, int_u, weight_intervals[c], margin, model.posterior_var)
        probability_args.append((model.posterior_mean, model.posterior_var, [int_l, int_u], 0.0, verbose, n_proc, True))
        output_values.append(val_l)
    print("Depth %s has %s intersections"%(depth, len(probability_args)))
    proc_pool = Pool(n_proc)
    stage1p = []
    for result in tqdm.tqdm(proc_pool.imap_unordered(compute_probability_subroutine, probability_args), total=len(probability_args)):
        stage1p.append(result)
    proc_pool.close()
    proc_pool.join()
    d1 = sum(np.multiply(stage1p,output_values))
    p1 = sum(stage1p)

    current_prob = current_prob + p1*(-1)**(depth-1)
    current_dec = current_dec + d1*(-1)**(depth-1)
    print("Current prob: ", current_prob, " Current dec: ", current_dec)
    #print("Current approximation: %s"%(current_approx))
    return current_prob, current_dec


def compute_decision_bonferroni(model, weight_intervals, values, margin, max_depth=4, verbose=True, n_proc=30, y_inf=0.0):
    print("About to compute intersection for this many intervals: ", len(weight_intervals))
    print("GOT THIS MANY VALUES: ", len(values), values)
    stage1_args = []
    int_l, int_u = [], []
    for wi in trange(len(weight_intervals), desc="Computing intersection weights"):
        stage1_args.append((model.posterior_mean, model.posterior_var, np.swapaxes(np.asarray([weight_intervals[wi]]),1,0), margin, verbose, n_proc, False))
    print("Depth 1 has %s intersections"%(len(stage1_args)))

    proc_pool = Pool(n_proc)
    stage1p = []
    for result in tqdm.tqdm(proc_pool.imap_unordered(compute_probability_subroutine, stage1_args), total=len(stage1_args)):
        stage1p.append(result)
    proc_pool.close()
    proc_pool.join()
    print(stage1p, values)
    d1 = sum(np.multiply(stage1p,values))
    p1 = sum(stage1p)
    print("Depth 1 prob: ", p1, "logit val: ", d1)

    current_prob, current_dec = compute_decision_bonferroni_n(model, weight_intervals, values, margin, 2, max_depth, p1, d1, verbose, n_proc)

    p_approx = current_prob
    d_approx = current_dec
    if(max_depth >= 3):
        for i in range(3, max_depth+1):
            p_approx, d_approx = compute_decision_bonferroni_n(model, weight_intervals, values, margin, i, max_depth, p_approx, d_approx, verbose, n_proc)
            print("Got this approximation: ", p_approx)
        return d_approx*p_approx + ((1-p_approx)*y_inf)
    else:
        return d_approx*p_approx + ((1-p_approx)*y_inf)

# Using Fréchet inequalities
def compute_probability_frechet(model, weight_intervals, margin, verbose=True, n_proc=30):
    # compute the probability of events
    overapprox = 0.0
    n = len(weight_intervals)
    probs = []
    for weight_interval in weight_intervals:
        weight_interval = np.asarray([weight_interval])
        p = compute_probability_subroutine(model, np.swapaxes(np.asarray(weight_interval),1,0), margin, verbose, n_proc)
        overapprox += p
        probs.append(p)
    print("Overapproximation: ", overapprox)
    frechet = max(0, overapprox - (n-1))
    print("Frechet Approx Lower: ", frechet)
    print("Frechet Approx Upper: ", max(probs))
    return frechet

# ============
# Full routine
# ============
def prob_veri(model, s0, s1, w_marg, samples, predicate, i0=0, depth=4):
    assert(samples >= (depth)) #, "Ensure samples > depth. Otherwise probability computation is unsound.")
    w_marg = w_marg*2
    safe_weights = []
    for i in trange(samples, desc="Checking Samples"):
        model.model.set_weights(model.sample())
        ol, ou = IBP_prob(model, s0, s1, model.model.get_weights(), w_marg)
        if(predicate(np.squeeze(s0), np.squeeze(s1), np.squeeze(ol), np.squeeze(ou))):
            safe_weights.append(model.model.get_weights())
    print("Found %s safe intervals"%(len(safe_weights)))
    p = compute_probability_bonferroni(model, safe_weights, w_marg, max_depth=depth)
    return p

def decision_veri(model, s0, s1, w_marg, samples, predicate, value, i0=0, depth=4):
    assert(samples >= (depth)) #, "Ensure samples > depth. Otherwise probability computation is unsound.")
    w_marg = w_marg*2
    safe_weights = []
    logit_values = []
    for i in trange(samples, desc="Checking Samples"):
        model.model.set_weights(model.sample())
        ol, ou = IBP_prob(model, s0, s1, model.model.get_weights(), w_marg, logits=True)
        #if(predicate(np.squeeze(s0), np.squeeze(s1), np.squeeze(ol), np.squeeze(ou))):
        #print("IS THIS GETTING PRINTED?")
        val = value(np.squeeze(s0), np.squeeze(s1), np.squeeze(ol), np.squeeze(ou))
        print("-- ", val)
        logit_values.append(value(np.squeeze(s0), np.squeeze(s1), np.squeeze(ol), np.squeeze(ou)))
        safe_weights.append(model.model.get_weights())
    print("Found %s safe intervals"%(len(safe_weights)))
    logit_values = np.asarray(logit_values)
    p = compute_decision_bonferroni(model, safe_weights, logit_values, w_marg, max_depth=depth, y_inf=0.0)
    return p


def decision_veri_upper(model, s0, s1, w_marg, samples, predicate, value, depth=4, loss_fn=tf.keras.losses.MeanAbsoluteError()):
    assert(samples >= (depth)) #, "Ensure samples > depth. Otherwise probability computation is unsound.")
    w_marg = w_marg*2
    safe_weights = []
    safe_outputs = []
    logit_values = []
    inp = s0+s1/2
    eps = s0-s1/2
    for i in trange(samples, desc="Checking Samples"):
        model.model.set_weights(model.sample())
        # Insert attacks here
        adv = attacks.PGD(model, inp, loss_fn, eps, direction=-2, num_models=-1, order=1, num_steps=7)
        ol, ou = IBP_prob(model, s0, s1, model.model.get_weights(), w_marg)
        logit_values.append(value(np.squeeze(adv), np.squeeze(adv), np.squeeze(ol), np.squeeze(ou)))
        #ol, ou = IBP_prob(model, s0, s1, model.model.get_weights(), w_marg)
        #logit_values.append(value(np.squeeze(s0), np.squeeze(s1), np.squeeze(ol), np.squeeze(ou)))
        safe_weights.append(model.model.get_weights())
    print("Found %s safe intervals"%(len(safe_weights)))
    p = compute_decision_bonferroni(model, safe_weights, logit_values, w_marg, max_depth=depth, y_inf=1.0)
    return p

def prob_veri_upper(model, s0, s1, w_marg, samples, predicate, depth=4, loss_fn=tf.keras.losses.MeanAbsoluteError()):
    assert(samples >= (depth)) #, "Ensure samples > depth. Otherwise probability computation is unsound.")
    w_marg = w_marg*2
    safe_weights = []
    safe_outputs = []
    inp = s0+s1/2
    eps = s0-s1
    for i in trange(samples, desc="Checking Samples"):
        model.model.set_weights(model.sample())
        # Insert attacks here
        #if(i%mod_option == 0):
        adv = attacks.PGD(model, inp, loss_fn, eps, direction=-1, num_models=-1, order=1, num_steps=25)
        ol, ou = IBP_prob(model, adv, adv, model.model.get_weights(), w_marg)
        unsafe = predicate(np.squeeze(adv), np.squeeze(adv), np.squeeze(ol), np.squeeze(ou))
        if(unsafe):
            safe_weights.append(model.model.get_weights())
    print("Found %s safe intervals"%(len(safe_weights)))
    #if(len(safe_weights) < 2):
    #    return 0.0, -1
    #p = compute_probability(model, np.swapaxes(np.asarray(safe_weights),1,0), w_marg)
    p = compute_probability_bonferroni(model, safe_weights, w_marg, max_depth=depth)
    return p #, np.squeeze(safe_outputs)

def prob_veri_samp(model, s0, s1, samples, predicate, y_scaler, i0=0):
    #print("Verification got inputs: ", s0, s1)
    #print("Given states 1: ", np.squeeze(s0)[0:2], np.squeeze(s1)[0:2])
    probability = 0
    num_samples = sum(model.frequency)
    num_unique = len(model.frequency)
    safe_weights = []
    outs = []
    for i in trange(min(len(model.frequency), samples), desc="Checking Samples", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        loaded_weights = np.load(model.path_to_model+"/samples/sample_%s.npy"%(i), allow_pickle=True)
        model.model.set_weights(loaded_weights)
        ol, ou = IBP_samp(model, s0, s1, loaded_weights, weight_margin=0.0, logits=False)
        if(predicate(np.squeeze(s0), np.squeeze(s1), np.squeeze(lower), np.squeeze(upper))):
            outs.append([lower,upper])
            probability += (model.frequency[i]/num_samples)
            #print("Adding safe probability: ", (model.frequency[i]/num_samples))
    return probability, np.squeeze(outs)

def decision_veri_samp(model, s0, s1, w_marg, samples, predicate, value, i0=0, depth=4):
    num_samples = model.num_post_samps
    outval = 0
    for i in trange(min(len(model.frequency), len(model.frequency)), desc="Checking Samples", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        loaded_weights = np.load(model.path_to_model+"/samples/sample_%s.npy"%(i), allow_pickle=True)
        model.model.set_weights(loaded_weights)
        ol, ou = IBP_samp(model, s0, s0, loaded_weights, weight_margin=0.0, logits=True)
        #print(value(np.squeeze(s0), np.squeeze(s1), np.squeeze(ol), np.squeeze(ou)))
        outval += model.frequency[i]  * value(np.squeeze(s0), np.squeeze(s1), np.squeeze(ol), np.squeeze(ou))
    print("Done with DecVeriHMC", outval)
    #print("Found %s safe intervals"%(len(safe_weights)))
    #logit_values = np.asarray(logit_values)
    #p = probability #compute_decision_bonferroni(model, safe_weights, logit_values, w_marg, max_depth=depth, y_inf=0.0)
    return outval

