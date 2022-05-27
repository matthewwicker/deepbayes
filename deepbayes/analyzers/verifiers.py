# Author: Matthew Wicker


from statsmodels.stats.proportion import proportion_confint
import math
import numpy as np
import tensorflow as tf
from tqdm import trange
from . import attacks

def propagate_conv2d(W, b, x_l, x_u, marg=0, b_marg=0):
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
    #marg = tf.divide(marg, 2)
    #b_marg = tf.divide(marg, 2)
    x_mu = tf.cast(tf.divide(tf.math.add(x_u, x_l), 2), dtype=tf.float64)
    x_r =  tf.cast(tf.divide(tf.math.subtract(x_u, x_l), 2), dtype=tf.float64)
    W_mu = tf.cast(W, dtype=tf.float64)
    W_r =  tf.cast(marg, dtype=tf.float64)
    if(type(marg) == int):
        W_r = 0.0 * W_mu
    b_u =  tf.cast(b + b_marg, dtype=tf.float64)
    b_l =  tf.cast(b - b_marg, dtype=tf.float64)
    #h_mu = tf.math.add(tf.matmul(x_mu, W_mu), b_mu)
    h_mu = tf.matmul(x_mu, W_mu)
    x_rad = tf.matmul(x_r, tf.math.abs(W_mu))
    W_rad = tf.matmul(tf.abs(x_mu), W_r)
    Quad = tf.matmul(tf.abs(x_r), tf.abs(W_r))
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
    W_l, W_u = W-marg, W+marg    	#Use eps as small symetric difference about the mean
    b_l, b_u = b-b_marg, b+b_marg   	#Use eps as small symetric difference about the mean 
    h_max = np.zeros(len(W[0])) 	#Placeholder variable for return value
    h_min = np.zeros(len(W[0])) 	#Placeholder variable for return value
    for i in range(len(W)):     	#This is literally just a step-by-step matrix multiplication
        for j in range(len(W[0])): 	# where we are taking the min and max of the possibilities
            out_arr = [W_l[i][j]*x_l[i], W_l[i][j]*x_u[i],
                       W_u[i][j]*x_l[i], W_u[i][j]*x_u[i]]
            h_min[j] += min(out_arr)
            h_max[j] += max(out_arr)
    h_min = h_min + b_l
    h_max = h_max + b_u
    return h_min, h_max         #Return the min and max of the intervals.
                                #(dont forget to apply activation function after)


def IBP(model, inp, weights, eps, predict=False):
    if(predict == False):
        h_u = tf.clip_by_value(tf.math.add(inp, eps), 0.0, 1.0)
        h_l = tf.clip_by_value(tf.math.subtract(inp, eps), 0.0, 1.0)
    else:
        h_u = inp + eps
        h_l = inp - eps
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
            h_l, h_u = propagate_interval(w, b, h_l, h_u)
            activate = True
        elif(len(w.shape) == 4):
            h_l, h_u = propagate_conv2d(w, b, h_l, h_u)
            activate = True
        if(predict == False and i >= len(layers)-1):
            continue
        else:
            h_l = model.model.layers[i].activation(h_l)
            h_u = model.model.layers[i].activation(h_u)
    return h_l, h_u

def IBP_state(model, s0, s1, weights, weight_margin=0, logits=True):
    h_l = s0
    h_u = s1
    layers = model.model.layers
    offset = 0
    for i in range(len(layers)):
        if(len(layers[i].get_weights()) == 0):
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
        #h_l, h_u = propagate_interval(w, b, h_l, h_u, marg=marg, b_marg=b_marg)
        if(i < len(layers)-1):
            h_l = model.model.layers[i].activation(h_l)
            h_u = model.model.layers[i].activation(h_u)
    return h_l, h_u

# Code for merging overlapping intervals. Taken from here: 
# https://stackoverflow.com/questions/49071081/merging-overlapping-intervals-in-python
# This function simple takes in a list of intervals and merges them into all 
# continuous intervals and returns that list 
def merge_intervals(intervals):
    sorted_intervals = sorted(intervals)
    interval_index = 0
    intervals = np.asarray(intervals)
    for  i in sorted_intervals:
        if i[0] > sorted_intervals[interval_index][1]:
            interval_index += 1
            sorted_intervals[interval_index] = i
        else:
            sorted_intervals[interval_index] = [sorted_intervals[interval_index][0], i[1]]
    return sorted_intervals[:interval_index+1] 



"""
Given a set of disjoint intervals, compute the probability of a random
sample from a guassian falling in these intervals. (Taken from lemma)
of the document
"""
import math
from scipy.special import erf
def compute_erf_prob(intervals, mean, var):
    prob = 0.0
    for interval in intervals:
#        val1 = erf((mean-interval[0])/(math.sqrt(2)*(var)))
#        val2 = erf((mean-interval[1])/(math.sqrt(2)*(var)))
        val1 = erf((mean-interval[0])/(math.sqrt(2*(var))))
        val2 = erf((mean-interval[1])/(math.sqrt(2*(var))))
        prob += 0.5*(val1-val2)
    return prob

"""
Given a set of possibly overlapping intervals:
    - Merge all intervals into maximum continuous, disjoint intervals
    - compute probability of these disjoint intervals
    - do this for ALL values in a weight matrix
"""
def compute_interval_probs_weight(vector_intervals, marg, mean, var):
    means = mean; # vars = var
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in trange(len(vector_intervals[0])):
        for j in range(len(vector_intervals[0][0])):
            intervals = []
            for num_found in range(len(vector_intervals)):
                interval = [vector_intervals[num_found][i][j]-(var[i][j]*marg), vector_intervals[num_found][i][j]+(var[i][j]*marg)]
                intervals.append(interval)
            p = compute_erf_prob(merge_intervals(intervals), means[i][j], var[i][j])
            prob_vec[i][j] = p
    return np.asarray(prob_vec)



"""

========================== SPEED TIME ===============================

"""

def IBP_full_multiproc(args):
    actives, s0, s1, weights, weight_margin, predicate, posterior_var = args
    h_l = s0
    h_u = s1
    layers = int(len(posterior_var)/2)
    offset = 0
    for i in range(layers):
        w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
        sigma = posterior_var[2*(i-offset)]
        b_sigma = posterior_var[2*(i-offset)+1]
        marg = weight_margin*sigma
        b_marg = weight_margin*b_sigma
        h_l, h_u = propagate_interval(w, b, h_l, h_u, marg=marg, b_marg=b_marg)
        #print("Layer %s Bounds: "%(i), h_l, h_u)
        h_l = actives[i](h_l) #model.model.layers[i].activation(h_l)
        h_u = actives[i](h_u) #model.model.layers[i].activation(h_u)

    ol, ou = h_l, h_u
    lower = np.squeeze(s0)[0:len(ol)] + ol;
    upper = np.squeeze(s1)[0:len(ou)] + ou

    if(predicate(np.squeeze(s0), np.squeeze(s1), np.squeeze(lower), np.squeeze(upper))):
        ol = np.squeeze(ol); ou = np.squeeze(ou)
        return [lower, upper]
    else:
        return None

def compute_interval_probs_weight_m(arg):
    vector_intervals, marg, mean, var = arg
    means = mean; # vars = var
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in trange(len(vector_intervals[0])):
        for j in range(len(vector_intervals[0][0])):
            intervals = []
            for num_found in range(len(vector_intervals)):
                interval = [vector_intervals[num_found][i][j]-(var[i][j]*marg), vector_intervals[num_found][i][j]+(var[i][j]*marg)]
                intervals.append(interval)
            p = compute_erf_prob(merge_intervals(intervals), means[i][j], var[i][j])
            prob_vec[i][j] = p
    return np.asarray(prob_vec)

def compute_interval_probs_bias_m(arg):
    vector_intervals, marg, mean, var = arg
    means = mean; #stds = var
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in range(len(vector_intervals[0])):
        intervals = []
        for num_found in range(len(vector_intervals)):
            #!*! Need to correct and make sure you scale margin
            interval = [vector_intervals[num_found][i]-(var[i]*marg), vector_intervals[num_found][i]+(var[i]*marg)]
            intervals.append(interval)
        p = compute_erf_prob(merge_intervals(intervals), means[i], var[i])
        prob_vec[i] = p
    return np.asarray(prob_vec).tolist()


def compute_probability(model, weight_intervals, margin, verbose=True, n_proc=30):
    full_p = 1.0
    if(verbose == True):
        func = range
    else:
        func = range
    args_bias = []
    args_weights = []
    for i in func(len(model.posterior_mean)):
        if(i % 2 == 0): # then its a weight vector
            #p = compute_interval_probs_weight(weight_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i]))
            args_weights.append((weight_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i])))
        else: # else it is a bias vector
            #p = compute_interval_probs_bias(weight_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i]))
            args_bias.append((weight_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i])))

    from multiprocessing import Pool
    print("Computing for bias")
    proc_pool = Pool(n_proc)
    ps_bias = proc_pool.map(compute_interval_probs_bias_m, args_bias)
    proc_pool.close()
    proc_pool.join()
    print("Computing for weight")
    proc_pool = Pool(n_proc)
    ps_weight = proc_pool.map(compute_interval_probs_weight_m, args_weights)
    proc_pool.close()
    proc_pool.join()

    import itertools
    ps_bias = np.concatenate(ps_bias).ravel()
    ps_weight = np.asarray(list(itertools.chain(*(itertools.chain(*ps_weight)))))

    full_p *= np.prod(ps_bias)
    full_p *= np.prod(ps_weight)
    return full_p




# ============

# Computing with intersections

# ============

def intersect_intervals(wi_a, wi_b, margin, var):
    intersection = []
    for i in range(len(wi_a)):
        if(wi_a[i][1] < wi_b[i][0]):
            # add this dimension of intersection
            intersection.append([max(wi_a[0], wi_b[0]), min(wi_a[1], wi_b[1])])
        else:
            return -1
    return intersection


def compute_interval_probs_weight_dep(arg):
    vector_intervals, marg, mean, var = arg
    means = mean; # vars = var
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in trange(len(vector_intervals[0])):
        for j in range(len(vector_intervals[0][0])):
            intervals = []
            p = 0.0
            for num_found in range(len(vector_intervals)):
                interval = [vector_intervals[num_found][i][j]-(var[i][j]*marg), vector_intervals[num_found][i][j]+(var[i][j]*marg)]
                intervals.append(interval)
                p += compute_erf_prob(interval, means[i][j], var[i][j])
            prob_vec[i][j] = p
    return np.asarray(prob_vec).tolist() # what is being returned here is the sum of the cumulative density for each entry in the weight vector

def compute_interval_probs_bias_dep(arg):
    vector_intervals, marg, mean, var = arg
    means = mean; #stds = var
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in range(len(vector_intervals[0])):
        intervals = []
        p = 0.0
        for num_found in range(len(vector_intervals)):
            interval = [vector_intervals[num_found][i]-(var[i]*marg), vector_intervals[num_found][i]+(var[i]*marg)]
            intervals.append(interval)
            p += compute_erf_prob(interval, means[i], var[i])
        prob_vec[i] = p
    return np.asarray(prob_vec).tolist()


def compute_probability_subroutine(model, weight_intervals, margin, verbose=True, n_proc=30):
    if(verbose == True):
        func = trange
    else:
        func = range

    # compute the probability of weight intervals
    dimensionwise_intervals = np.swapaxes(np.asarray(safe_weights),1,0)
    args_bias = []
    args_weights = []
    for i in func(len(model.posterior_mean)):
        if(i % 2 == 0): # then its a weight vector
            args_weights.append((dimensionwise_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i])))
        else: # else it is a bias vector
            args_bias.append((dimensionwise_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i])))

    from multiprocessing import Pool
    print("Computing for bias")
    proc_pool = Pool(n_proc)
    ps_bias = proc_pool.map(compute_interval_probs_bias_dep, args_bias)
    proc_pool.close()
    proc_pool.join()
    print("Computing for weight")
    proc_pool = Pool(n_proc)
    ps_weight = proc_pool.map(compute_interval_probs_weight_dep, args_weights)
    proc_pool.close()
    proc_pool.join()

    import itertools
    ps_bias = np.concatenate(ps_bias).ravel()
    ps_weight = np.asarray(list(itertools.chain(*(itertools.chain(*ps_weight)))))
    full_p = 1
    full_p *= np.prod(ps_bias)
    full_p *= np.prod(ps_weight)
    return full_p


def compute_prob_intersect_full(model, weight_intervals, margin, verbose=True, n_proc=30):
    intersections = []
    for wi in range(len(weight_intervals)):
        for wj in range(len(weight_intervals)):
            if(wi == wj):
                continue
            result = intersect_intervals(weight_intervals[wi], weight_intervals[wj], margin, model.posterior_var)
            if(type(result) == int):
                continue
            else:
                intersections.append(result)

    # compute the probability of intersections
    overapprox = compute_probability_subroutine(model, weight_intervals, margin, verbose, n_proc)
    # return the subtraction of the two
    correction = compute_probability_subroutine(model, intersections, margin, verbose, n_proc)
    return None

"""

========================== SPEED TIME ===============================

"""



"""
Given a set of possibly overlapping intervals:
    - Merge all intervals into continuous, disjoint intervals
    - compute probability of these disjoint intervals
    - do this for ALL values in a *flat* bias matrix (vector)
"""
def compute_interval_probs_bias(vector_intervals, marg, mean, var):
    means = mean; #stds = var
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in range(len(vector_intervals[0])):
        intervals = []
        for num_found in range(len(vector_intervals)):
            #!*! Need to correct and make sure you scale margin
            interval = [vector_intervals[num_found][i]-(var[i]*marg), vector_intervals[num_found][i]+(var[i]*marg)]
            intervals.append(interval)
        p = compute_erf_prob(merge_intervals(intervals), means[i], var[i])
        prob_vec[i] = p
    return np.asarray(prob_vec)
        
def compute_probability_s(model, weight_intervals, margin, verbose=True):
    full_p = 1.0
    if(verbose == True):
        func = trange
    else:
        func = range
    # for every weight vector, get the intervals
    for i in func(len(model.posterior_mean)):
        if(i % 2 == 0): # then its a weight vector
#            p = compute_interval_probs_weight(weight_intervals[i], margin, model.posterior_mean[i], np.square(model.posterior_var[i]))
            p = compute_interval_probs_weight(weight_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i]))
        else:
#            p = compute_interval_probs_bias(weight_intervals[i], margin, model.posterior_mean[i], np.square(model.posterior_var[i]))
            p = compute_interval_probs_bias(weight_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i]))
        #print("Average weight prob: ", np.mean(p))
        p = np.prod(p)
        full_p *= p
    return full_p

def IBP_prob(model, s0, s1, w_marg, samples, predicate, i0=0, inflate=1.0):
    w_marg = w_marg**2
    safe_weights = []
    safe_outputs = []
    for i in trange(samples, desc="Checking Samples"):
        model.model.set_weights(model.sample(inflate=inflate))
        ol, ou = IBP_state(model, s0, s1, model.model.get_weights(), w_marg)
        if(predicate(np.squeeze(s0), np.squeeze(s1), np.squeeze(ol), np.squeeze(ou))):
            safe_weights.append(model.model.get_weights())
            ol = np.squeeze(ol); ou = np.squeeze(ou)
            #lower = np.squeeze(s0)[0:len(ol)] + ol; upper = np.squeeze(s1)[0:len(ou)] + ou
            safe_outputs.append([-1,1]) # This is used ONLY for control loops which needs its own verification section
    print("Found %s safe intervals"%(len(safe_weights)))
    if(len(safe_weights) < 2):
        return 0.0, -1
    p = compute_probability(model, np.swapaxes(np.asarray(safe_weights),1,0), w_marg)
    return p, np.squeeze(safe_outputs)



def IBP_upper(model, s0, s1, w_marg, samples, predicate, loss_fn, eps, inputs=[], inflate=1.0, mod_option=10):
    w_marg = w_marg**2
    safe_weights = []
    safe_outputs = []
    for i in trange(samples, desc="Checking Samples"):
        model.model.set_weights(model.sample(inflate=inflate))
        # Insert attacks here
        if(i%mod_option == 0):
            adv = attacks.FGSM(model, s0, loss_fn, eps, direction=-1, num_models=-1, order=1)
            ol, ou = IBP_state(model, adv, adv, model.model.get_weights(), w_marg)
            unsafe = predicate(np.squeeze(adv), np.squeeze(adv), np.squeeze(ol), np.squeeze(ou))
        else:
            unsafe = False
        for inp in inputs:
            if(unsafe == True):
                break
            ol, ou = IBP_state(model, inp, inp, model.model.get_weights(), w_marg)
            if(predicate(np.squeeze(inp), np.squeeze(inp), np.squeeze(ol), np.squeeze(ou))):
                unsafe = True
                break

        if(unsafe):
            safe_weights.append(model.model.get_weights())
            ol = np.squeeze(ol); ou = np.squeeze(ou)
    print("Found %s safe intervals"%(len(safe_weights)))
    if(len(safe_weights) < 2):
        return 0.0, -1
    p = compute_probability(model, np.swapaxes(np.asarray(safe_weights),1,0), w_marg)
    return p, np.squeeze(safe_outputs)


def IBP_uncert(model, s0, s1, w_marg, samples, predicate, i0=0, inflate=1.0):
    #def predicate_uncertain(iml, imu, ol, ou):
    # Step 1 : compute the max difference of the logit weights in the softmax layer (clip above 0)
    # Step 2 : compute the min difference of the logit weights in the softmax layer (clip above 0)
    # Step 3 : compute the difference in biases for the two classes
    # Step 4 : affine pass of upper and lower bounds through those values
    # Step 5 : profit
    w_marg = w_marg**2
    safe_weights = []
    safe_outputs = []
    for i in range(samples):
        model.model.set_weights(model.sample(inflate=inflate))
        checks = []
        softmax_diff = IBP_conf(model, s0, s1, model.model.get_weights(), w_marg)
        uncertain = predicate(np.squeeze(s0), np.squeeze(softmax_diff))	
        if(uncertain):
            safe_weights.append(model.model.get_weights())
    print("Found %s safe intervals"%(len(safe_weights)))
    if(len(safe_weights) < 2):
        return 0.0, -1
    p = compute_probability(model, np.swapaxes(np.asarray(safe_weights),1,0), w_marg)
    return p, np.squeeze(safe_outputs)


def IBP_prob_w(model, s0, s1, w_marg, w, predicate, i0=0):
    model.model.set_weights(model.sample())
    ol, ou = IBP_state(model, s0, s1, w, w_marg)
    if(predicate(np.squeeze(s0)[i0:i0+2], np.squeeze(s1)[i0:i0+2], np.squeeze(ol)[i0:i0+2], np.squeeze(ou)[i0:i0+2])):
        p = compute_probability(model, np.swapaxes(np.asarray([w]),1,0), w_marg)
        return p, -1
    else:
        return 0.0, -1



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
def massart_bound_check(model, inp, eps, predicate, **kwargs):
    delta = kwargs.get('delta', 0.3)
    alpha = kwargs.get('alpha', 0.05)
    confidence = kwargs.get('confidence', 0.95)
    verbose = kwargs.get('verbose', False)
    verify = kwargs.get('verify', True)
    classification = kwargs.get('classification', False)
    decision = kwargs.get('decision', False)
    cls = kwargs.get('cls', None)
    if(not classification):
        attack_loss = kwargs.get('loss_fn', tf.keras.losses.mean_squared_error)
    else:
        attack_loss = kwargs.get('loss_fn', tf.keras.losses.sparse_categorical_crossentropy)
    chernoff_override = kwargs.get('chernoff', False)

    atk_locs = []
    epsilon = 1-confidence
    chernoff_bound = math.ceil( (1/(2*epsilon**2)) * math.log(2/delta) )
    print("BayesKeras. Maximum sample bound = %s"%(chernoff_bound))
    successes, iterations, misses = 0.0, 0.0, 0.0
    halting_bound = chernoff_bound
    I = [0,1]
    mean = 0.0*model.predict(inp)
    if(chernoff_override):
        glob_adv = attacks.FGSM(model, inp, attack_loss, eps, samples=35)
    while(iterations <= halting_bound):
        if(iterations > 0 and verbose):
            print("Working on iteration: %s \t Bound: %s \t Param: %s"%(iterations, halting_bound, successes/iterations))  
        model.set_weights(model.sample())
        #print("PREDICTION: ", model._predict(inp))
        if(verify and classification):
            logit_l, logit_u = IBP(model, inp, model.model.get_weights(), eps, predict=False)
            logit_l, logit_u = logit_l.numpy(), logit_u.numpy()
            #print("LOGIT L ", logit_l)
            #print("LOGIT U ", logit_u)
            try:
                v1 = tf.one_hot(cls, depth=10)
                v2 = 1 - tf.one_hot(cls, depth=10)
                worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
            except:
                v1 = tf.one_hot(cls, depth=2)
                v2 = 1 - tf.one_hot(cls, depth=2)
                worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
            from scipy.special import softmax
            worst_case = softmax(worst_case)
            #worst_case = np.exp(worst_case)/sum(np.exp(worst_case))
            #print("WORST: ", worst_case)
        elif(verify and not classification):
            logit_l, logit_u = IBP(model, inp, model.model.get_weights(), eps, predict=True)
            diff_above = logit_u - cls
            diff_below = logit_l - cls
            logit_l = np.asarray(logit_l)
            logit_u = np.asarray(logit_u)
            zeros = 0.0* logit_l
            zeros[np.abs(diff_above) > np.abs(diff_below)] = logit_u[np.abs(diff_above) > np.abs(diff_below)]
            zeros[np.abs(diff_above) <= np.abs(diff_below)] = logit_l[np.abs(diff_above) <= np.abs(diff_below)]
            worst_case = zeros
            #print(logit_l, logit_u, cls, worst_case)

        elif(not verify):
            adv = attacks.FGSM(model, inp, attack_loss, eps, samples=1) #, direction=cls)
            #print(adv-inp)
            worst_case = model._predict(adv)
            #print(worst_case)
        #if(np.argmax(np.squeeze(worst_case)) != cls):
        if(not predicate(worst_case)):
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
        if(hb == -1 or chernoff_override):
            halting_bound = chernoff_bound
        else:
            halting_bound = min(hb, chernoff_bound)
        if(chernoff_override):
            mean += model._predict(glob_adv)
    if(verbose):
        print("Exited becuase %s >= %s"%(iterations, halting_bound))
    if(not chernoff_override):
        print("Mean is returned as zero because massart does not provide valid bounds on the mean.")
    return successes/iterations, iterations, mean/iterations



