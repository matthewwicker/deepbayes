from statsmodels.stats.proportion import proportion_confint
import itertools
import math
import numpy as np
import tensorflow as tf
import tqdm
#from tqdm import tqdm
from tqdm import trange
from . import attacks
import copy 

from multiprocessing import Pool

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

def IBP_prob(model, s0, s1, weights, weight_margin=0, logits=True):
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

# ============

# Computing with intersections

# ============

# Code for merging overlapping intervals. Taken from here: 
# https://stackoverflow.com/questions/49071081/merging-overlapping-intervals-in-python
# This function simple takes in a list of intervals and merges them into all 
# continuous intervals and returns that list 
#def merge_intervals(intervals):
#    sorted_intervals = sorted(intervals)
#    interval_index = 0
#    intervals = np.asarray(intervals)
#    for  i in sorted_intervals:
#        if i[0] > sorted_intervals[interval_index][1]:
#            interval_index += 1
#            sorted_intervals[interval_index] = i
#        else:
#            sorted_intervals[interval_index] = [sorted_intervals[interval_index][0], i[1]]
#    return sorted_intervals[:interval_index+1] 

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
#        val1 = erf((mean-interval[0])/(math.sqrt(2*(var))))
#        val2 = erf((mean-interval[1])/(math.sqrt(2*(var))))
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

def intersection_double_bounds(wa_l, wa_u, wb_l, wb_u):
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


def merge_intervals(wa_l, wa_u, wb_l, wb_u):
    retval_a_l = []
    retval_a_u = []
    retval_b_l = []
    retval_b_u = []
    for l in range(len(wa_l)):
        # Chech subsumption (a):
        if((wa_u[l] < wb_u[l]).all() and (wa_l[l] > wb_l[l]).all()):
            retval_a_l.append(wb_l[l])
            retval_a_u.append(wb_u[l])
            retval_b_l.append(wb_l[l]*0.0)
            retval_b_u.append(wb_u[l]*0.0)
            continue
        # Check subsumption (b):
        if((wb_u[l] < wa_u[l]).all() and (wb_l[l] > wa_l[l]).all()):
            retval_a_l.append(wa_l[l])
            retval_a_u.append(wa_u[l])
            retval_b_l.append(wa_l[l]*0.0)
            retval_b_u.append(wa_u[l]*0.0)
            continue
        # Check intersection:
        component_a_l = np.zeros(wb_l[l].shape)
        component_a_u = np.zeros(wb_l[l].shape)
        component_b_l = np.zeros(wb_l[l].shape)
        component_b_u = np.zeros(wb_l[l].shape)

        intersect_l = np.maximum(wa_l[l], wb_l[l])
        intersect_u = np.minimum(wa_u[l], wb_u[l])
        continuous_l = np.minimum(wa_l[l], wb_l[l])
        continuous_u = np.maximum(wa_u[l], wb_u[l])

        # Where they intersect, we want to set the retval a to their merge and retval b to empty set
        intersection_points = np.where((intersect_u - intersect_l) <= 0, 0, 1)
        component_a_l += continuous_l*intersection_points
        component_a_u += continuous_u*intersection_points
        component_a_l += wa_l[l]*(1-intersection_points)
        component_a_u += wa_u[l]*(1-intersection_points)

        component_b_l += wb_l[l]*(1-intersection_points)
        component_b_u += wb_u[l]*(1-intersection_points)

        retval_a_l.append(component_a_l)
        retval_a_u.append(component_a_u)

        retval_b_l.append(component_b_l)
        retval_b_u.append(component_b_u)
    return retval_a_l, retval_a_u, retval_b_l, retval_b_u


def get_bounds(wi_a, margin, var):
    wi_l = []
    wi_u = []
    for l in range(len(wi_a)):
        wi_a_l = (wi_a[l] - (var[l]*margin)).numpy()
        wi_a_u = (wi_a[l] + (var[l]*margin)).numpy()
        wi_l.append(wi_a_l)
        wi_u.append(wi_a_u)
    return wi_l, wi_u

def symdiff_intervals(wi_a_l, wi_a_u, wi_b, margin, var):
    wn_a_l = []
    wn_a_u = []
    for l in range(len(wi_a_l)):
        wa_u = wi_a_l[l] # Upper bound for these variables
        wa_l = wi_a_u[l] # Lower bound for these variables
        wb_u = (wi_b[l] + (var[l]*margin)).numpy() # Upper bound for these variables
        wb_l = (wi_b[l] - (var[l]*margin)).numpy() # Lower bound for these variables

        # Check for subsumption first
        if((wa_u < wb_u).all() and (wa_l > wb_l).all()):
            assert(False) #fucking hell
        else:
            w_l = np.where(((wb_u > wa_l) & (wb_u < wa_u)), wb_u, wa_l)
            w_u = np.where(((wb_l < wa_u) & (wb_l > wa_l)), wb_l, wa_u)
            wn_a_l.append(np.asarray(w_l))
            wn_a_u.append(np.asarray(w_u))
    return wn_a_l, wn_a_u



"""
This is an interative approach to intersection that I know is correct
and is written out to help me optimize a faster algorithm
Runtime for small problems: 8-12 hours
"""
def intersect_intervals_slow(wi_a, wi_b, margin, var):
    intersection_interval = []
    scaled_marg = margin*var
    for l in range(len(wi_a)): # This iterates 2x the number of layers
        shape = np.shape(wi_a[l]); shape=list(shape); shape.append(2)
        vector_interval = np.zeros(shape)
        #vector_interval = np.expand_dims(vector_interval, axis=-1)
        if(l%2==0): # We know this is a weight vector for a layer
            for i in range(len(wi_a[l])):
                for j in range(len(wi_a[l][i])):
                    wi_a_u = wi_a[l][i][j]+scaled_marg[l][i][j] # Upper bound for this variable
                    wi_a_l = wi_a[l][i][j]-scaled_marg[l][i][j] # Lower bound for this variable
                    wi_b_u = wi_b[l][i][j]+scaled_marg[l][i][j] # Upper bound for this variable
                    wi_b_l = wi_b[l][i][j]-scaled_marg[l][i][j] # Lower bound for this variable
                    # If any of these conditions are satisfied then we have a non-null intersection
                    if(wi_a_u > wi_b_l or wi_b_u > wi_a_l or (wi_a_l < wi_b_l and wi_a_u > wi_b_u) or (wi_b_l < wi_a_l and wi_b_u > wi_a_u)):
                        # In any case, this is the intersection
                        vector_interval[i][j] = [max(wi_a_l, wi_b_l), min(wi_a_u, wi_b_u)]
                    else:
                        # If for even one random variable the intervals dont intersect than the whole thing doesnt intersect
                        return -1
            intersection_interval.append(vector_interval)
        else: # We know this is a bias vector for a layer
            for i in range(len(wi_a[l])):
                wi_a_u = wi_a[l][i]+scaled_marg[l][i] # Upper bound for this variable
                wi_a_l = wi_a[l][i]-scaled_marg[l][i] # Lower bound for this variable
                wi_b_u = wi_b[l][i]+scaled_marg[l][i] # Upper bound for this variable
                wi_b_l = wi_b[l][i]-scaled_marg[l][i] # Lower bound for this variable
                # If any of these conditions are satisfied then we have a non-null intersection
                if(wi_a_u > wi_b_l or wi_b_u > wi_a_l or (wi_a_l < wi_b_l and wi_a_u > wi_b_u) or (wi_b_l < wi_a_l and wi_b_u > wi_a_u)):
                    # In any case, this is the intersection
                    vector_interval[i] = [max(wi_a_l, wi_b_l), min(wi_a_u, wi_b_u)]
                else:
                    return -1
            intersection_interval.append(vector_interval)
    print("found a full intersection")
    return intersection_interval


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
    print("Interval Prob: ", full_p)
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
    return full_p


def compute_probability_bonferroni_n(model, weight_intervals, margin, depth, max_depth, current_approx, verbose=True, n_proc=30):
    #print("In depth: ", depth)
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

    #if(depth == max_depth):
    #    return  (p1 * (-1)**(depth-1)) 
    #else:
    #    total_prob = (p1 * (-1)**(depth-1)) + compute_probability_bonferroni_n(model, weight_intervals, margin, depth+1, max_depth, current_approx, verbose, n_proc)
    #    current_approx = current_approx + compute_probability_bonferroni_n(model, weight_intervals, margin, depth+1, max_depth, current_approx, verbose, n_proc)
    #    print("Why is this the total probability? ", current_approx)
    #    return current_approx


def compute_probability_bonferroni(model, weight_intervals, margin, max_depth=4, verbose=True, n_proc=30):
    #intersections_l, intersections_u = [], []
    print("About to compute intersection for this many intervals: ", len(weight_intervals))
    stage1_args = []
    stage2_args = []
    int_l, int_u = [], []
    for wi in trange(len(weight_intervals), desc="Computing intersection weights"):
        stage1_args.append((model.posterior_mean, model.posterior_var, np.swapaxes(np.asarray([weight_intervals[wi]]),1,0), margin, verbose, n_proc, False))
        #for wj in range(wi+1, len(weight_intervals)):
        #    i_l, i_u = intersect_intervals(weight_intervals[wi], weight_intervals[wj], margin, model.posterior_var)
        #    stage2_args.append((model.posterior_mean, model.posterior_var, [i_l, i_u], 0.0, verbose, n_proc, True))
        #    int_l.append(i_l); int_u.append(i_u)
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
    print("Approx 2 from function: ", current_approx)

    #print("Depth 2 has %s intersections"%(len(stage2_args)))
    #proc_pool = Pool(n_proc)
    #stage2p = [] #proc_pool.map(compute_probability_subroutine, stage2_args)
    #for result in tqdm.tqdm(proc_pool.imap_unordered(compute_probability_subroutine, stage2_args), total=len(stage2_args)):
    #    stage2p.append(result)
    #proc_pool.close()
    #proc_pool.join()
    #p2 = sum(stage2p)

    #print("Depth 2 prob: ", -p2)
    #print("Current Approximation: ", p1 - p2)
    #current_approx = p1 - p2
    if(max_depth >= 3):
        approx = current_approx
        for i in range(3, max_depth+1):
            approx = compute_probability_bonferroni_n(model, weight_intervals, margin, i, max_depth, approx, verbose, n_proc)
            print("Got this approximation: ", approx)
        return approx
    else:
        return p1 - p2


def compute_probability_bonferroni_2(model, weight_intervals, margin, verbose=True, n_proc=30):
    #intersections_l, intersections_u = [], []
    print("About to compute intersection for this many intervals: ", len(weight_intervals))
    stage1_args = []
    stage2_args = []
    for wi in trange(len(weight_intervals), desc="Computing intersection weights"):
        stage1_args.append((model.posterior_mean, model.posterior_var, np.swapaxes(np.asarray([weight_intervals[wi]]),1,0), margin, verbose, n_proc, False))
        for wj in range(wi+1, len(weight_intervals)):
            i_l, i_u = intersect_intervals(weight_intervals[wi], weight_intervals[wj], margin, model.posterior_var)
            stage2_args.append((model.posterior_mean, model.posterior_var, [i_l, i_u], 0.0, verbose, n_proc, True))

    proc_pool = Pool(n_proc)
    stage1p = []
    #stage1p = proc_pool.map(compute_probability_subroutine, stage1_args)
    for result in tqdm.tqdm(proc_pool.imap_unordered(compute_probability_subroutine, stage1_args), total=len(stage1_args)):
        stage1p.append(result)
    proc_pool.close()
    proc_pool.join()
    p1 = sum(stage1p)

    proc_pool = Pool(n_proc)
    stage2p = [] #proc_pool.map(compute_probability_subroutine, stage2_args)
    for result in tqdm.tqdm(proc_pool.imap_unordered(compute_probability_subroutine, stage2_args), total=len(stage2_args)):
        stage2p.append(result)
    proc_pool.close()
    proc_pool.join()
    p2 = sum(stage2p)

    print("First stage: ", p1)
    print("Second stage: ", p2)
    print("Approximation: ", p1 - p2)
    return p1 - p2

def compute_probability_bonferroni_3(model, weight_intervals, margin, verbose=True, n_proc=1):
    intersections_l, intersections_u = [], []
    dintersections_l, dintersections_u = [], []
    print("About to compute intersection for this many intervals: ", len(weight_intervals))
    for wi in trange(len(weight_intervals), desc="Computing intersection weights"):
        for wj in range(wi+1, len(weight_intervals)):
            i_l, i_u = intersect_intervals(weight_intervals[wi], weight_intervals[wj], margin, model.posterior_var)
            intersections_l.append(i_l)
            intersections_u.append(i_u)
            for wk in range(wj+1, len(weight_intervals)):
                ii_l, ii_u = intersection_bounds(i_l, i_u, weight_intervals[wk], margin, model.posterior_var)
                dintersections_l.append(ii_l)
                dintersections_u.append(ii_u)
    print("We found this many 2 dependances: ", len(intersections_l))
    print("We found this many 3 dependances: ", len(dintersections_l))

    # compute the probability of events
    overapprox = 0.0
    for weight_interval in weight_intervals:
        weight_interval = np.asarray([weight_interval])
        overapprox += compute_probability_subroutine_multiprocess(model, np.swapaxes(np.asarray(weight_interval),1,0), margin, verbose, n_proc)
    print("Overapproximation: ", overapprox)

    correction = 0.0
    for i in range(len(intersections_l)):
        correction += compute_probability_subroutine_multiprocess(model, [intersections_l[i], intersections_u[i]], 0.0, verbose, n_proc, correction=True)
    print("Correction: ", correction)

    adjustment = 0.0
    for i in range(len(dintersections_l)):
        adjustment += compute_probability_subroutine_multiprocess(model, [dintersections_l[i], dintersections_u[i]], 0.0, verbose, n_proc, correction=True)
    print("Adjustment: ", adjustment)
    print("Result: ", overapprox - correction + adjustment)

    return  overapprox - correction + adjustment

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
def prob_veri_intersection(model, s0, s1, w_marg, samples, predicate, i0=0, inflate=1.0):
    #w_marg = w_marg**2
    w_marg = w_marg*2
    safe_weights = []
    #safe_outputs = []
    for i in trange(samples, desc="Checking Samples"):
        model.model.set_weights(model.sample())
        #model.model.set_weights(model.posterior_mean)
        ol, ou = IBP_prob(model, s0, s1, model.model.get_weights(), w_marg)
        if(predicate(np.squeeze(s0), np.squeeze(s1), np.squeeze(ol), np.squeeze(ou))):
            safe_weights.append(model.model.get_weights())
            #ol = np.squeeze(ol); ou = np.squeeze(ou)
    print("Found %s safe intervals"%(len(safe_weights)))
    #p = compute_probability(model, safe_weights, w_marg)
    p = compute_probability_bonferroni(model, safe_weights, w_marg, max_depth=4)
    #p2 = compute_probability_bonferroni_3(model, safe_weights, w_marg)
    #assert(p == p2)
    return p #, np.squeeze(safe_outputs)


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


