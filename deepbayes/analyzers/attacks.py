# Author: Matthew Wicker
"""
Importantly we assume that we are EITHER passed an 
optimizer object OR a posteriormodel object. All method
must be tested against both interfaces in order for 
training and a posteriori analysis to work.
"""
import math
import copy
import numpy as np
from tqdm import tqdm
from tqdm import trange
import tensorflow as tf


# Zeroth order gradients
def zeroth_order_gradient(model, inp, direction, loss_fn, num_models=25, step=0.35):
    number_images = inp.shape[0]
    feature_shape = inp.shape[1:]
    feature_size = 1
    for i in inp.shape[1:]:
        feature_size *= i
    mask = tf.eye(feature_size)
    step_mask = mask * step
    resize_shape = [-1]
    for i in feature_shape:
        resize_shape.append(i)
    resize_shape = tuple(resize_shape)
    step_mask = tf.reshape(step_mask, resize_shape)
    grads = []
    #print(step_mask[0])
    #print(step_mask[-1])
    for i in trange(number_images):
        plus_inps = inp[i] + step_mask
        subt_inps =  inp[i] - step_mask
        #print(plus_inps[i] - subt_inps[i])
        preds_plus = model.predict(plus_inps, n=num_models)
        losses_p = np.asarray([loss_fn(direction[i], j) for j in preds_plus])
        losses_p = losses_p.reshape(feature_shape)
        preds_subs = model.predict(subt_inps, n=num_models)
        losses_m = np.asarray([loss_fn(direction[i], j) for j in preds_subs])
        losses_m = losses_m.reshape(feature_shape)
        grad = (losses_p - losses_m)/(2.0*step)
        grads.append(grad)
    return np.asarray(grads)


# First order gradients
def gradient_expectation(model, inp, direction, loss_fn, num_models=10):
    gradient_sum = tf.zeros(inp.shape)
    inp = tf.convert_to_tensor(inp)
    val = num_models
    if(num_models < 1):
        num_models = 1
    for i in range(num_models):
        if(model.det or val == -1):
            no_op = 0
        else:
            model.model.set_weights(model.sample())
        # Establish Gradient Tape Context (for input this time)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inp)
            # Get the probabilities
            predictions = model.predict(inp)
            loss = loss_fn(direction, predictions)
        # Get the gradients
        inp_gradient = tape.gradient(loss, inp)
        #print("GRAD: ", inp_gradient)
        try:
            gradient_sum += inp_gradient
        except:
            gradient_sum += tf.cast(inp_gradient, 'float32')
        if(model.det or val == -1):
            break
    return gradient_sum

# Approx Second order gradients 
# - To be implimented when it becomes important for l2 certification
#   Default is set to 1 model for training ease
# FGSM
def FGSM(model, inp, loss_fn, eps, direction=-1, num_models=1, order=1):
    inp = np.asarray(inp)
    maxi = inp + eps; mini = inp - eps
    direc = direction # !*! come back and fix
    if(type(direc) == int):
        direc = np.squeeze(model.predict(inp))
        try:
            direc = np.argmax(direc, axis=1)
        except:
            direc = np.argmax(direc)
    if(order == 1):
        grad = gradient_expectation(model, inp, direc, loss_fn, num_models)
    elif(order == 0):
        grad = zeroth_order_gradient(model, inp, direc, loss_fn, num_models)
    
    grad = np.sign(grad)
    #print("GRAD: ", grad[0])
    adv = inp + eps*np.asarray(grad)
    adv = np.clip(adv, mini, maxi)
    adv = np.clip(adv, 0, 1)
    return adv


def axis_mult(a, b, axis=1):
    try:
        a = a.numpy()
    except:
        pass
    try:
        b = b.numpy()
    except:
        pass
    # Given axis along which elementwise multiplication with broadcasting 
    # is to be performed
    given_axis = axis

    # Create an array which would be used to reshape 1D array, b to have 
    # singleton dimensions except for the given axis where we would put -1 
    # signifying to use the entire length of elements along that axis  
    dim_array = np.ones((1,a.ndim),int).ravel()
    dim_array[given_axis] = -1

    # Reshape b with dim_array and perform elementwise multiplication with 
    # broadcasting along the singleton dimensions for the final output
    b_reshaped = b.reshape(dim_array)
    mult_out = a*b_reshaped
    return mult_out

from tqdm import trange

# PGD
def _PGD(model, inp, loss_fn, eps, direc=-1, step=0.1, num_steps=15, num_models=35, order=1):
    input_shape = np.squeeze(inp).shape
    output = model.predict(inp)
    inp_copy = copy.deepcopy(inp)
    
    if(type(direc) == int):
        direc = np.squeeze(model.predict(inp))
        try:
            direc = np.argmax(direc, axis=1)
        except:
            direc = np.argmax(direc)

    adv = np.asarray(inp)
    maxi = adv + eps; mini = adv - eps
    adv = adv + ((eps/10) * np.sign(np.random.normal(0.0, 1.0, size=adv.shape)))
    adv = np.clip(adv, 0.0, 1.0)
    #print("PERFORMING PGD")
    for j in range(num_steps+1):
        if(order == 1):
            grad = gradient_expectation(model, adv, direc, loss_fn, num_models)
        #elif(order == 1):
        #    grad = zeroth_order_gradient(model, adv, direc, loss_fn, num_models)
        #grad = grad/np.max(grad, axis=1) #(grad-np.min(grad))/(np.max(grad)-np.min(grad))
        grad = np.sign(grad)
        # Normalize as below if you want to do l2 optimization
        #norm = np.max(grad, axis=1)
        #grad = np.asarray([grad[i]/norm[i] for i in range(len(norm))])
        grad *= (eps/float(num_steps)) # an empirically good learning rate
        adv = adv + grad
        adv = tf.cast(adv, 'float32')
    adv = np.clip(adv, mini, maxi)
    adv = np.clip(adv, 0, 1)
    return adv

def PGD(model, inp, loss_fn, eps, direction=-1, step=0.1, num_steps=5, num_models=-1, order=1, restarts=0):
    advs = []
    for i in range(restarts+1):
        adv = _PGD(model, inp, loss_fn, eps, direc=direction, 
                  step=step, num_steps=num_steps, num_models=num_models, order=order)
        advs.append(adv)
    if(restarts == 0):
        return adv
    else:
        return advs

# CW
def CW(model, inp, loss_fn, eps=0.01, num_steps=5, num_models=25, direction=-1,
             stepsize=None, decay1=0.99, decay2=0.9999, epsilon=1e-04, order=1):
    direc = direction # !*! come back and fix
    if(type(direc) == int):
        direc = np.squeeze(model.predict(inp))
        try:
            direc = np.argmax(direc, axis=1)
        except:
            direc = np.argmax(direc)
    moment1, moment2, time = 0, 0, 0
    stepsize = eps/float(num_steps)
    input_shape = np.squeeze(inp).shape
    inp_copy = copy.deepcopy(inp)
    adv = np.asarray(inp)
    maxi = adv + eps; mini = adv - eps
    if(type(direc) == int):
        direc = np.argmax(model.predict(inp))
    for i in trange(num_steps, desc="CW iterations"):
        if(order == 1):
            grad = gradient_expectation(model, adv, direc, loss_fn, num_models)
        elif(order == 1):
            grad = zeroth_order_gradient(model, adv, direc, loss_fn, num_models)
        #grad = gradient_expectation(model, inp, direc, loss_fn, num_models)
        # Now we have a gradient approximation, lets finish the steps for an iteratio
        # of adam
        time = time + 1
        # Compute moment vectors
        moment1 = (moment1 * decay1) + ((1-decay1) * grad)
        moment2 = (moment2 * decay2) + ((1-decay2) * (grad**2))
        # Bias correct the moment vectors
        moment1 = moment1 / (1-(decay1**time))
        moment2 = moment2 / (1-(decay2**time))
        # Update adversarial example
        d = (moment1/(np.sqrt(moment2) + epsilon))
        d = np.sign(d)
        #norm = np.max(d, axis=1) #(d-np.min(d))/(np.max(d)-np.min(d))
        #d = axis_mult(d, norm, axis=0)
        #d = np.asarray([d[i]/norm[i] for i in range(len(norm))])
        #print("grad in cw", d[0])
        #print(d.shape)
        d *= (eps/float(num_steps))
        adv = adv + d
        #adv = np.clip(adv, 0, 1)
        adv = np.clip(adv, 0, 1)
    adv = np.clip(adv, mini, maxi)
    adv = np.clip(adv, 0, 1)
    return adv

# Substitute Model Attack


# L2 Bounded Certification (Hessian Derivation)
# -- To be implimented, nothing in the paper works for CNNs yet. 
