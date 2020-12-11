# Author: Matthew Wicker
# We consider the worst case version of BPDA where
# we take a DNN of the exact BNN architecture and
# train a dnn to match the BNN prediction.

from deepbayes import optimizers
from . import attacks

def BPDA(model, inp, direction, loss_fn, X_train, y_train, epsilon=0.1,
         num_models=10, epochs=15, lr=0.05, save_inter_path="temp_model"):
    dnn = optimizers.sgd(model.model, loss_fn, epochs=epochs, learning_rate=lr)
    dnn.train(X_train, y_train)
    dnn.save("bpda_%s"%(save_inter_path))
    
    inp = np.asarray(inp)
    maxi = inp + eps; mini = inp - eps
    direc = direction # !*! come back and fix
    if(type(direc) == int):
        direc = np.squeeze(model.predict(inp))
        try:
            direc = np.argmax(direc, axis=1)
        except:
            direc = np.argmax(direc)
    grad = gradient_expectation(dnn, inp, direc, loss_fn, num_models)
    grad = np.sign(grad)
    #print("GRAD: ", grad[0])
    adv = inp + eps*np.asarray(grad)
    adv = np.clip(adv, mini, maxi)
    adv = np.clip(adv, 0, 1)
    return adv

