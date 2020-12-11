# Matthew Wicker
# TODO: Survey literature for algorithms which 
# properly capture uncertainty of the posterior
# predictive distribution
import numpy as np

"""
Paper:
Uncertainty quantification using Bayesian neural networks in classification: 
Application to ischemic stroke lesion segmentation
https://www.sciencedirect.com/science/article/pii/S016794731930163X?via%3Dihub
This is also explained very nicely here:
https://towardsdatascience.com/what-uncertainties-tell-you-in-bayesian-neural-networks-6fbd5f85648e
"""
def variational_uncertainty(model, input, num_samples=35):
    if(model.det == True):
        num_samples=1
    y_preds_per_samp = []
    for i in range(num_samples):
        y_pred = model.predict(input, n=1)
        y_preds_per_samp.append(y_pred)
    y_preds_per_samp = np.asarray(y_preds_per_samp)
    """
    for i in range(len(input)):
        for j in range(num_samples):
            vec = y_pred_mean[i] - y_preds_per_samp[j][i]
            epistemic_unc += sum(vec**2)
            #aleatoric_unc += sum(y_pred_mean[i] *  y_preds_per_samp[j][i])
    epistemic_unc /= float(len(input))
    epistemic = epistemic_unc/float(num_samples)
    """
    print("Pred shape: ", y_preds_per_samp.shape)
    p_hat = y_preds_per_samp
    epistemic = np.mean(p_hat**2, axis=0) - np.mean(p_hat, axis=0)**2
    aleatoric = np.mean(p_hat*(1-p_hat), axis=0)
    return epistemic, aleatoric


"""
Returns the predictive entropy of the mean of the posterior predictive (per input)
"""
def predictive_entropy(model, input, num_models=35):
    if(model.det == True):
        num_samples=1
    y_pred = model.predict(input, n=num_models)
    y_pred += 0.001
    entropy = []
    for i in y_pred:
        entropy.append(-1 * np.sum(i*np.log(i)))
    entropy = np.nan_to_num(entropy)
    return list(entropy)
"""
Returna the likelihood ratio of the two inputs meant to
represent the calibration of the uncerainty of one wrt 
the other
"""
def likelihood_ratio(model, input_indist, input_outdist, labels=-1, num_models=35):
    indist_pred = model.predict(input_indist, n=num_models)
    outdist_pred = model.predict(input_outdist, n=num_models)
    in_like, out_like = [], []
    for i in range(len(indist_pred)):
#        if(type(labels) != int):
#            if(np.argmax(outdist_pred) != labels[i]):
#                pass
#            else:
#                continue
        in_like.append(np.max(indist_pred[i]))
        out_like.append(np.max(outdist_pred[i]))
    in_like = np.asarray(in_like)
    out_like = np.asarray(out_like)
    return np.mean(out_like)/np.mean(in_like) #, np.mean(out_like), np.mean(in_like)

"""
This is not really an uncertainty metric, more of a quality
metric, however it certainly does not belong in the attacks
file so I put it here.
"""
def auroc(model, input, labels, num_samples=35):
    from sklearn.metrics import roc_auc_score
    y_pred = model.predict(input, n=num_samples)
    roc_val = roc_auc_score(labels, y_pred, average='macro', multi_class="ovr")
    return roc_val


