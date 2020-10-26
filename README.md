# deepbayes

deepbayes is a package that aims to make training Bayesian neural networks (BNNs) simple.

Implemented on top of TensorFlow 2.0 and working with tf.keras as a primary interface, one can simply declare a keras (or TF) model and pass it to 
the desired deepbayes optimizer and it will compile the model into an object that is ready to perform approximate inference.

deepbayes is actively under development, and location and names of packages may change (such changes will be noted at the bottom of this repo).  jax support coming soon :) 

#### Install:

`pip install deepbayes`

## Supported Optimizers 

  #### BayesByBackprop
  Implemented based on description in https://arxiv.org/abs/1505.05424
  #### Hamiltonian (Hybrid) Monte Carlo
  Implemented based on description in https://arxiv.org/abs/1206.1901
  #### NoisyAdam (aka Vadam)
  Implemented based on description in https://arxiv.org/abs/1712.02390
  #### Variational Online Gauss Newton (aka VOGN)
  Implemented based on description in https://arxiv.org/abs/1806.04854 and https://arxiv.org/abs/2002.10060
  #### Stochastic Weight Averaging - Gaussian (aka SWAG)
  Implemented bsed on description in https://arxiv.org/pdf/1902.02476.pdf
  #### Stochastic Gradient Descent
  Not sure what to reference you to here... a google will suffice

## Supported Analysis

In addition to producing posterior distributions for BNNs, deepbayes also supports some basic analysis including

- Adversarial attacks for BNNs (FGSM, PGD, CW)
- Uncertainty quantification 
- Statistical guarentees of robustness - https://arxiv.org/pdf/1903.01980.pdf
- Probabalistic safety estimates  - https://arxiv.org/pdf/2004.10281.pdf

#### Dependancies:

tensorflow, tensorflow-probability, numpy, tqdm, statsmodels,

### Future Support

Below we have a tentative to-do list of inference methods to impliment (7 to be exact) and other properties we want to tool to have. 

- [ ] HTML documentation 
- [ ] Non-Gaussian Posterior (requires extension of posterior representation as well)
- [ ] Stochastic Variational Inference (SVI)
- [ ] Probabilistic Backprop (PBP)
- [ ] Sequential Monte Carlo optimizer (SMC)
- [ ] Monte Carlo Dropout optimizer (MCD)
- [ ] Stochastic Gradient Langevin Dynamics optimizer (SGLD)
- [ ] Stochastic Gradient Markov Chain Monte Carlo (SGMCMC)
- [ ] Riemann manifold HMC (RMHMC)
- [ ] JAX autodif library instead of TF (deepbayes.jax.optimizers)
- [ ] Binary Bayesian neural networks
