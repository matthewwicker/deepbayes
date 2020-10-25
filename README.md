# deepbayes

deepbayes is a package that aims to make training Bayesian neural networks (BNNs) simple.

Implemented on top of TensorFlow 2.0 and working with tf.keras as a primary interface, one can simply declare a keras (or TF) model and pass it to 
the desired deepbayes optimizer and it will compile the model into an object that is ready to perform approximate inference.

deepbayes is actively under development, and location and names of packages may change (such changes will be noted at the bottom of this repo).  jax support coming soon :) 

#### Install:

`pip install deepbayes`

#### Minimal Example using deepbayes

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


#### Dependancies:

tensorflow, tensorflow-probability, numpy, tqdm, statsmodels,

### Future Support

- [ ] HTML documentation 
- [ ] JAX autodif library instead of TF (deepbayes.jax.optimizers)
- [ ] Sequential Monte Carlo optimizer
- [ ] Monte Carlo Dropout optimizer
- [ ] Stochastic Gradient Langevin Dynamics optimizer
- [ ] Binary Bayesian neural networks
