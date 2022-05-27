# deepbayes

deepbayes is a package that aims to make training Bayesian neural networks (BNNs) simple.

Implemented on top of TensorFlow 2.0 and working with tf.keras as a primary interface, one can simply declare a keras (or TF) model and pass it to 
the desired deepbayes optimizer and it will compile the model into an object that is ready to perform approximate inference.

We also support a variety of adversarial robustness analysis methods for Bayesian Neural Networks.

deepbayes is actively under development, and location and names of packages may change. 

Current version 0.1.0 (Updated 27 May 2022) 

#### Install:

`pip3 install deepbayes`


## Supported Analysis

In addition to producing posterior distributions for BNNs, deepbayes also supports some basic analysis including:

- Adversarial attacks for BNNs (FGSM, PGD, CW)
- Statistical Guarantees for the Robustness of Bayesian Neural Networks        - https://arxiv.org/pdf/1903.01980.pdf
- Probabilistic Safety for Bayesian Neural Networks                            - https://arxiv.org/pdf/2004.10281.pdf
- Adversarial Robustness of Bayesian Neural Networks                           - https://bit.ly/3lLqENo
- Bayesian Inference with Certifiable Adversarial Robustness                   - https://arxiv.org/abs/2102.05289

#### For each of the above papers, a tutorial has been included in this repository that reproduces the key result of each paper

### This package was used to produce the following papers: 

- Uncertainty Quantification with Statistical Guarantees in End-to-End Autonomous Driving Control   - https://arxiv.org/pdf/1909.09884.pdf
- Robustness of Bayesian Neural Networks to Gradient-Based Attacks                                  - https://arxiv.org/abs/2002.04359
- Gradient-Free Adversarial Attacks for Bayesian Neural Networks                                    - https://arxiv.org/pdf/2012.12640.pdf

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
  Implemented based on description in https://arxiv.org/pdf/1902.02476.pdf
  #### Stochastic Gradient Descent (SGD)
  [Original paper by Robbins and Monro](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full)
  #### Adam Optimizer (ADAM)
  Implimentation based on description in https://arxiv.org/abs/1412.6980

#### Dependancies:

tensorflow, tensorflow-probability, numpy, tqdm, statsmodels,

