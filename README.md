# ContBayes: Online training of Bayesian Neural Networks

This repository contains the code for the paper "Rapid Online Bayesian Learning for Deep Receivers" 
submitted for the ICASSP 2025 conference.

## Description

We aim to enhance the adaptability, efficiency, and robustness of deep receivers in MIMO uplink systems.

## Key Features

* A Bayesian Neural Network class `BayesNN`, based on TyXe `VariationalBNN`, allows interaction with the weights of the
    model and Jacobian computations.
* `BayesianDeepSIC` - A Bayesian version of the DeepSIC model defined in [1].
* Tracker classes `EKF` (for general `BayesNN` models) and `DeepsicEKF` (for `BayesianDeepSIC` model) for online
    training using the extended Kalman Filter.

## References

[1] N. Shlezinger, R. Fu and Y. C. Eldar, "DeepSIC: Deep Soft Interference Cancellation for Multiuser MIMO Detection," 
in IEEE Transactions on Wireless Communications, vol. 20, no. 2, pp. 1349-1362, Feb. 2021.
