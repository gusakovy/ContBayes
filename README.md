# ContBayes: Online training of Bayesian Neural Networks using the Extended Kalman Filter

### Editor: Yakov Gusakov
### Supervisors: Prof. Tirza Routtenberg and Dr. Nir Shlezinger


## Description

This project explores and implements the use of the Extended Kalman Filter (EKF) for online training of 
Bayesian Neural Networks (BNNs).

The idea of training Bayesian neural networks using the extended Kalman filter was first introduced in [1]. 
In our work we primarily focus on online training of deep receivers, where we first pre-train a Bayesian network using
SVI, and then proceed to track its weights using the extended Kalman filter with multiple observations (pilot messages)
as the channel changes with time.

## Project Goals

We aim to enhance the adaptability, efficiency, and robustness in MIMO uplink systems.

## Key Features

* A Bayesian Neural Network class `BayesNN`, based on TyXe `VariationalBNN`, allows interaction with the weights of the
    model and Jacobian computations.
* `BayesianDeepSIC` - A Bayesian version of the DeepSIC model defined in [2], trained via SVI.
* Tracker classes `EKF` (for general `BayesNN` models) and `DeepsicEKF` (for `BayesianDeepSIC` model) for online
    training using the extended Kalman Filter.
* Tracker classes `SqrtEKF` (for general `BayesNN` models) and `DeepsicSqrtEKF` (for `BayesianDeepSIC` model) for 
    online training using the square root extended Kalman Filter.

## References

[1] Peter G. Chang, Kevin Patrick Murphy, and Matt Jones. On diagonal approximations to the extended Kalman filter 
for online training of Bayesian neural networks. In Continual Lifelong Learning Workshop at ACML 2022, 2022.

[2] N. Shlezinger, R. Fu and Y. C. Eldar, "DeepSIC: Deep Soft Interference Cancellation for Multiuser MIMO Detection," 
in IEEE Transactions on Wireless Communications, vol. 20, no. 2, pp. 1349-1362, Feb. 2021.
https://ieeexplore.ieee.org/document/9242305

