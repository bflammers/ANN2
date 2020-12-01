[![Licence](https://img.shields.io/badge/licence-GPL--3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html) 
[![CRAN\_Status\_Badge](http://www.r-pkg.org/badges/version/ANN2)](https://cran.r-project.org/package=ANN2) 
![Monthly downloads](https://cranlogs.r-pkg.org/badges/ANN2)
![R CMD check](https://github.com/bflammers/ANN2/workflows/R-CMD-check/badge.svg)
[![codecov](https://codecov.io/gh/bflammers/ANN2/branch/dev/graph/badge.svg)](https://codecov.io/gh/bflammers/ANN2)

# ANN2
Artificial Neural Networks package for R 

Training of neural networks for classification and regression tasks using mini-batch gradient descent. Special features include a function for training autoencoders, which can be used to detect anomalies, and some related plotting functions. Multiple activation functions are supported, including tanh, relu, step and ramp. For the use of the step and ramp activation functions in detecting anomalies using autoencoders, see Hawkins et al. (2002). Furthermore, several loss functions are supporterd, including robust ones such as Huber and pseudo-Huber loss, as well as L1 and L2 regularization. The possible options for optimization algorithms are RMSprop, Adam and SGD with momentum. The package contains a vectorized C++ implementation that facilitates fast training through mini-batch learning.

***
  