#ifndef ACTIVATIONS
#define ACTIVATIONS

#include <RcppArmadillo.h>

// Base class Activation
class Activation {
public: 
  virtual arma::mat eval(arma::mat X) = 0;
  virtual arma::mat grad(arma::mat X) = 0;
};

// Class for creating Activation classes that inherit from base class
class ActivationFactory
{
public:
  std::string type;
  Rcpp::List activ_param;
  ActivationFactory ();
  ActivationFactory (Rcpp::List activ_param_);
  Activation *createActivation ();
};

#endif