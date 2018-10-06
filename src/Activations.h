#ifndef ACTIVATIONS
#define ACTIVATIONS

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// Base class Activation
class Activation {
public: 
  virtual mat eval(mat X);
  virtual mat grad(mat X);
};

// Class for creating Activation classes that inherit from base class
class ActivationFactory
{
public:
  std::string type;
  List activ_param;
  ActivationFactory ();
  ActivationFactory (List activ_param_);
  Activation *createActivation ();
};

#endif