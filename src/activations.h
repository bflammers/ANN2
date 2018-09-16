#ifndef ACTIVATIONS
#define ACTIVATIONS

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// Base class activation
class activation {
public: 
  virtual mat eval(mat X);
  virtual mat grad(mat X);
};

// Class for creating activation classes that inherit from base class
class activationFactory
{
public:
  std::string type;
  List activ_param;
  activationFactory ();
  activationFactory (List activ_param_);
  activation *createActivation ();
};

#endif