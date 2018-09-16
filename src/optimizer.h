#ifndef OPTIMIZER
#define OPTIMIZER

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// Base class optimizer
class optimizer {
public:
  virtual mat updateW(mat W, mat D, mat A_prev);
  virtual vec updateb(vec b, mat D);
};

// Class for creating optimizers which inherit from base class
class optimizerFactory
{
public:
  std::string type;
  List optim_param;
  mat W_templ;
  vec b_templ;
  optimizerFactory ();
  optimizerFactory (mat W_, vec b_, List optim_param_);
  optimizer *createOptimizer ();
};


#endif