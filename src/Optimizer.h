#ifndef OPTIMIZER
#define OPTIMIZER

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// Base class Optimizer
class Optimizer {
public:
  virtual mat updateW(mat W, mat D, mat A_prev);
  virtual vec updateb(vec b, mat D);
};

// Class for creating optimizers which inherit from base class
class OptimizerFactory
{
public:
  std::string type;
  List optim_param;
  mat W_templ;
  vec b_templ;
  OptimizerFactory ();
  OptimizerFactory (mat W_, vec b_, List optim_param_);
  Optimizer *createOptimizer ();
};


#endif