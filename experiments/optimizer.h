#ifndef OPTIMIZER
#define OPTIMIZER

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// Base struct for optimizer parameters
struct optimParams { };

// Base class optimizer
class optimizer {
public: 
  virtual mat updateW(mat W, mat D, mat A_prev);
  virtual vec updateb(vec b, mat D);
  virtual void setParams(optimParams P);
};

// Class for creating optimizers which inherit from base class
class optimizerFactory
{
public:
  double lambda, m, L1, L2;
  mat W_templ;
  vec b_templ;
  optimizerFactory ();
  optimizerFactory (mat W_, vec b_, optimParams P);
  
  optimizer *createOptimizer (String type);
  
};


#endif