#ifndef OPTIMIZER
#define OPTIMIZER

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// Base struct for optimizer parameters
struct optimParams { };
struct sgdParams : optimParams {
  double lambda, m, L1, L2;
  sgdParams ();
  sgdParams (double lambda_, double m_, double L1_, double L2_);
};
struct rmspropParams : optimParams {
  double lambda, m;
  rmspropParams ();
  rmspropParams (double lambda_, double m_);
};

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
  mat W_templ;
  vec b_templ;
  optimizerFactory ();
  optimizerFactory (mat W_, vec b_);
  optimizer *createOptimizer (String type);
};


#endif