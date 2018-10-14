#ifndef OPTIMIZER
#define OPTIMIZER

#include <RcppArmadillo.h>

// Base class Optimizer
class Optimizer {
public:
  virtual arma::mat updateW(arma::mat W, arma::mat D, arma::mat A_prev);
  virtual arma::vec updateb(arma::vec b, arma::mat D);
};

// Class for creating optimizers which inherit from base class
class OptimizerFactory
{
public:
  std::string type;
  Rcpp::List optim_param;
  arma::mat W_templ;
  arma::vec b_templ;
  OptimizerFactory ();
  OptimizerFactory (arma::mat W_, arma::vec b_, Rcpp::List optim_param_);
  Optimizer *createOptimizer ();
};


#endif