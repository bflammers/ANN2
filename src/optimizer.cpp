// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "optimizer.h"
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Methods of base class optimizer
// ---------------------------------------------------------------------------//

mat optimizer::updateW(mat W, mat D, mat A_prev) { return W.zeros(); }
vec optimizer::updateb(vec b, mat D) { return b.zeros(); }

// ---------------------------------------------------------------------------//
// Optimizers
// ---------------------------------------------------------------------------//

// SGD 
class SGD : public optimizer
{
private:
  double learn_rate, m, L1, L2;
  int batch_size;
  mat mW;
  vec mb;
public:
  SGD (mat W_templ_, vec b_templ_, List optim_param_) 
    : learn_rate ( optim_param_["learn_rate"]),
      m ( optim_param_["m"]),
      L1 ( optim_param_["L1"]),
      L2 ( optim_param_["L2"])
  {
    // Initialize momentum matrices
    mW = zeros<mat>(size(W_templ_));
    mb = zeros<vec>(size(b_templ_));
  }
  
  mat updateW(mat W, mat D, mat A_prev) {
    batch_size = A_prev.n_cols;
    mat gW = A_prev * D / batch_size;
    mW = m * mW - learn_rate * gW.t();
    return (1 - learn_rate * L2) * W - learn_rate * L1 * sign(W) + mW;
  }
  
  vec updateb(vec b, mat D) {
    mb = m * mb - learn_rate * sum(D, 0).t() / batch_size;
    return b + mb;
  }
};

// RMSprop 
class RMSprop : public optimizer
{
private:
  double learn_rate, m;
  mat mW;
  vec mb;
public:
  RMSprop (mat W_templ_, vec b_templ_, List optim_param_) {
    // Initialize momentum matrices
    mW = zeros<mat>(size(W_templ_));
    mb = zeros<vec>(size(b_templ_));
    
    // Set optimization params
    learn_rate = optim_param_["learn_rate"];
    m = optim_param_["m"];
  }
  
  mat updateW(mat W, mat D, mat A_prev) {
    return W.zeros();
  }
  
  vec updateb(vec b, mat D) {
    return b.zeros();
  }
  
};

// ---------------------------------------------------------------------------//
// Methods for class optimizer factory 
// ---------------------------------------------------------------------------//

// Constructor
optimizerFactory::optimizerFactory (mat W_, vec b_, List optim_param_) :
  optim_param(optim_param_) {
  // Store templates of W and b for initialization purposes
  W_templ = zeros<mat>(size(W_));
  b_templ = zeros<vec>(size(b_));
  
  // Set optimization type
  type = as<std::string>(optim_param_["type"]);
}

// Method for creating optimizers
optimizer *optimizerFactory::createOptimizer () {
  if      (type == "sgd")     return new SGD(W_templ, b_templ, optim_param);
  else if (type == "rmsprop") return new RMSprop(W_templ, b_templ, optim_param);
  else                return NULL;
}


