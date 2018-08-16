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
  double lambda, m, L1, L2;
  int batch_size;
  mat mW;
  vec mb;
public:
  SGD (mat W_templ_, vec b_templ_, List optim_param_) {
    // Initialize momentum matrices
    mW = zeros<mat>(size(W_templ_));
    mb = zeros<vec>(size(b_templ_));
    
    // Set optimization parameters
    lambda = optim_param_["lambda"];
    m = optim_param_["m"];
    L1 = optim_param_["L1"];
    L2 = optim_param_["L2"];
  }
  
  mat updateW(mat W, mat D, mat A_prev) {
    batch_size = A_prev.n_cols;
    mat gW = A_prev * D / batch_size;
    mW = m * mW - lambda * gW.t();
    return (1 - lambda - L2) * W - lambda * L1 * sign(W) + mW;
  }
  
  vec updateb(vec b, mat D) {
    mb = m * mb - lambda * sum(D, 0).t() / batch_size;
    return b + mb;
  }
};

// RMSprop 
class RMSprop : public optimizer
{
private:
  double lambda, m;
  mat mW;
  vec mb;
public:
  RMSprop (mat W_templ_, vec b_templ_, List optim_param_) {
    // Initialize momentum matrices
    mW = zeros<mat>(size(W_templ_));
    mb = zeros<vec>(size(b_templ_));
    
    // Set optimization params
    lambda = optim_param_["lambda"];
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
  type = optim_param["type"];
}

// Method for creating optimizers
optimizer *optimizerFactory::createOptimizer () {
  if      (type == 0) return new SGD(W_templ, b_templ, optim_param);
  else if (type == 1) return new RMSprop(W_templ, b_templ, optim_param);
  else                return NULL;
}


