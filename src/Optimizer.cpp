// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "Optimizer.h"
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Stochastic Gradient Descent optimizer
// ---------------------------------------------------------------------------//

// Default constructor needed for serialization
SGD::SGD () { type = "SGD"; } 

SGD::SGD (mat W_templ_, vec b_templ_, List optim_param_) 
  : learn_rate( optim_param_["learn_rate"]), m(optim_param_["m"]),
    L1(optim_param_["L1"]), L2(optim_param_["L2"]) {
  
  // Set optimizer type  
  type = "SGD";
  
  // Initialize momentum matrices
  mW = zeros<mat>(size(W_templ_));
  mb = zeros<vec>(size(b_templ_));
}

mat SGD::updateW(mat W, mat D, mat A_prev) {
  batch_size = A_prev.n_cols;
  mat gW = A_prev * D / batch_size;
  mW = m * mW - learn_rate * gW.t();
  return (1 - learn_rate * L2) * W - learn_rate * L1 * sign(W) + mW;
}

vec SGD::updateb(vec b, mat D) {
  mb = m * mb - learn_rate * sum(D, 0).t() / batch_size;
  return b + mb;
}

// ---------------------------------------------------------------------------//
// Optimizer factory 
// ---------------------------------------------------------------------------//

std::shared_ptr<Optimizer> OptimizerFactory (mat W_templ, mat b_templ, List optim_param) {
  std::string type = as<std::string>(optim_param["type"]);
  if    (type == "sgd") return std::shared_ptr<Optimizer>(new SGD(W_templ, b_templ, optim_param));
  else                  return NULL;
}


