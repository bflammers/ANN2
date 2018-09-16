// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "layer.h"
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Layer class
// ---------------------------------------------------------------------------//

layer::layer(int nodes_in_, int nodes_out_, List activ_param_, List optim_param_)
{
  // Initialize weight matrix and biasvector
  W = randn<mat>(nodes_out_, nodes_in_) / sqrt(nodes_in_);
  b = zeros<vec>(nodes_out_);
  
  // Set optimizer
  optimizerFactory oFact(W, b, optim_param_);
  O = oFact.createOptimizer();
  
  // Set activation function
  activationFactory aFact(activ_param_); 
  g = aFact.createActivation();
  
  Rcout << "\n Layer - "<< nodes_out_ << " nodes - " << 
    as<std::string>(activ_param_["type"]);
}
  
mat layer::forward (mat X) 
{
  A_prev = X; 
  Z = W * X + repColVec(b, X.n_cols);
  return g->eval(Z);
}

mat layer::backward (mat E) 
{
  mat D = E % g->grad(Z).t();
  W = O->updateW(W, D, A_prev);
  b = O->updateb(b, D);
  return D * W;
}
