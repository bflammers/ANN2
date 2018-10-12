// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "Layer.h"
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Layer class
// ---------------------------------------------------------------------------//

Layer::Layer(int nodes_in_, int nodes_out_, List activ_param_, List optim_param_)
  : n_nodes(nodes_out_), 
    activ_type(as<std::string>(activ_param_["type"]))
{
  
  // Initialize weight matrix and biasvector
  W = randn<mat>(nodes_out_, nodes_in_) / sqrt(nodes_in_);
  b = zeros<vec>(nodes_out_);
  
  // Set optimizer
  OptimizerFactory oFact(W, b, optim_param_);
  O = oFact.createOptimizer();
  
  // Set activation function
  ActivationFactory aFact(activ_param_); 
  g = aFact.createActivation();
  
}
  
mat Layer::forward (mat X) 
{
  A_prev = X; 
  Z = W * X;
  Z.each_col() += b;
  return g->eval(Z);
}

mat Layer::backward (mat E) 
{
  mat D = E % g->grad(Z).t();
  W = O->updateW(W, D, A_prev);
  b = O->updateb(b, D);
  return D * W;
}
