// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

#include "Layer.h"
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Layer class
// ---------------------------------------------------------------------------//
Layer::Layer () {}

Layer::Layer(int nodes_in_, int nodes_out_, List activ_param_, List optim_param_)
  : n_nodes(nodes_out_)
{
  
  // Initialize weight matrix and biasvector
  W = randn<mat>(nodes_out_, nodes_in_) / std::sqrt(nodes_in_);
  b = zeros<vec>(nodes_out_);
  
  // Set optimizer
  O = OptimizerFactory (W, b, optim_param_);
  
  // Set activation function
  g = ActivationFactory (activ_param_);
  
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

std::string Layer::print() {
  std::stringstream out;
  out << "  Layer - " << n_nodes << " nodes - " << g->type << " \n";
  return out.str();
}
