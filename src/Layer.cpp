// Enable C++11 via this plugin 
// [[Rcpp::plugins("cpp11")]]

// [[Rcpp::depends(Rcereal)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <cereal/types/vector.hpp>
#include <cereal/archives/portable_binary.hpp>
#include "Layer.h"
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Layer class
// ---------------------------------------------------------------------------//
Layer::Layer () {}

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

// Serialize
template<class Archive>
void Layer::save(Archive & archive) const
{
  MatSerializer serW(W);
  archive( serW, activ_type, n_nodes ); 
}

// Deserialze
template<class Archive>
void Layer::load(Archive & archive)
{
  MatSerializer serW;
  archive( serW, activ_type, n_nodes );
  W = serW.getMat();
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
