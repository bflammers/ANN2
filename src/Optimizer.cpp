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
  : learn_rate( optim_param_["learn_rate"]), momentum(optim_param_["m"]),
    L1(optim_param_["L1"]), L2(optim_param_["L2"]) {
  
  // Set optimizer type  
  type = "SGD";
  
  // Initialize momentum matrices
  mW = zeros<mat>(size(W_templ_));
  mb = zeros<vec>(size(b_templ_));
}

mat SGD::updateW(mat W, mat dW) {
  
  // Calculate momentum term
  mW = momentum * mW + learn_rate * dW.t();
  
  // Update weights with momentum term using L1 and L2 regularization
  // note that sign(0) = 0, which is what we want
  // values L1 and L2 should be scaled by n_train according to Nielsen - bit strange if you ask me
  return (1 - learn_rate * L2) * W - learn_rate * L1 * sign(W) - mW;
}

vec SGD::updateb(vec b, vec db) {
  
  // Calculate momentum term
  mb = momentum * mb + learn_rate * db;
  
  // Return updated biases, no regularization for these
  return b - mb;
}

// ---------------------------------------------------------------------------//
// RMSprop optimizer
// ---------------------------------------------------------------------------//

// Default constructor needed for serialization
RMSprop::RMSprop () { type = "RMSprop"; } 

RMSprop::RMSprop (mat W_templ_, vec b_templ_, List optim_param_) 
  : learn_rate( optim_param_["learn_rate"]), 
    decay(0.9),
    epsilon(0.0001),
    L2(optim_param_["L2"]) {
  
  // Set optimizer type  
  type = "RMSprop";
  
  // Initialize momentum matrices
  rmsW = zeros<mat>(size(W_templ_));
  rmsb = zeros<vec>(size(b_templ_));
}

mat RMSprop::updateW(mat W, mat dW) {
  rmsW = decay * rmsW + (1 - decay) * square(dW.t());
  return (1 - learn_rate * L2) * W - learn_rate / sqrt(rmsW + epsilon) % dW.t();
}

vec RMSprop::updateb(vec b, vec db) {
  rmsb = decay * rmsb + (1 - decay) * square(db);
  return b - learn_rate / sqrt(rmsb + epsilon) % db;
}

// ---------------------------------------------------------------------------//
// Adam optimizer
// ---------------------------------------------------------------------------//

// Default constructor needed for serialization
Adam::Adam () { type = "Adam"; } 

Adam::Adam (mat W_templ_, vec b_templ_, List optim_param_) 
  : learn_rate( optim_param_["learn_rate"]), 
    beta1(0.9),
    beta2(0.999),
    epsilon(1e-8),
    L2(optim_param_["L2"]) {
  
  // Set optimizer type  
  type = "Adam";
  
  // Initialize matrices
  mW = zeros<mat>(size(W_templ_));
  vW = zeros<mat>(size(W_templ_));
  
  // Initialize vectors
  mb = zeros<vec>(size(b_templ_));
  vb = zeros<vec>(size(b_templ_));
}

mat Adam::updateW(mat W, mat dW) {
  
  // Calculate mW and bias corrected mW
  mW = beta1 * mW + (1 - beta1) * dW.t();
  mat bc_mW = mW / (1-std::pow(beta1, 10));
  
  // Calculate mW and bias corrected mW
  vW = beta2 * vW + (1 - beta2) * square(dW.t());
  mat bc_vW = vW / (1-std::pow(beta2, 10));
  return (1 - learn_rate * L2) * W - learn_rate / sqrt(bc_vW + epsilon) % bc_mW;
}

vec Adam::updateb(vec b, vec db) {
  
  // Calculate mW and bias corrected mW
  mb = beta1 * mb + (1 - beta1) * db;
  vec bc_mb = mb / (1-std::pow(beta1, 10));
  
  // Calculate mW and bias corrected mW
  vb = beta2 * vb + (1 - beta2) * square(db);
  vec bc_vb = vb / (1-std::pow(beta2, 10));
  return b - learn_rate / sqrt(bc_vb + epsilon) % bc_mb;
}

// ---------------------------------------------------------------------------//
// Optimizer factory 
// ---------------------------------------------------------------------------//

std::unique_ptr<Optimizer> OptimizerFactory (mat W_templ, mat b_templ, List optim_param) {
  std::string type = as<std::string>(optim_param["type"]);
  if    (type == "sgd")     return std::unique_ptr<Optimizer>(new SGD(W_templ, b_templ, optim_param));
  if    (type == "rmsprop") return std::unique_ptr<Optimizer>(new RMSprop(W_templ, b_templ, optim_param));
  if    (type == "adam")    return std::unique_ptr<Optimizer>(new Adam(W_templ, b_templ, optim_param));
  else                      return NULL;
}

// [[Rcpp::export]]
vec rosenbrock (vec x, vec y) {
  return square(1 - x) + 100 * square(y - square(x));
}

// [[Rcpp::export]]
vec drosenbrock_x (vec x, vec y) {
  return -400 * x * (y - square(x)) - 2 * (1 - x);
}

// [[Rcpp::export]]
vec drosenbrock_y (vec x, vec y) {
  return 200 * (y - square(x));
}

RCPP_MODULE(SGD) {
  using namespace Rcpp ;
  class_<SGD>( "SGD" )
    .constructor<mat, vec, List>()
    .method( "updateW", &SGD::updateW)
    .method( "updateb", &SGD::updateb)
  ;
}

RCPP_MODULE(RMSprop) {
  using namespace Rcpp ;
  class_<RMSprop>( "RMSprop" )
    .constructor<mat, vec, List>()
    .method( "updateW", &RMSprop::updateW)
    .method( "updateb", &RMSprop::updateb)
  ;
}

RCPP_MODULE(Adam) {
  using namespace Rcpp ;
  class_<Adam>( "Adam" )
    .constructor<mat, vec, List>()
    .method( "updateW", &Adam::updateW)
    .method( "updateb", &Adam::updateb)
  ;
}



