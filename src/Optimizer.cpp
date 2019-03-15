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
  : learn_rate ( optim_param_["learn_rate"]   ), 
    L1         ( optim_param_["L1"]           ), 
    L2         ( optim_param_["L2"]           ),
    momentum   ( optim_param_["sgd_momentum"] ) {
  
  // Set optimizer type  
  type = "SGD";
  
  // Initialize momentum matrices
  mW = zeros<mat>(size(W_templ_));
  mb = zeros<vec>(size(b_templ_));
}

mat SGD::updateW(mat W, mat dW, int batch_size) {
  
  // Calculate momentum term
  mW = momentum * mW + learn_rate * dW.t();
  
  // Determine scaled regularization parameters
  double lambda1 = double(batch_size) / n_train * L1;
  double lambda2 = double(batch_size) / n_train * L2;
  
  // Update weights with momentum term using L1 and L2 regularization
  // note that sign(0) = 0, which is what we want
  return (1 - learn_rate * lambda2) * W - learn_rate * lambda1 * sign(W) - mW;
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

// Actually used constructor
RMSprop::RMSprop (mat W_templ_, vec b_templ_, List optim_param_) 
  : learn_rate ( optim_param_["learn_rate"]      ),
    L1         ( optim_param_["L1"]              ), 
    L2         ( optim_param_["L2"]              ),
    decay      ( optim_param_["rmsprop_decay"]   ),
    epsilon    ( 1E-8 ) {
  
  // Set optimizer type  
  type = "RMSprop";
  
  // Initialize momentum matrices
  rmsW = zeros<mat>(size(W_templ_));
  rmsb = zeros<vec>(size(b_templ_));
}

// Update step WEIGHTS
mat RMSprop::updateW(mat W, mat dW, int batch_size) {
  
  // Calculate leaky RMS metric of squared dW to scale dW by before update step
  rmsW = decay * rmsW + (1 - decay) * square(dW.t());
  
  // Calculate gradient descent step to take
  mat W_step = ( learn_rate / (sqrt(rmsW) + epsilon) ) % dW.t();
  
  // Determine scaled regularization parameters
  double lambda1 = double(batch_size) / n_train * L1;
  double lambda2 = double(batch_size) / n_train * L2;
  
  // Update weights with momentum term using L1 and L2 regularization
  // note that sign(0) = 0, which is what we want
  return (1 - learn_rate * lambda2) * W - learn_rate * lambda1 * sign(W) - W_step;
}

// Update step BIASES
vec RMSprop::updateb(vec b, vec db) {
  
  // Calculate leaky RMS metric of squared db 
  rmsb = decay * rmsb + (1 - decay) * square(db);
  
  // Calculate gradient descent step to take
  vec b_step = ( learn_rate / sqrt(rmsb + epsilon) ) % db;
  
  // Return updated bias vector, no regularization for these
  return b - b_step;
}

// ---------------------------------------------------------------------------//
// Adam optimizer
// ---------------------------------------------------------------------------//

// Default constructor needed for serialization
Adam::Adam () { type = "Adam"; } 

Adam::Adam (mat W_templ_, vec b_templ_, List optim_param_) 
  : learn_rate ( optim_param_["learn_rate"] ),
    L1         ( optim_param_["L1"]         ), 
    L2         ( optim_param_["L2"]         ),
    beta1      ( optim_param_["adam_beta1"] ),
    beta2      ( optim_param_["adam_beta2"] ),
    epsilon    ( 1E-8 ),
    tW         ( 1    ),
    tb         ( 1    )
{
  
  // Set optimizer type  
  type = "Adam";
  
  // Initialize matrices
  mW = zeros<mat>(size(W_templ_));
  vW = zeros<mat>(size(W_templ_));
  
  // Initialize vectors
  mb = zeros<vec>(size(b_templ_));
  vb = zeros<vec>(size(b_templ_));
}

// Update step WEIGHTS
mat Adam::updateW(mat W, mat dW, int batch_size) {
  
  // Calculate mW and bias corrected mW
  mW = beta1 * mW + (1 - beta1) * dW.t();
  mat bc_mW = mW / (1-std::pow(beta1, tW));
  
  // Calculate mW and bias corrected mW
  vW = beta2 * vW + (1 - beta2) * square(dW.t());
  mat bc_vW = vW / (1-std::pow(beta2, tW));
  
  // Calculate gradient descent step to take
  mat W_step = ( learn_rate / (sqrt(bc_vW) + epsilon) ) % bc_mW;
  
  // Determine scaled regularization parameters
  double lambda1 = double(batch_size) / n_train * L1;
  double lambda2 = double(batch_size) / n_train * L2;
  
  // Increase counter for number of updates (used for bias correction)
  tW++;
  
  // Update weights with momentum term using L1 and L2 regularization
  // note that sign(0) = 0, which is what we want
  return (1 - learn_rate * lambda2) * W - learn_rate * lambda1 * sign(W) - W_step;
}

// Update step BIASES
vec Adam::updateb(vec b, vec db) {
  
  // Calculate mW and bias corrected mW
  mb = beta1 * mb + (1 - beta1) * db;
  vec bc_mb = mb / (1-std::pow(beta1, tb));
  
  // Calculate mW and bias corrected mW
  vb = beta2 * vb + (1 - beta2) * square(db);
  vec bc_vb = vb / (1-std::pow(beta2, tb));
  
  // Calculate gradient descent step to take
  mat b_step = ( learn_rate / (sqrt(bc_vb) + epsilon) ) % bc_mb;
  
  // Increase counter for number of updates (used for bias correction)
  tb++;
  
  // Return updated bias vector, no regularization for these
  return b - b_step;
}

// ---------------------------------------------------------------------------//
// Optimizer factory 
// ---------------------------------------------------------------------------//

std::unique_ptr<Optimizer> OptimizerFactory (mat W_templ, mat b_templ, List optim_param) {
  std::string type = as<std::string>(optim_param["type"]);
  if      (type == "sgd")     return std::unique_ptr<Optimizer>(new SGD(W_templ, b_templ, optim_param));
  else if (type == "rmsprop") return std::unique_ptr<Optimizer>(new RMSprop(W_templ, b_templ, optim_param));
  else if (type == "adam")    return std::unique_ptr<Optimizer>(new Adam(W_templ, b_templ, optim_param));
  else stop("optim.type not implemented");
}



