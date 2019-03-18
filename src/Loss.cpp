// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "Loss.h"

using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Constants
// ---------------------------------------------------------------------------//

double double_min = std::numeric_limits<double>::min();
double double_max = std::numeric_limits<double>::max();

// ---------------------------------------------------------------------------//
// Log loss class
// ---------------------------------------------------------------------------//

LogLoss::LogLoss () { type = "log"; }

mat LogLoss::eval(mat y, mat y_fit) {
  y_fit = clamp(y_fit, 1e-15, 1-1e-15);
  return - y % log( y_fit );
}

mat LogLoss::grad(mat y, mat y_fit) { 
  y_fit = clamp(y_fit, 1e-15, 1-1e-15);
  // Only for use with Softmax activation function!
  // This is not the elementwise derivative of the log loss function (this would
  // be -y/y_fit ) but the derivative wrt the inputs to the softmax function. 
  // Error is thrown for classification with another loss function!!
  // See: https://peterroelants.github.io/posts/cross-entropy-softmax/
  return y_fit - y; 
}

// ---------------------------------------------------------------------------//
// Squared loss class
// ---------------------------------------------------------------------------//

SquaredLoss::SquaredLoss () { type = "squared"; }
  
mat SquaredLoss::eval(mat y, mat y_fit) {
  return pow(y_fit - y, 2);
}

mat SquaredLoss::grad(mat y, mat y_fit) { 
  return 2 * (y_fit - y);
}

// ---------------------------------------------------------------------------//
// Absolute loss class
// ---------------------------------------------------------------------------//

AbsoluteLoss::AbsoluteLoss () { type = "absolute"; }
  
mat AbsoluteLoss::eval(mat y, mat y_fit) {
  return abs(y_fit - y);
}

mat AbsoluteLoss::grad(mat y, mat y_fit) { 
  return sign( y_fit - y );
}

// ---------------------------------------------------------------------------//
// Huber loss class
// ---------------------------------------------------------------------------//

HuberLoss::HuberLoss () { type = "huber"; }

HuberLoss::HuberLoss(List loss_param_)
    : huber_delta( loss_param_["huber_delta"] ) { type = "huber"; }
  
mat HuberLoss::eval(mat y, mat y_fit) {
  mat E   = abs(y_fit - y);
  mat l   = huber_delta * (E - huber_delta/2);
  uvec iE = find(E <= huber_delta);
  l(iE)   = pow(E(iE), 2)/2;
  return l;
}

mat HuberLoss::grad(mat y, mat y_fit) { 
  mat E   = y_fit - y;
  mat dl  = huber_delta * sign(E);
  uvec iE = find(abs(E) <= huber_delta);
  dl(iE)  = E(iE);
  return dl;
}

// ---------------------------------------------------------------------------//
// Pseudo-Huber loss class
// ---------------------------------------------------------------------------//

PseudoHuberLoss::PseudoHuberLoss () { type = "pseudo-huber"; }

PseudoHuberLoss::PseudoHuberLoss (List loss_param_)
    : huber_delta( loss_param_["huber_delta"] ) { type = "pseudo-huber"; }
  
mat PseudoHuberLoss::eval(mat y, mat y_fit) {
  return sqrt(1 + pow( (y_fit - y) / huber_delta, 2)) - 1;
}

mat PseudoHuberLoss::grad(mat y, mat y_fit) { 
  mat E = y_fit - y;
  return E % ( 1 / sqrt(1 + pow(E/huber_delta, 2)) );
}

// ---------------------------------------------------------------------------//
// Loss factory 
// ---------------------------------------------------------------------------//

std::unique_ptr<Loss> LossFactory (List loss_param)
{
  std::string type = as<std::string>(loss_param["type"]);
  if      (type == "log")          return std::unique_ptr<Loss>(new LogLoss());
  else if (type == "squared")      return std::unique_ptr<Loss>(new SquaredLoss());
  else if (type == "absolute")     return std::unique_ptr<Loss>(new AbsoluteLoss());
  else if (type == "huber")        return std::unique_ptr<Loss>(new HuberLoss(loss_param));
  else if (type == "pseudo-huber") return std::unique_ptr<Loss>(new PseudoHuberLoss(loss_param));
  else stop("loss.type not implemented");
}

