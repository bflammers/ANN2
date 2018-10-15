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

double LogLoss::eval(mat y, mat y_fit) {
  mat l = - y % log( y_fit );
  l = clamp(l, double_min, double_max);
  return accu(l) / y.n_rows;
}

mat LogLoss::grad(mat y, mat y_fit) { 
  return y_fit-y;
}

// ---------------------------------------------------------------------------//
// Squared loss class
// ---------------------------------------------------------------------------//

SquaredLoss::SquaredLoss () { type = "squared"; }
  
double SquaredLoss::eval(mat y, mat y_fit) {
  mat l = pow(y_fit - y, 2);
  return accu( l ) / y.n_rows;
}

mat SquaredLoss::grad(mat y, mat y_fit) { 
  return 2 * (y_fit - y);
}

// ---------------------------------------------------------------------------//
// Absolute loss class
// ---------------------------------------------------------------------------//

AbsoluteLoss::AbsoluteLoss () { type = "absolute"; }
  
double AbsoluteLoss::eval(mat y, mat y_fit) {
  mat l = abs(y_fit - y);
  return accu( l ) / y.n_rows;
}

mat AbsoluteLoss::grad(mat y, mat y_fit) { 
  return sign( y_fit - y );
}

// ---------------------------------------------------------------------------//
// Huber loss class
// ---------------------------------------------------------------------------//

HuberLoss::HuberLoss () { type = "huber"; }

HuberLoss::HuberLoss(List loss_param_)
    : delta_huber(loss_param_["delta_huber"]) { type = "huber"; }
  
double HuberLoss::eval(mat y, mat y_fit) {
  mat E   = abs(y_fit - y);
  mat l   = delta_huber * (E - delta_huber/2);
  uvec iE = find(E <= delta_huber);
  l(iE)   = pow(E(iE), 2)/2;
  return accu( l ) / y.n_rows;
}

mat HuberLoss::grad(mat y, mat y_fit) { 
  mat E   = y_fit - y;
  mat dl  = delta_huber * sign(E);
  uvec iE = find(abs(E) <= delta_huber);
  dl(iE)  = E(iE);
  return dl;
}

// ---------------------------------------------------------------------------//
// Pseudo-Huber loss class
// ---------------------------------------------------------------------------//

PseudoHuberLoss::PseudoHuberLoss () { type = "pseudo-huber"; }

PseudoHuberLoss::PseudoHuberLoss (List loss_param_)
    : delta_huber(loss_param_["delta_huber"]) { type = "pseudo-huber"; }
  
double PseudoHuberLoss::eval(mat y, mat y_fit) {
  mat l = sqrt(1 + pow( (y_fit - y) / delta_huber, 2)) - 1;
  return accu( l ) / y.n_rows;
}

mat PseudoHuberLoss::grad(mat y, mat y_fit) { 
  mat E = y_fit - y;
  return E % ( 1 / sqrt(1 + pow(E/delta_huber, 2)) );
}

// ---------------------------------------------------------------------------//
// Loss factory 
// ---------------------------------------------------------------------------//

std::unique_ptr<Loss> LossFactory (List loss_param)
{
  std::string type = as<std::string>(loss_param["type"]);
  if      (type == "log")         return std::unique_ptr<LogLoss>(new LogLoss());
  else if (type == "squared")     return std::unique_ptr<SquaredLoss>(new SquaredLoss());
  else if (type == "absolute")    return std::unique_ptr<AbsoluteLoss>(new AbsoluteLoss());
  else if (type == "huber")       return std::unique_ptr<HuberLoss>(new HuberLoss(loss_param));
  else if (type == "pseudoHuber") return std::unique_ptr<PseudoHuberLoss>(new PseudoHuberLoss(loss_param));
  else                            return NULL;
}

