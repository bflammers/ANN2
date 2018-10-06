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
// Base Loss class
// ---------------------------------------------------------------------------//

double Loss::eval(mat y, mat y_fit) { return 0; }
mat Loss::grad(mat y, mat y_fit) { return 0; }

// ---------------------------------------------------------------------------//
// Loss classes
// ---------------------------------------------------------------------------//

class LogLoss : public Loss
{
public:
  LogLoss () {}
  
  double eval(mat y, mat y_fit) 
  {
    mat l = - y % log( y_fit );
    l = clamp(l, double_min, double_max);
    return accu(l) / y.n_rows;
  }
  
  mat grad(mat y, mat y_fit) 
  { 
    return y_fit-y;
  }
};

class SquaredLoss : public Loss 
{
public:
  SquaredLoss () {}
  
  double eval(mat y, mat y_fit) 
  {
    mat l = pow(y_fit - y, 2);
    return accu( l ) / y.n_rows;
  }
  
  mat grad(mat y, mat y_fit) 
  { 
    return 2 * (y_fit - y);
  }
};

class AbsoluteLoss : public Loss
{
public:
  AbsoluteLoss () {}
  
  double eval(mat y, mat y_fit) 
  {
    mat l = abs(y_fit - y);
    return accu( l ) / y.n_rows;
  }
  
  mat grad(mat y, mat y_fit) 
  { 
    return sign( y_fit - y );
  }
};

class HuberLoss : public Loss
{
private:
  double delta_huber;
public:
  HuberLoss(List loss_param_) 
    : delta_huber(loss_param_["delta_huber"]) {}
  
  double eval(mat y, mat y_fit) 
  {
    mat E   = abs(y_fit - y);
    mat l   = delta_huber * (E - delta_huber/2);
    uvec iE = find(E <= delta_huber);
    l(iE)   = pow(E(iE), 2)/2;
    return accu( l ) / y.n_rows;
  }
  
  mat grad(mat y, mat y_fit) 
  { 
    mat E   = y_fit - y;
    mat dl  = delta_huber * sign(E);
    uvec iE = find(abs(E) <= delta_huber);
    dl(iE)  = E(iE);
    return dl;
  }
};

class PseudoHuberLoss : public Loss
{
private:
  double delta_huber;
public:
  PseudoHuberLoss(List loss_param_) 
    : delta_huber(loss_param_["delta_huber"]) {}
  
  double eval(mat y, mat y_fit) 
  {
    mat l = sqrt(1 + pow( (y_fit - y) / delta_huber, 2)) - 1;
    return accu( l ) / y.n_rows;
  }
  
  mat grad(mat y, mat y_fit) 
  { 
    mat E = y_fit - y;
    return E % ( 1 / sqrt(1 + pow(E/delta_huber, 2)) );
  }
};

// ---------------------------------------------------------------------------//
// Methods for Loss factory 
// ---------------------------------------------------------------------------//

// Constructor
LossFactory::LossFactory (List loss_param_) : loss_param(loss_param_) 
{
  // Set optimization type
  type = as<std::string>(loss_param["type"]);
}

// Method for creating optimizers
Loss *LossFactory::createLoss () 
{
  if      (type == "log")         return new LogLoss();
  else if (type == "squared")     return new SquaredLoss();
  else if (type == "absolute")    return new AbsoluteLoss();
  else if (type == "huber")       return new HuberLoss(loss_param);
  else if (type == "pseudoHuber") return new PseudoHuberLoss(loss_param);
  else                            return NULL;
}

