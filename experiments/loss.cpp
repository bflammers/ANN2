// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "loss.h"
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Base loss class
// ---------------------------------------------------------------------------//

mat loss::eval(mat y, mat y_fit) { return y.zeros(); }
mat loss::grad(mat y, mat y_fit) { return y.zeros(); }

// ---------------------------------------------------------------------------//
// Loss classes
// ---------------------------------------------------------------------------//

class logLoss : public loss
{
public:
  logLoss () {}
  mat eval(mat y, mat y_fit) {
    vec result = -log( y_fit.elem(find(y == 1)) );
    return clamp(result, std::numeric_limits<double>::min(),
                 std::numeric_limits<double>::max());
  }
  mat grad(mat y, mat y_fit) { 
    return y_fit-y;
  }
};

class squaredLoss : public loss 
{
public:
  squaredLoss () {}
  mat eval(mat y, mat y_fit) { 
    return sum(pow(y_fit - y, 2), 1);
  }
  mat grad(mat y, mat y_fit) { 
    return 2 * (y_fit - y);
  }
};

class absoluteLoss : public loss
{
public:
  absoluteLoss () {}
  mat eval(mat y, mat y_fit) { 
    return sum(abs(y_fit - y), 1);
  }
  mat grad(mat y, mat y_fit) { 
    return sign( y_fit - y );
  }
};

class pseudoHuberLoss : public loss
{
public:
  double dHuber;
  pseudoHuberLoss(List loss_param_) {
    dHuber = loss_param_["dHuber"];
  }
  mat eval(mat y, mat y_fit) { 
    return sum(sqrt(1 + pow( (y_fit.t() - y) / dHuber, 2)) - 1, 1);
  }
  mat grad(mat y, mat y_fit) { 
    mat E = y_fit - y;
    return E % ( 1 / sqrt(1 + pow(E/dHuber, 2)) );
  }
};

// ---------------------------------------------------------------------------//
// Methods for loss factory 
// ---------------------------------------------------------------------------//

// Constructor
lossFactory::lossFactory (List loss_param_) : loss_param(loss_param_) {
  // Set optimization type
  type = as<std::string>(loss_param["type"]);
}

// Method for creating optimizers
loss *lossFactory::createLoss () {
  if      (type == "log")         return new logLoss();
  else if (type == "squared")     return new squaredLoss();
  else if (type == "absolute")    return new absoluteLoss();
  else if (type == "pseudoHuber") return new pseudoHuberLoss(loss_param);
  else {
    Rcout << "\n\nloss factory failed!!!!\n\n";
    return NULL;
  }                            
}

