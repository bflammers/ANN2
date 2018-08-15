// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "loss.h"
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Loss functions
// ---------------------------------------------------------------------------//
mat logLoss(mat& y, mat& y_fit, double& dHuber){
  vec result = -log( y_fit.elem(find(y == 1)) );
  return clamp(result, std::numeric_limits<double>::min(),
               std::numeric_limits<double>::max());
}

mat squaredLoss(mat& y, mat& y_fit, double& dHuber){
  return sum(pow(y_fit - y, 2), 1);
}

mat absoluteLoss(mat& y, mat& y_fit, double& dHuber){
  return sum(abs(y_fit - y), 1);
}

mat pseudoHuberLoss(mat& y, mat& y_fit, double& dHuber){
  return sum(sqrt(1 + pow( (y_fit.t() - y) / dHuber, 2)) - 1, 1);
}

// ---------------------------------------------------------------------------//
// Loss functions derivatives
// ---------------------------------------------------------------------------//
mat logLossDeriv(mat& y, mat& y_fit, double& dHuber){
  return y_fit-y;
}

mat squaredLossDeriv(mat& y, mat& y_fit, double& dHuber){
  return 2 * (y_fit - y);
}

mat absoluteLossDeriv(mat& y, mat& y_fit, double& dHuber){
  return sign( y_fit - y );
}

mat pseudoHuberLossDeriv(mat& y, mat& y_fit, double& dHuber){
  mat E = y_fit - y;
  return E % ( 1 / sqrt(1 + pow(E/dHuber, 2)) );
}

// ---------------------------------------------------------------------------//
// Assign loss function and derivative based on string
// ---------------------------------------------------------------------------//
typedef mat (*funcPtrL)(mat& y, mat& y_fit, double& dHuber);

XPtr<funcPtrL> assignLoss(String loss_) {
  if (loss_ == "log")
    return(XPtr<funcPtrL>(new funcPtrL(&logLoss)));
  else if (loss_ == "squared")
    return(XPtr<funcPtrL>(new funcPtrL(&squaredLoss)));
  else if (loss_ == "absolute")
    return(XPtr<funcPtrL>(new funcPtrL(&absoluteLoss)));
  else if (loss_ == "pseudo-huber")
    return(XPtr<funcPtrL>(new funcPtrL(&pseudoHuberLoss)));
  else
    return XPtr<funcPtrL>(R_NilValue); // runtime error as NULL no XPtr
}

XPtr<funcPtrL> assignLossDeriv(String loss_) {
  if (loss_ == "log")
    return(XPtr<funcPtrL>(new funcPtrL(&logLossDeriv)));
  else if (loss_ == "squared")
    return(XPtr<funcPtrL>(new funcPtrL(&squaredLossDeriv)));
  else if (loss_ == "absolute")
    return(XPtr<funcPtrL>(new funcPtrL(&absoluteLossDeriv)));
  else if (loss_ == "pseudo-huber")
    return(XPtr<funcPtrL>(new funcPtrL(&pseudoHuberLossDeriv)));
  else
    return XPtr<funcPtrL>(R_NilValue); // runtime error as NULL no XPtr
}


// ---------------------------------------------------------------------------//
// Loss class
// ---------------------------------------------------------------------------//
loss::loss(String loss_, double dHuber_) : dHuber(dHuber_) {
    
  // Assign activation function based on string
  XPtr<funcPtrL> L_pointer = assignLoss(loss_);
  L = *L_pointer;
    
  // Assign derivative function based on string
  XPtr<funcPtrL> dL_pointer = assignLossDeriv(loss_);
  dL = *dL_pointer;
}
  
mat loss::eval (mat y, mat y_fit) {
  return L(y, y_fit, dHuber);
}
  
mat loss::grad (mat y, mat y_fit) {
  return dL(y, y_fit, dHuber);
}
