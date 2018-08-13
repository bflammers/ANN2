// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "activation_functions.h"
using namespace Rcpp;
using namespace arma;

// ----------- Activation functions ----------- //
mat tanhActiv(mat& X, int& H, int& k){
  return 1.725*tanh(2*X/3);
}

mat reluActiv(mat& X, int& H, int& k){
  return clamp(X, 0, X.max()); 
}

mat linearActiv(mat& X, int& H, int& k){
  return X;
}

// -------- Activation functions derivatives -------- //
mat tanhActivDeriv(mat& X, int& H, int& k){
  return 1.15*(1-pow(tanh(2*X/3), 2));
}

mat reluActivDeriv(mat& X, int& H, int& k){
  mat d(size(X), fill::zeros);
  d.elem(find(X > 0)).fill(1);
  return d; 
}

mat linearActivDeriv(mat& X, int& H, int& k){
  return X.fill(1);
}

// ----------- Assign functions based on string ----------- //
typedef mat (*funcPtrA)(mat& X, int& H, int& k);

XPtr<funcPtrA> assignActiv(String activation_) {
  if (activation_ == "tanh")
    return(XPtr<funcPtrA>(new funcPtrA(&tanhActiv)));
  else if (activation_ == "relu")
    return(XPtr<funcPtrA>(new funcPtrA(&reluActiv)));
  else if (activation_ == "linear")
    return(XPtr<funcPtrA>(new funcPtrA(&linearActiv)));
  else
    return XPtr<funcPtrA>(R_NilValue); // runtime error as NULL no XPtr
}

XPtr<funcPtrA> assignActivDeriv(String activation_) {
  if (activation_ == "tanh")
    return(XPtr<funcPtrA>(new funcPtrA(&tanhActivDeriv)));
  else if (activation_ == "relu")
    return(XPtr<funcPtrA>(new funcPtrA(&reluActivDeriv)));
  else if (activation_ == "linear")
    return(XPtr<funcPtrA>(new funcPtrA(&linearActivDeriv)));
  else
    return XPtr<funcPtrA>(R_NilValue); // runtime error as NULL no XPtr
}
