// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "nn_functions.h"
using namespace Rcpp;
using namespace arma;

// ----------- Activation functions ----------- //
mat tanhActivation(mat& X, int& H, int& k){
  return 1.725*tanh(2*X/3);
}

mat reluActivation(mat& X, int& H, int& k){
  return clamp(X, 0, X.max()); 
}

mat linearActivation(mat& X, int& H, int& k){
  return X;
}

// ----------- Derivative functions ----------- //
mat tanhDerivative(mat& X, int& H, int& k){
  return 1.15*(1-pow(tanh(2*X/3), 2));
}

mat reluDerivative(mat& X, int& H, int& k){
  mat d(size(X), fill::zeros);
  d.elem(find(X > 0)).fill(1);
  return d; 
}

mat linearDerivative(mat& X, int& H, int& k){
  return X.fill(1);
}


// ----------- Assign functions based on string ----------- //
typedef mat (*funcPtr)(mat& X, int& H, int& k);

XPtr<funcPtr> assignActivation(String activation_) {
  if (activation_ == "tanh")
    return(XPtr<funcPtr>(new funcPtr(&tanhActivation)));
  else if (activation_ == "relu")
    return(XPtr<funcPtr>(new funcPtr(&reluActivation)));
  else if (activation_ == "linear")
    return(XPtr<funcPtr>(new funcPtr(&linearActivation)));
  else
    return XPtr<funcPtr>(R_NilValue); // runtime error as NULL no XPtr
}

XPtr<funcPtr> assignDerivative(String activation_) {
  if (activation_ == "tanh")
    return(XPtr<funcPtr>(new funcPtr(&tanhDerivative)));
  else if (activation_ == "relu")
    return(XPtr<funcPtr>(new funcPtr(&reluDerivative)));
  else if (activation_ == "linear")
    return(XPtr<funcPtr>(new funcPtr(&linearDerivative)));
  else
    return XPtr<funcPtr>(R_NilValue); // runtime error as NULL no XPtr
}


