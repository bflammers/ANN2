// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "nn_functions.h"
using namespace Rcpp;
using namespace arma;

// ----------- Activation functions ----------- //
mat tanhActivation(mat& x){
  return 1.725*tanh(2*x/3);
}

mat reluActivation(mat& x){
  return clamp(x, 0, x.max()); 
}

// ----------- Used to declare activation based on string ----------- //
typedef mat (*funcPtr)(mat& x);

XPtr<funcPtr> assignActivation(String activation_) {
  if (activation_ == "tanh")
    return(XPtr<funcPtr>(new funcPtr(&tanhActivation)));
  else if (activation_ == "relu")
    return(XPtr<funcPtr>(new funcPtr(&reluActivation)));
  else
    return XPtr<funcPtr>(R_NilValue); // runtime error as NULL no XPtr
}