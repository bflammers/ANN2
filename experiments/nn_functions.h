#ifndef NN_FUNCTIONS_H 
#define NN_FUNCTIONS_H

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

typedef mat (*funcPtr)(mat& x, int& H, int& k);
XPtr<funcPtr> assignActivation(String activation_);
XPtr<funcPtr> assignDerivative(String activation_);

#endif