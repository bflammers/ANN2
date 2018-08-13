#ifndef ACTIVATION_FUNCTIONS
#define ACTIVATION_FUNCTIONS

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

typedef mat (*funcPtrA)(mat& X, int& H, int& k);
XPtr<funcPtrA> assignActiv(String activation_);
XPtr<funcPtrA> assignActivDeriv(String activation_);

#endif