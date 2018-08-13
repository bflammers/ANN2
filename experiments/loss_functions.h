#ifndef LOSS_FUNCTIONS
#define LOSS_FUNCTIONS

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

typedef mat (*funcPtrL)(mat& y, mat& y_fit, double& dHuber);
XPtr<funcPtrL> assignLoss(String activation_);
XPtr<funcPtrL> assignLossDeriv(String activation_);

#endif