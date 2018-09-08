// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "activations.h"
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
mat testFactory(List activ_params, mat X) {
  activationFactory fact(activ_params);
  activation *A = NULL; 
  A = fact.createActivation();
  return A->eval(X.t()).t();
};



/*** R
activ_param <- list(type = 'softmax')
X <- cbind(rnorm(5, sd = 5), rnorm(5, sd = 1))
z <- testFactory(activ_param, X)
z
rowSums(z)
*/
