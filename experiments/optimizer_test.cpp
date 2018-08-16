// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "optimizer.h"
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
void testFactory(mat W_, vec b_, List optim_params, mat D_, mat A_prev_) {
  optimizerFactory fact(W_, b_, optim_params);
  optimizer *O = NULL; 
  O = fact.createOptimizer();
  mat uW = O->updateW(W_, D_, A_prev_);
  Rcout << "W:\n" << uW << "\n\n"; 
  vec ub = O->updateb(b_, D_);
  Rcout << "b:\n" << ub << "\n\n"; 
};



/*** R
# c(2,5,4,3,2)
n_in <- 5
n_out <- 4
b_s <- 15
W <- matrix(rnorm(n_in * n_out), n_out, n_in)
b <- rnorm(n_out)
D <- matrix(rnorm(n_out * b_s), b_s, n_out)
A_prev <- matrix(rnorm(n_in * b_s), n_in, b_s)
optim_param <- list(type = 0, lambda = 0.001, m = 0.8, L1 = 0.001, L2 = 0.01)
testFactory(W, b, optim_param, D, A_prev)
*/
