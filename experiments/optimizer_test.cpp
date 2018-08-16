// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "optimizer.h"
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
void testFactory(String type, mat W_, vec b_, double lambda_, double m_,
                   double L1_, double L2_, mat D_, mat A_prev_) {
  sgdParams P(lambda_, m_, L1_, L2_);
  optimizerFactory fact(W_, b_);
  optimizer *O = NULL; 
  O = fact.createOptimizer(type);
  O->setParams(P);
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
testFactory("SGD", W, b, 0.0001, 0.8, 0.5, 0.3, D, A_prev)
*/
