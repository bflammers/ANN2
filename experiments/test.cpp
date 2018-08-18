// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins("cpp11")]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
arma::mat colSoftMax(arma::mat x) {
  int nrow = x.n_rows;
  mat out(x);
  for (int i = 0; i < nrow; i++) {
    rowvec exp_x = exp( x.row(i) - max(x.row(i)) );
    out.row(i)      = exp_x / sum(exp_x);
  }
  return out;
}

// [[Rcpp::export]]
mat softMax(mat X) {
  rowvec max_X = max(X);
  X.each_row() -= max_X;
  mat A = exp(X);
  rowvec t = sum(A);
  A.each_row() /= t;
  return A;
}

/*** R
x <- matrix(rnorm(15), 5, 3)
tx <- t(x)

colSoftMax(x)
softMax(tx)

library('rbenchmark')
benchmark(colSoftMax(x), softMax(tx), replications = 1000000)
*/