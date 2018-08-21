// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;


// [[Rcpp::export]]
int testfun(ivec x) { 
  return x[0];
}

/*** R
x <- 5:15
testfun(x)

# library("microbenchmark")
# microbenchmark(s$scale(x), s$unscale(t), times = 1000)
*/