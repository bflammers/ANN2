// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"
using namespace Rcpp;
using namespace arma;

RCPP_MODULE(mod_sampler) {
  class_<sampler>( "sampler" )
  .constructor<mat, mat, List>()
  .method( "shuffle", &sampler::shuffle)
  .method( "nextBatchX", &sampler::nextBatchX)
  .method( "nextBatchY", &sampler::nextBatchY)
  .method( "getValX", &sampler::getValX)
  .method( "getValY", &sampler::getValY)
  ;
}


// class tester 
// {
// public:
//   int a;
//   tester(int b) : a(b) {};
// };
// 
// // [[Rcpp::export]]
// void testfn(int b) {
//   tester t(b);
//   Rcout << t.a;
// }

// [[Rcpp::export]]
void tester(mat X, int a, int b) {
  // Divide X and Y in training and validation sets
  X.resize(a,b);
  Rcout << X;
}

/*** R
 
X <- matrix(1:100, 25, 4)
Y <- matrix(1:50, 25, 2)

tester(matrix(1:4, 2, 2), 4, 2)

# train_param <- list(batch_size = 10, val_prop = 0.1)
# s <- new(sampler, X, Y, train_param)
# 
# X
# Y
# 
# s$getValX()
# s$getValY()
# s$nextBatchX()
# s$nextBatchY()
# s$shuffle()
# 
# any( s$getValX() %in% s$nextBatchX() )


*/