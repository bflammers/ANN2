// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"
using namespace Rcpp;
using namespace arma;

RCPP_MODULE(mod_sampler) {
  class_<sampler>( "sampler" )
  .constructor<mat, mat, int, double>()
  .method( "nextBatchX", &sampler::nextBatchX)
  .method( "nextBatchY", &sampler::nextBatchY)
  .method( "getValX", &sampler::getValX)
  .method( "getValY", &sampler::getValY)
  ;
}

// [[Rcpp::export]]
uvec testfn(mat X, double batch_size) {
  
  int n_obs = X.n_rows;
  int n_batch = ceil(n_obs / batch_size);
  
  // Randomly shuffle X and Y
  uvec rand_perm = round(linspace<uvec>(0, n_obs - 1, n_batch * batch_size));
  rand_perm = arma::shuffle(rand_perm);
  uvec batch_range = regspace<uvec>(0, batch_size - 1);
  
  std::list<uvec> indices;
  
  for (int i = 0; i != n_batch; i++) {
    indices.push_back ( rand_perm(batch_range + i * batch_size) );
    Rcout << batch_range + i * batch_size << std::endl;
  }
  
  return rand_perm;
};



/*** R
X <- matrix(rnorm(100), 25, 4)
Y <- matrix(sample(1:2, 50, replace = TRUE), 25, 2)
s <- new(sampler, X, Y, 10, 0.1)

a <- testfn(X, 2)
table(a)
#s$getValX()
#s$nextBatchX()

*/