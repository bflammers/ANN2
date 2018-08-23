// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"

RCPP_MODULE(mod_sampler) {
  class_<sampler>( "sampler" )
  .constructor<List>()
  .method( "sampleTrainX", &sampler::sampleTrainX)
  .method( "sampleTrainY", &sampler::sampleTrainY)
  .method( "getValX", &sampler::getValX)
  .method( "getValY", &sampler::getValY)
  ;
}

/*** R
X <- matrix(rnorm(200), 50, 4)
Y <- matrix(sample(1:2, 100, replace = TRUE), 50, 2)
s <- new(sampler, X, Y, 32, 0.8)

*/