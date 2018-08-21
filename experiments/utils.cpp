// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
mat repColVec(vec colvec, int n){
  mat result(colvec.size(), n);
  result.each_col() = colvec;
  return result;
}

// ---------------------------------------------------------------------------//
// Scaler class
// ---------------------------------------------------------------------------//
scaler::scaler (List net_param_) 
  : standardize(net_param_["standardize"])
{
  ivec num_nodes = net_param_["num_nodes"];
  x_mu = zeros<rowvec>(num_nodes[0]);
  x_sd = ones<rowvec>(num_nodes[num_nodes.size() - 1]);
}

void scaler::fit (mat x)
{
  if ( standardize ) {
    x_mu = mean(x);
    x_sd = stddev(x);
  } 
}

mat scaler::scale(mat x) 
{ 
  if ( standardize ) {
    x.each_row() -= x_mu;
    x.each_row() /= x_sd;
  }
  return x;
}

mat scaler::unscale(mat x) 
{
  if ( standardize ) {
    x.each_row() %= x_sd;
    x.each_row() += x_mu;
  }
  return x;
}

