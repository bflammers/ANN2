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

// ---------------------------------------------------------------------------//
// Sampler class
// ---------------------------------------------------------------------------//
sampler::sampler (mat X_, mat Y_, int batch_size_, double val_prop_) 
  : batch_size(batch_size_)
{
  int n_obs = X_.n_rows;
  int n_train = ceil ( (1 - val_prop_ ) * n_obs);
  
  // Randomly shuffle X and Y
  uvec rand_perm = arma::shuffle(regspace<uvec>(0, n_obs - 1));
  mat X = X_.rows(rand_perm);
  mat Y = Y_.rows(rand_perm);
  
  // Divide X and Y in training and validation sets
  X_train = X.rows(regspace<uvec>(0, n_train - 1));
  Y_train = X.rows(regspace<uvec>(0, n_train - 1));
  X_val = X.rows(regspace<uvec>(n_train, n_obs - 1));
  Y_val = X.rows(regspace<uvec>(n_train, n_obs - 1));
  
  // Determine bounds of batches
  batch_bounds = regspace<uvec>(0, batch_size, n_obs + batch_size - 1);
  batch_bounds = unique(clamp(batch_bounds, 0, n_obs - 1));
}

void sampler::shuffle () {
  // Set list of batch indices to new random uvecs
  
  // Set list iterators to begin
};
mat sampler::nextBatchX () { return X_train; };
mat sampler::nextBatchY () { return Y_train; };
mat sampler::getValX () { return X_val; };
mat sampler::getValY () { return Y_val; };

