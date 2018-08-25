// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"
using namespace Rcpp;
using namespace arma;

// Make a matrix consisting of one repeated column
mat repColVec(vec colvec, int n)
{
  mat result(colvec.size(), n);
  result.each_col() = colvec;
  return result;
}

// Armadillo modulo function 
template<typename T>
T modulo(T a, int n)
{
  return a - floor(a/n)*n;
}

// ---------------------------------------------------------------------------//
// Scaler class
// ---------------------------------------------------------------------------//
tracker::tracker () : k(0) {}

void tracker::setTracker (int n_eval) 
{
  train_history.resize(n_eval, 2);
}

void tracker::track(double train_loss, double val_loss)
{
  Rcout << "k: " << k << std::endl << train_history;
  rowvec loss_vec = {train_loss, val_loss};
  train_history.row(k) = loss_vec;
  k += 1; 
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
sampler::sampler (mat X_, mat Y_, List train_param)
{
  // Training parameters 
  int batch_size = train_param["batch_size"];
  double val_prop = train_param["val_prop"];
  
  // Derived parameters
  int n_obs = X_.n_rows;
  n_train = ceil ( (1 - val_prop ) * n_obs );
  n_batch = ceil ( double( n_train ) / batch_size );
  batch_prop = double( batch_size ) / n_train;
  validate = ( n_train < n_obs );
  
  // Randomly shuffle X and Y for train/validation split
  uvec rand_perm = arma::shuffle(regspace<uvec>(0, n_obs - 1));
  mat X = X_.rows(rand_perm);
  mat Y = Y_.rows(rand_perm);
  
  // Divide X and Y in training and validation sets
  X_train = X.rows(0, n_train - 1);
  Y_train = Y.rows(0, n_train - 1);
  if ( validate ) {
    X_val = X.rows(n_train, n_obs - 1);
    Y_val = Y.rows(n_train, n_obs - 1);
  }
  
  // Fill indices list with uvecs to subset X_train and Y_train for batches
  // after shuffling in method .shuffle()
  // All batches consist of exactly n_batch observations
  // During n_batch batches, more than n_train observations can be used
  uvec epoch_range = regspace<uvec>(0, n_batch * batch_size);
  uvec train_range = modulo(epoch_range, n_train);
  uvec batch_range = regspace<uvec>(0, batch_size - 1);
  for (int i = 0; i != n_batch; i++) {
    indices.push_back ( train_range(batch_range + i * batch_size) );
  }
  
  // Set list iterators to begin
  Xit = indices.begin();
  Yit = indices.begin();
}

void sampler::shuffle () 
{
  // Randomly shuffle X and Y for train/validation split
  uvec rand_perm = arma::shuffle(regspace<uvec>(0, n_train - 1));
  X_train = X_train.rows(rand_perm);
  Y_train = Y_train.rows(rand_perm);
  
  // Set list iterators to begin
  Xit = indices.begin();
  Yit = indices.begin();
}

mat sampler::nextBatchX () 
{ 
  mat X_batch = X_train.rows( (*Xit) );
  std::advance(Xit, 1);
  return X_batch; 
}

mat sampler::nextBatchY () 
{ 
  mat Y_batch = Y_train.rows( (*Yit) );
  std::advance(Yit, 1);
  return Y_batch; 
}

mat sampler::getValX () 
{ 
  return X_val; 
}

mat sampler::getValY () 
{ 
  return Y_val; 
}

mat sampler::getTrainX () 
{
  return X_train; 
}

mat sampler::getTrainY () 
{ 
  return Y_train; 
}


