#ifndef UTILS_H 
#define UTILS_H

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

mat repColVec(vec colvec, int n);

class scaler 
{
private:
  rowvec x_mu, x_sd;
  bool standardize;
public:
  scaler (List net_param_);
  void fit(mat x);
  mat scale(mat x);
  mat unscale(mat x);
};

class sampler 
{
private:
  mat X_train, Y_train, X_val, Y_val;
  int batch_size;
  uvec batch_bounds;
  std::list<uvec> batch_indices;
  std::list<uvec>::iterator uit;
public:
  sampler (mat X_, mat y_, int batch_size, double val_prop_);
  void shuffle();
  mat nextBatchX();
  mat nextBatchY();
  mat getValX();
  mat getValY();
};

#endif