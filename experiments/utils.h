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
  int n_train;
  mat X_train, Y_train, X_val, Y_val;
  std::list<uvec> indices;
  std::list<uvec>::iterator Xit;
  std::list<uvec>::iterator Yit;
public:
  int n_batch;
  double batch_prop;
  bool validate;
  sampler (mat X_, mat y_, List train_param);
  void shuffle();
  mat nextBatchX();
  mat nextBatchY();
  mat getValX();
  mat getValY();
  mat getTrainX();
  mat getTrainY();
};

// Base class optimizer
class tracker 
{
private:
  uword k;
public:
  mat train_history;
  tracker();
  void setTracker(int n_eval);
  void track(double train_loss, double val_loss);
};

#endif