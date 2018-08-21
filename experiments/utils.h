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

#endif