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
