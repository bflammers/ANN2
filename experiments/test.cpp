// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
mat repVec(vec vector, int ntimes){
  mat result(vector.size(), ntimes);
  for(int i = 0; i!=ntimes; i++){
    result.col(i) =vector;
  }
  return(result);
}

// [[Rcpp::export]]
mat repColVector(vec colvec, int n){
  mat result(colvec.size(), n);
  result.each_col() = colvec;
  return result;
}

// [[Rcpp::export]]
mat repColVec2(vec colvec, int n){
  mat result(colvec.size(), n);
  result.each_col() = colvec;
  return result;
}

/*** R
vec <- 1:10
repVec(vec, 3)
repColVector(vec, 3)
*/
