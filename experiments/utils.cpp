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

class scaler 
{
private:
  rowvec X_mu, X_sd;
public:
  scaler () {}
  
  void fit(mat X) 
  {
    X_mu = mean(X);
    X_sd = stddev(X);
  }
  
  mat transform(mat X) 
  { 
    X.each_row() -= X_mu;
    X.each_row() /= X_sd;
    return X;
  }
};

RCPP_MODULE(mod_scaler) {
  class_<scaler>( "scaler" )
  .constructor()
  .method( "fit", &scaler::fit)
  .method( "transform", &scaler::transform)
  ;
}

/*** R
n <- 1000
x <- cbind(rnorm(n, 3, 9), rnorm(n, -1, 0.4))
s <- new(scaler)

s$fit(x)
t <- s$transform(x)
colMeans(t)
apply(t, 2, sd)
#library("microbenchmark")
#microbenchmark(sfO$eval(x), sf$eval(x), replications = 100000)
#microbenchmark(sfO$grad(x), sf$grad(x), replications = 100000)
  */