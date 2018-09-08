// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;



// [[Rcpp::export]]
void test(int a, int b, double c){
  Rcout << a % b;
  Rcout << std::endl;
  Rcout << std::ceil(c);
}


/*** R
test(9, 4, 1.1)

*/