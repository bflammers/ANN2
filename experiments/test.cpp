// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
void f1 () {
  mat A(4,5, fill::zeros);
  size(A)
  Rcout << size(A);
}

// [[Rcpp::export]]
void f2 (int n) {
  List l;
  List::iterator it;
  
  for(int i = 0; i != n; ++i){
    l.push_back(i);
  }
  
  for(it = l.begin(); it != l.end(); ++it){
    n = *it;
    
    //Rcout << j << std::endl;
  }

}

/*** R
f1()
#f2(9)

#library('rbenchmark')
#benchmark(f1(10), f2(10), replications = 10000000)
*/