// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
void f1 (List ll) {
  Rcpp::List xlist(ll);
  double a = xlist["a"];
  int b = xlist["b"];
  String c = xlist["c"];
  Rcout << a << " ";
  Rcout << b << " ";
  //Rcout << c << " ";
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
l <- list(a = 0.3, b = 1, c = "hallo")
f1(l)
#f2(9)

#library('rbenchmark')
#benchmark(f1(10), f2(10), replications = 10000000)
*/