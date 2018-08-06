#include <Rcpp.h>
using namespace Rcpp;

class Bar {
public:
  Bar(double x_) : x(x_), nread(0), nwrite(0) {}
  
  double get_x() {
    nread++;
    return x;
  }
  
  void set_x(double x_) {
    nwrite++;
    x = x_;
  }
  
  IntegerVector stats() const {
    return
    IntegerVector::create(_["read"] = nread,
                          _["write"] = nwrite);
  }
private:
  double x;
  int nread, nwrite;
};

RCPP_MODULE(mod_bar) {
  class_<Bar>( "Bar" )
  .constructor<double>()
  .property( "x", &Bar::get_x, &Bar::set_x )
  .method( "stats", &Bar::stats )
  ;
}


/*** R
b <- new(Bar, 10)
b$x + b$x
b$stats()
b$x <- 20
b$stats()

*/
