#ifndef LOSS
#define LOSS

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

typedef mat (*funcPtrL)(mat& y, mat& y_fit, double& dHuber);

class loss {
private:
  double dHuber;
  funcPtrL L, dL;
  
public:
  loss(String loss_, double dHuber_);
  mat eval (mat y, mat y_fit);
  mat grad (mat y, mat y_fit);
  
};


#endif