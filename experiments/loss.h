#ifndef LOSS
#define LOSS

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// Base class loss
class loss {
public: 
  virtual mat eval(mat y, mat y_fit);
  virtual mat grad(mat y, mat y_fit);
};

// Class for creating loss classes that inherit from base class
class lossFactory
{
public:
  std::string type;
  List loss_param;
  lossFactory ();
  lossFactory (List loss_param_);
  loss *createLoss ();
};

#endif