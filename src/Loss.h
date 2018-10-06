#ifndef LOSS
#define LOSS

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// Base class Loss
class Loss {
public: 
  virtual double eval(mat y, mat y_fit);
  virtual mat grad(mat y, mat y_fit);
};

// Class for creating Loss classes that inherit from base class
class LossFactory
{
public:
  std::string type;
  List loss_param;
  LossFactory ();
  LossFactory (List loss_param_);
  Loss *createLoss ();
};

#endif