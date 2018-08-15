#ifndef ACTIVATIONS
#define ACTIVATIONS

#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// Function pointer to assign activation and derivative based on string
typedef mat (*funcPtrA)(mat& X, int& H, int& k);

// Activation class
class activation {
private:
  int H, k;
  funcPtrA g, dg;
  
public:
  activation(String activation_, int H_, int k_);
  mat eval (mat X);
  mat grad (mat X);
  
};

#endif