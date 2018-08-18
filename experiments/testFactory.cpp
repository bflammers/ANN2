// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "testFactory.h"
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Base activation clas
// ---------------------------------------------------------------------------//

mat activation::eval(mat X) { return X.ones(); }
mat activation::grad(mat X) { return X.ones(); }

// ---------------------------------------------------------------------------//
// Activation classes
// ---------------------------------------------------------------------------//

tanhActivation::tanhActivation () {};
mat tanhActivation::eval(mat X) {
  return 1.725*tanh(2*X/3);
};
mat tanhActivation::grad(mat X) { 
  return 1.15*(1-pow(tanh(2*X/3), 2));
};


// ---------------------------------------------------------------------------//
// Methods for activation factory 
// ---------------------------------------------------------------------------//

// Constructor
activationFactory::activationFactory (List activ_param_) : activ_param(activ_param_) {
  
  // Set optimization type
  type = as<std::string>(activ_param_["type"]);
  
}

// Method for creating optimizers
activation *activationFactory::createActivation () {
  Rcout << "\n Activation " << type;
  if      (type == "tanh")    return new tanhActivation();
  else                        return NULL;
}

