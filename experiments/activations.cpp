// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "activations.h"
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

class tanhActivation : public activation
{
public:
  tanhActivation () {}
  mat eval(mat X) {
    return 1.725*tanh(2*X/3);
  }
  mat grad(mat X) { 
    return 1.15*(1-pow(tanh(2*X/3), 2));
  }
};

class reluActivation : public activation
{
public:
  reluActivation () {}
  mat eval(mat X) {
    return clamp(X, 0, X.max());
  }
  mat grad(mat X) { 
    mat d(size(X), fill::zeros);
    d.elem(find(X > 0)).fill(1);
    return d; 
  }
};

class linearActivation : public activation
{
public:
  linearActivation () {}
  mat eval(mat X) {
    return X;
  }
  mat grad(mat X) { 
    return X.fill(1);
  }
};

class softMaxActivation : public activation
{
public:
  mat A;
  softMaxActivation () {}
  
  mat eval(mat X) {
    rowvec max_X = max(X);
    X.each_row() -= max_X;
    A = exp(X);
    rowvec t = sum(A);
    A.each_row() /= t;
    return A;
  }
  
  mat grad(mat X) { 
    return A % (1 - A);
  }
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
  if      (type == "tanh")    return new tanhActivation();
  else if (type == "relu")    return new reluActivation();
  else if (type == "linear")  return new linearActivation();
  else if (type == "softmax") return new softMaxActivation();
  else                        return NULL;
}

