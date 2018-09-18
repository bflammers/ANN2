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
private:
  mat A;
public:
  // Constructor
  tanhActivation () {}
  
  // Evaluate tanh
  mat eval(mat X) 
  {
    A = tanh( 2 * X / 3 );
    return 1.725 * A;
  }
  
  // Derivative tanh
  mat grad(mat X) 
  {
    // Reuse A from .eval()
    return 1.15 * ( 1 - pow( A, 2 ) );
  }
};

class sigmoidActivation : public activation
{
private:
  mat A;
public:
  // Constructor
  sigmoidActivation () {}
  
  // Evaluate sigmoid
  mat eval(mat X) 
  {
    A = 1/(1+exp(-X));
    return A;
  }
  
  // Derivative sigmoid
  mat grad(mat X)
  {
    // Reuse A from .eval()
    return A % (1 - A);
  }
};

class reluActivation : public activation
{
public:
  // Constructor
  reluActivation () {}
  
  // Evaluate relu
  mat eval(mat X) 
  {
    return clamp(X, 0, std::numeric_limits<double>::max());
  }
  
  // Derivative relu
  mat grad(mat X) 
  { 
    mat dA(size(X), fill::zeros);
    dA.elem(find(X > 0)).fill(1);
    return dA; 
  }
};

class linearActivation : public activation
{
public:
  // Constructor
  linearActivation () {}
  
  // Evaluate
  mat eval(mat X)
  {
    return X;
  }
  
  // Derivative
  mat grad(mat X) 
  { 
    return X.fill(1);
  }
};

class softMaxActivation : public activation
{
private:
  mat A;
public:
  // Constructor
  softMaxActivation () { }
  
  // Evaluate softmax
  mat eval(mat X) 
  {
    rowvec max_X = max(X, 0);
    X.each_row() -= max_X;
    A = exp(X);
    rowvec t = sum(A, 0);
    A.each_row() /= t;
    return A;
  }
  
  // Derivative softmax
  mat grad(mat X) 
  { 
    // reuse A from .eval()
    return A % (1 - A);
  }
};

class stepActivation : public activation
{
private:
  int H; 
  double k;
  vec seqH;
  mat::iterator mit;
public:
  // Constructor 
  stepActivation (List activ_param_)
    : H(activ_param_["H"]), 
      k(activ_param_["k"]) 
  {
    seqH = linspace(1, (H - 1), (H - 1))/ H;
  }
  
  // Evaluate stepfunction
  mat eval(mat X) 
  {
    mat A = X;
    for(mit = A.begin(); mit!=A.end(); mit++){
      (*mit) = sum( tanh( k * ((*mit) - seqH) ) );
    }
    return 0.5 + A / ( 2 * (H - 1) );
  }
  
  // Derivative stepfunction
  mat grad(mat X) 
  {
    mat gA = X;
    for(mit = gA.begin(); mit!=gA.end(); mit++){
      (*mit) = sum(1 - pow( tanh(k * ((*mit) - seqH)), 2) );
    }
    return k * gA / ( 2 * (H-1) );
  }
  
};

class rampActivation : public activation
{
public:
  // Constructor
  rampActivation () {}
  
  // Evaluate ramp
  mat eval(mat X) 
  {
    mat A = X;
    return clamp(A, 0, 1);
  }
  
  // Derivative ramp
  mat grad(mat X) 
  { 
    mat A(size(X), fill::zeros);
    A.elem(find(0<X && X<1)).fill(1);
    return A;
  }
};


// ---------------------------------------------------------------------------//
// Methods for activation factory 
// ---------------------------------------------------------------------------//

// Constructor
activationFactory::activationFactory (List activ_param_) 
  : activ_param(activ_param_)
{
  // Set optimization type
  type = as<std::string>(activ_param_["type"]);
}

// Method for creating optimizers
activation *activationFactory::createActivation ()
{
  if      (type == "tanh")    return new tanhActivation();
  if      (type == "sigmoid") return new sigmoidActivation();
  else if (type == "relu")    return new reluActivation();
  else if (type == "linear")  return new linearActivation();
  else if (type == "softmax") return new softMaxActivation();
  else if (type == "ramp")    return new rampActivation();
  else if (type == "step")    return new stepActivation(activ_param);
  else                        return NULL;
}

