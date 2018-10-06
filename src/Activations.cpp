// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "Activations.h"
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Base Activation clas
// ---------------------------------------------------------------------------//

mat Activation::eval(mat X) { return X.ones(); }
mat Activation::grad(mat X) { return X.ones(); }

// ---------------------------------------------------------------------------//
// Activation classes
// ---------------------------------------------------------------------------//

class TanhActivation : public Activation
{
private:
  mat A;
public:
  // Constructor
  TanhActivation () {}
  
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

class SigmoidActivation : public Activation
{
private:
  mat A;
public:
  // Constructor
  SigmoidActivation () {}
  
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

class ReluActivation : public Activation
{
public:
  // Constructor
  ReluActivation () {}
  
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

class LinearActivation : public Activation
{
public:
  // Constructor
  LinearActivation () {}
  
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

class SoftMaxActivation : public Activation
{
private:
  mat A;
public:
  // Constructor
  SoftMaxActivation () { }
  
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

class StepActivation : public Activation
{
private:
  int H; 
  double k;
  vec seqH;
  mat::iterator mit;
public:
  // Constructor 
  StepActivation (List activ_param_)
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

class RampActivation : public Activation
{
public:
  // Constructor
  RampActivation () {}
  
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
// Methods for Activation factory 
// ---------------------------------------------------------------------------//

// Constructor
ActivationFactory::ActivationFactory (List activ_param_) 
  : activ_param(activ_param_)
{
  // Set optimization type
  type = as<std::string>(activ_param_["type"]);
}

// Method for creating optimizers
Activation *ActivationFactory::createActivation ()
{
  if      (type == "tanh")    return new TanhActivation();
  if      (type == "sigmoid") return new SigmoidActivation();
  else if (type == "relu")    return new ReluActivation();
  else if (type == "linear")  return new LinearActivation();
  else if (type == "softmax") return new SoftMaxActivation();
  else if (type == "ramp")    return new RampActivation();
  else if (type == "step")    return new StepActivation(activ_param);
  else                        return NULL;
}

