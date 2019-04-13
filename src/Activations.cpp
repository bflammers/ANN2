
#include <RcppArmadillo.h>
#include "Activations.h"

using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Tanh activation class
// ---------------------------------------------------------------------------//

// Constructor
TanhActivation::TanhActivation () { type = "tanh"; }

// Evaluate tanh
mat TanhActivation::eval(mat X) 
{
  A = tanh( 2 * X / 3 );
  return 1.725 * A;
}

// Derivative tanh
mat TanhActivation::grad(mat X) 
{
  // Reuse A from .eval()
  return 1.15 * ( 1 - pow( A, 2 ) );
}

// ---------------------------------------------------------------------------//
// Sigmoid activation class
// ---------------------------------------------------------------------------//

// Constructor
SigmoidActivation::SigmoidActivation () { type = "sigmoid"; }

// Evaluate sigmoid
mat SigmoidActivation::SigmoidActivation::eval(mat X) 
{
  A = 1/(1+exp(-X));
  return A;
}

// Derivative sigmoid
mat SigmoidActivation::SigmoidActivation::grad(mat X)
{
  // Reuse A from .eval()
  return A % (1 - A);
}

// ---------------------------------------------------------------------------//
// Rectifier activation class
// ---------------------------------------------------------------------------//

// Constructor
ReluActivation::ReluActivation () { type = "relu"; }

// Evaluate relu
mat ReluActivation::eval(mat X) 
{
  return clamp(X, 0, std::numeric_limits<double>::max());
}

// Derivative relu
mat ReluActivation::grad(mat X) 
{ 
  mat dA(size(X), fill::zeros);
  dA.elem(find(X > 0)).fill(1);
  return dA; 
}

// ---------------------------------------------------------------------------//
// Linear activation class
// ---------------------------------------------------------------------------//

// Constructor
LinearActivation::LinearActivation () { type = "linear"; }

// Evaluate
mat LinearActivation::eval(mat X)
{
  return X;
}

// Derivative
mat LinearActivation::grad(mat X) 
{ 
  return X.fill(1);
}

// ---------------------------------------------------------------------------//
// Softmax activation class
// ---------------------------------------------------------------------------//

// Constructor
SoftMaxActivation::SoftMaxActivation () { type = "softmax"; }

// Evaluate softmax
mat SoftMaxActivation::eval(mat X) 
{
  rowvec max_X = max(X, 0);
  X.each_row() -= max_X;
  A = exp(X);
  rowvec t = sum(A, 0);
  A.each_row() /= t;
  return A;
}

// Derivative softmax
mat SoftMaxActivation::grad(mat X) 
{ 
  // This is not the elementwise derivative of the eval() method, as is the case
  // for the other activation functions. However, this leads to a correct 
  // gradient when used with log loss (and only with log loss)!! This is because
  // the derivative of the loss function wrt the inputs to the softmax is 
  // completely implemented in the Log loss class. 
  // Therefore, the softmax activation is only to be used with log loss!! 
  // Error should be thrown if this is not the case.
  // See https://peterroelants.github.io/posts/cross-entropy-softmax/ for info
  return X.ones(); 
}

// ---------------------------------------------------------------------------//
// Step function activation class
// ---------------------------------------------------------------------------//

// Default constructor needed for serlialization
StepActivation::StepActivation () { type = "step"; }

// Constructor 
StepActivation::StepActivation (List activ_param_)
  : H ( activ_param_["step_H"] ), 
    k ( activ_param_["step_k"] ) {
  type = "step"; 
  seqH = linspace(1, (H - 1), (H - 1))/ H;
}

// Evaluate stepfunction
mat StepActivation::eval(mat X) 
{
  mat A = X;
  for(mit = A.begin(); mit!=A.end(); mit++){
    (*mit) = sum( tanh( k * ((*mit) - seqH) ) );
  }
  return 0.5 + A / ( 2 * (H - 1) );
}

// Derivative stepfunction
mat StepActivation::grad(mat X) 
{
  mat gA = X;
  for(mit = gA.begin(); mit!=gA.end(); mit++){
    (*mit) = sum(1 - pow( tanh(k * ((*mit) - seqH)), 2) );
  }
  return k * gA / ( 2 * (H-1) );
}

// ---------------------------------------------------------------------------//
// Ramp function activation class
// ---------------------------------------------------------------------------//

// Constructor
RampActivation::RampActivation () { type = "ramp"; }

// Evaluate ramp
mat RampActivation::eval(mat X) 
{
  mat A = X;
  return clamp(A, 0, 1);
}

// Derivative ramp
mat RampActivation::grad(mat X) 
{ 
  mat A(size(X), fill::zeros);
  A.elem(find(0<X && X<1)).fill(1);
  return A;
}

// ---------------------------------------------------------------------------//
// Activation factory
// ---------------------------------------------------------------------------//

std::unique_ptr<Activation> ActivationFactory (List activ_param) {
  std::string type = as<std::string>(activ_param["type"]);
  if      (type == "tanh")    return std::unique_ptr<Activation>(new TanhActivation());
  else if (type == "sigmoid") return std::unique_ptr<Activation>(new SigmoidActivation());
  else if (type == "relu")    return std::unique_ptr<Activation>(new ReluActivation());
  else if (type == "linear")  return std::unique_ptr<Activation>(new LinearActivation());
  else if (type == "softmax") return std::unique_ptr<Activation>(new SoftMaxActivation());
  else if (type == "ramp")    return std::unique_ptr<Activation>(new RampActivation());
  else if (type == "step")    return std::unique_ptr<Activation>(new StepActivation(activ_param));
  else stop("activ.type not implemented");
}

