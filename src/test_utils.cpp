
#include <RcppArmadillo.h>
#include "test_utils.h"

using namespace Rcpp;
using namespace arma;


// ---------------------------------------------------------------------------//
// ACTIVATION FUNCTIONS TESTING
// ---------------------------------------------------------------------------//

ActivationTester::ActivationTester (std::string activ_type, double rel_tol_, 
                                    double abs_tol_) 
  : abs_tol(abs_tol_), rel_tol(rel_tol_) {
  
  List activ_param = List::create(Named("type") = activ_type, 
                                  Named("step_H") = 5, 
                                  Named("step_k") = 40);
  g = ActivationFactory(activ_param);
}

// Gradient checking
// See: http://cs231n.github.io/neural-networks-3/#gradcheck
bool ActivationTester::gradient_check (mat X) {
  
  mat num_gradient = (g->eval(X + 1e-5) - g->eval(X - 1e-5)) / 2e-5;
  mat _ = g->eval(X); // Needed because grad() reuses A from eval()
  mat ana_gradient = g->grad(X);
  
  return approx_equal(num_gradient, ana_gradient, "reldiff", rel_tol);
}

// Eval activation function: input, output check
bool ActivationTester::eval_check (double in_value, double out_value) {
  
  mat A(1,1); A.fill(in_value);
  mat B = g->eval(A);
  mat C(1,1); C.fill(out_value);
  
  return approx_equal(B, C, "both", abs_tol, rel_tol) ;
}
