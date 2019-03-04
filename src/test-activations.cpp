
// All test files should include the <testthat.h> header file.
#include <testthat.h>
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// Include code to test
#include "Activations.h"

// This is needed for random number generation since Armadillo's RNG does not 
// seem to work nicely with testthat. Not sure what is going wrong. The first 
// time the test runs, the number are generated as expected by randu or randn
// but the second time, it generates all values close to zero
std::mt19937 engine2;  // Mersenne twister random number engine
std::uniform_real_distribution<double> unif_distr(-4.0, 4.0);

mat gradient_check (mat X, std::string activ_type) {
  
  List activ_param = List::create(Named("type") = activ_type);
  std::unique_ptr<Activation> g = ActivationFactory(activ_param);
  
  mat num_gradient = (g->eval(X + 1e-5) - g->eval(X - 1e-5)) / 2e-5;
  mat ana_gradient = g->grad(X);
  
  return abs(ana_gradient - num_gradient); // Scale by elementwise max
}

// Tests for Scaler class
context("Tanh activation function") {
  
  int n_rows = 32;
  int n_cols = 4;
  
  mat A(n_rows, n_cols);
  A.imbue( [&]() { return unif_distr(engine2); } );
  
  test_that("the tanh works correctly") {
    expect_true( all(vectorise(gradient_check(A, "tanh")) < 1e-4) );
  }
  
}
