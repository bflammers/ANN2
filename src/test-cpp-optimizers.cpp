
// All test files should include the <testthat.h> header file.
#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_utils.h"

using namespace Rcpp;
using namespace arma;


// ---------------------------------------------------------------------------//
// OPTIMIZERS
// ---------------------------------------------------------------------------//

// Tests for optimizers
context("Tests for optimizers") {
  
  double abs_tol = 1e-3;
  int n_steps = 1e5;
  
  // SGD optimizer
  test_that("the SGD optimizer works correctly") {
    
    // Construct optimizer tester
    OptimizerTester SGDTester("sgd", abs_tol);
    
    for (int i; i < n_steps; i++) {
      SGDTester.step_W();
      SGDTester.step_b();
    }
    
    // Get x, y values from OptimizerTester object
    mat W = SGDTester.W;
    double W_x = W(0,0);
    double W_y = W(0,1);
    vec b = SGDTester.b;
    double b_x = b(0);
    double b_y = b(1);
    
    // Run tests
    expect_true( std::abs(SGDTester.rosenbrock_eval(W)) < abs_tol );
    expect_true( std::abs(SGDTester.rosenbrock_eval(b)) < abs_tol );
    expect_true( std::abs(W_x-1) < abs_tol );
    expect_true( std::abs(W_y-1) < abs_tol );
    expect_true( std::abs(b_x-1) < abs_tol );
    expect_true( std::abs(b_y-1) < abs_tol );
  }
  
  // RMSprop Loss
  test_that("the RMSprop optimizer works correctly") {
    
    // Construct optimizer tester
    OptimizerTester RMSPropTester("rmsprop", abs_tol);
    
    for (int i; i < n_steps; i++) {
      RMSPropTester.step_W();
      RMSPropTester.step_b();
    }
    
    // Get x, y values from OptimizerTester object
    mat W = RMSPropTester.W;
    double W_x = W(0,0);
    double W_y = W(0,1);
    vec b = RMSPropTester.b;
    double b_x = b(0);
    double b_y = b(1);
    
    // Run tests
    expect_true( std::abs(RMSPropTester.rosenbrock_eval(W)) < abs_tol );
    expect_true( std::abs(RMSPropTester.rosenbrock_eval(b)) < abs_tol );
    expect_true( std::abs(W_x-1) < abs_tol );
    expect_true( std::abs(W_y-1) < abs_tol );
    expect_true( std::abs(b_x-1) < abs_tol );
    expect_true( std::abs(b_y-1) < abs_tol );
  }
  
  // Adam optimizer
  test_that("the ADAM optimizer works correctly") {
    
    // Construct optimizer tester
    OptimizerTester AdamTester("adam", abs_tol);
    
    for (int i; i < n_steps; i++) {
      AdamTester.step_W();
      AdamTester.step_b();
    }
    
    // Get x, y values from OptimizerTester object
    mat W = AdamTester.W;
    double W_x = W(0,0);
    double W_y = W(0,1);
    vec b = AdamTester.b;
    double b_x = b(0);
    double b_y = b(1);
    
    // Run tests
    expect_true( std::abs(AdamTester.rosenbrock_eval(W)) < abs_tol );
    expect_true( std::abs(AdamTester.rosenbrock_eval(b)) < abs_tol );
    expect_true( std::abs(W_x-1) < abs_tol );
    expect_true( std::abs(W_y-1) < abs_tol );
    expect_true( std::abs(b_x-1) < abs_tol );
    expect_true( std::abs(b_y-1) < abs_tol );
  }
}
