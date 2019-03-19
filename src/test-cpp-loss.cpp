
// All test files should include the <testthat.h> header file.
#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_utils.h"

using namespace Rcpp;
using namespace arma;


// ---------------------------------------------------------------------------//
// LOSS FUNCTIONS
// ---------------------------------------------------------------------------//

// Tests for Loss functions
context("Tests for loss functions") {

  int n_rows = 8; // No of observations
  int n_cols = 4;  // No of classes
  double rel_tol_kinks = 1e-4;
  double rel_tol_smooth = 1e-7;
  double abs_tol = 1e-7;

  // LOG Loss
  test_that("the log loss works correctly") {

    // Construct activation tester and matrix with random numbers
    LossTester LogTester("log", rel_tol_smooth, abs_tol);

    // Run tests
    // No gradient check for log loss function because it is implemented only 
    // for use with the Softmax function. The implementation of grad() is not 
    // the elementwise derivative of the eval() method
    expect_true( LogTester.eval_check(1, 1, 0) );
    expect_true( LogTester.eval_check(0, 1, 0) );
    expect_true( LogTester.eval_check(1, .5, 0.6931472) );
  }
  
  // SQUARED Loss
  test_that("the squared loss works correctly") {
    
    // Construct activation tester and matrix with random numbers
    LossTester SquaredTester("squared", rel_tol_smooth, abs_tol);
    mat y = RNG_uniform(n_rows, n_cols, -1.0, 1.0);
    mat y_fit = RNG_uniform(n_rows, n_cols, -1.0, 1.0);
    
    // Run tests
    expect_true( SquaredTester.grad_check(y, y_fit) );
    expect_true( SquaredTester.eval_check(1, 1, 0) );
  }
  
  // ABSOLUTE Loss
  test_that("the absolute loss works correctly") {
    
    // Construct activation tester and matrix with random numbers
    LossTester AbsoluteTester("absolute", rel_tol_smooth, abs_tol);
    mat y = RNG_uniform(n_rows, n_cols, -1.0, 1.0);
    mat y_fit = RNG_uniform(n_rows, n_cols, -1.0, 1.0);
    
    // Run tests
    expect_true( AbsoluteTester.grad_check(y, y_fit) );
    expect_true( AbsoluteTester.eval_check(1, 1, 0) );
  }
  
  // HUBER Loss
  test_that("the huber loss works correctly") {
    
    // Construct activation tester and matrix with random numbers
    LossTester HuberTester("huber", rel_tol_smooth, abs_tol);
    mat y = RNG_uniform(n_rows, n_cols, -1.0, 1.0);
    mat y_fit = RNG_uniform(n_rows, n_cols, -1.0, 1.0);
    
    // Run tests
    expect_true( HuberTester.grad_check(y, y_fit) );
    expect_true( HuberTester.eval_check(1, 1, 0) );
  }
  
  // PSEUDO-HUBER Loss
  test_that("the pseudo-huber loss works correctly") {
    
    // Construct activation tester and matrix with random numbers
    LossTester PseudoHuberTester("pseudo-huber", rel_tol_smooth, abs_tol);
    mat y = RNG_uniform(n_rows, n_cols, -1.0, 1.0);
    mat y_fit = RNG_uniform(n_rows, n_cols, -1.0, 1.0);
    
    // Run tests
    expect_true( PseudoHuberTester.grad_check(y, y_fit) );
    expect_true( PseudoHuberTester.eval_check(1, 1, 0) );
  }
}
