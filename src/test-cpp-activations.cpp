
// All test files should include the <testthat.h> header file.
#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_utils.h"

using namespace Rcpp;
using namespace arma;


// ---------------------------------------------------------------------------//
// ACTIVATION FUNCTIONS
// ---------------------------------------------------------------------------//

// Tests for Activation functions
context("Tests for activation functions") {
  
  int n_rows = 4;  // No of classes since t(X) is propagated through network
  int n_cols = 32; // No of observations
  double rel_tol_kinks = 1e-4;
  double rel_tol_smooth = 1e-7;
  double abs_tol = 1e-7;
  
  rowvec onesvec = ones<rowvec>(n_cols);
  
  // TANH
  test_that("the tanh works correctly") {
    
    // Construct activation tester and matrix with random numbers
    ActivationTester TanhTester("tanh", rel_tol_smooth, abs_tol);
    mat A = RNG_uniform(n_rows, n_cols, -2.0, 2.0);
    
    // Run tests
    expect_true( TanhTester.grad_check(A) );
    expect_true( TanhTester.eval_check(0, 0) );
    expect_true( TanhTester.eval_check(1e10, 1.725) );
    expect_true( TanhTester.eval_check(-1e10, -1.725) );
  }
  
  // SIGMOID
  test_that("the sigmoid works correctly") {
    
    // Construct activation tester and matrix with random numbers
    ActivationTester SigmoidTester("sigmoid", rel_tol_smooth, abs_tol);
    mat A = RNG_uniform(n_rows, n_cols, -1.0, 3.0);
    
    // Run tests
    expect_true( SigmoidTester.grad_check(A) );
    expect_true( all(vectorise(SigmoidTester.g->eval(A) > 0)) );
    expect_true( SigmoidTester.eval_check(0, 0.5) );
    expect_true( SigmoidTester.eval_check(1e10, 1) );
    expect_true( SigmoidTester.eval_check(-1e10, 0) );
  }
  
  test_that("the relu works correctly") {

    // Construct activation tester and matrix with random numbers
    ActivationTester ReluTester("relu", rel_tol_kinks, abs_tol);
    mat A = RNG_uniform(n_rows, n_cols, -1.0, 3.0);

    // Run tests
    expect_true( ReluTester.grad_check(A) );
    expect_true( ReluTester.eval_check(0, 0) );
    expect_true( ReluTester.eval_check(-1, 0) );
    expect_true( ReluTester.eval_check(1, 1) );
  }

  test_that("the linear activation function works correctly") {

    // Construct activation tester and matrix with random numbers
    ActivationTester LinearTester("linear", rel_tol_smooth, abs_tol);
    mat A = RNG_uniform(n_rows, n_cols, -2.0, 2.0);

    // Run tests
    expect_true( LinearTester.grad_check(A) );
    expect_true( LinearTester.eval_check(0, 0) );
    expect_true( LinearTester.eval_check(1, 1) );
    expect_true( LinearTester.eval_check(-1, -1) );
  }

  test_that("the softmax works correctly") {

    // Construct activation tester and matrix with random numbers
    ActivationTester SoftmaxTester("softmax", rel_tol_smooth, abs_tol);
    mat A = RNG_uniform(n_rows, n_cols, -2.0, 2.0);

    // Run tests
    // No gradient check for Softmax because it is implemented only for use with 
    // the log loss function. The implementation of grad() is not the 
    // elementwise derivative of the eval() method
    rowvec row_sums = sum(SoftmaxTester.g->eval(A), 0);
    expect_true( approx_equal(row_sums, onesvec, "absdiff", abs_tol) );
    expect_true( SoftmaxTester.eval_check(10, 1) );
    expect_true( SoftmaxTester.eval_check(-10, 1) );
  }
  
  test_that("the ramp activation function works correctly") {

    // Construct activation tester and matrix with random numbers
    ActivationTester RampTester("ramp", rel_tol_kinks, abs_tol);
    mat A = RNG_uniform(n_rows, n_cols, -2.0, 2.0);

    // Run tests
    expect_true( RampTester.grad_check(A) );
    expect_true( RampTester.eval_check(0, 0) );
    expect_true( RampTester.eval_check(-10, 0) );
    expect_true( RampTester.eval_check(0.5, 0.5) );
    expect_true( RampTester.eval_check(10, 1) );
  }

  test_that("the step activation function works correctly") {

    // Construct activation tester and matrix with random numbers
    ActivationTester StepTester("step", rel_tol_kinks, abs_tol);
    mat A = RNG_uniform(n_rows, n_cols, 0.1, 0.9); // Domain with positive grad

    // Run tests
    expect_true( StepTester.grad_check(A) );
    expect_true( StepTester.eval_check(-10, 0) );
    expect_true( StepTester.eval_check(10, 1) );
  }

}
