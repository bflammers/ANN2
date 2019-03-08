
// All test files should include the <testthat.h> header file.
#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_utils.h"

using namespace Rcpp;
using namespace arma;

// Include code to test
#include "utils.h"
#include "Activations.h"

// ---------------------------------------------------------------------------//
// ANN
// ---------------------------------------------------------------------------//

// ---------------------------------------------------------------------------//
// LAYER
// ---------------------------------------------------------------------------//

// ---------------------------------------------------------------------------//
// OPTIMIZERS
// ---------------------------------------------------------------------------//

// ---------------------------------------------------------------------------//
// LOSS FUNCTIONS
// ---------------------------------------------------------------------------//

// // Tests for Loss functions
// context("LOSS") {
//   
//   int n_rows = 4;  // No of classes since t(X) is propagated through network
//   int n_cols = 32; // No of observations
//   double rel_tol_kinks = 1e-4;
//   double rel_tol_smooth = 1e-7;
//   double abs_tol = 1e-7;
//   
//   // LOG Loss
//   test_that("the log loss works correctly") {
//     
//     // Construct activation tester and matrix with random numbers
//     ActivationTester LogTester("tanh", rel_tol_smooth, abs_tol);
//     mat A = RNG_uniform(n_rows, n_cols, -2.0, 2.0);
//     
//     // Run tests
//     expect_true( TanhTester.grad_check(A) );
//     expect_true( TanhTester.eval_check(0, 0) );
//     expect_true( TanhTester.eval_check(1e10, 1.725) );
//     expect_true( TanhTester.eval_check(-1e10, -1.725) );
//   }
// }

// ---------------------------------------------------------------------------//
// ACTIVATION FUNCTIONS
// ---------------------------------------------------------------------------//

// Tests for Activation functions
context("ACTIVATIONS") {
  
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
    expect_true( SoftmaxTester.grad_check(A, true) );
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


// ---------------------------------------------------------------------------//
// UTILS
// ---------------------------------------------------------------------------//

// Tests for Scaler class
context("UTILS - Scaler class") {
  
  int n_rows = 32;
  int n_cols = 4;
  
  rowvec zerovec = zeros<rowvec>(n_cols);
  rowvec onesvec = ones<rowvec>(n_cols);
  
  mat A = RNG_gaussian(n_rows, n_cols);
  
  test_that("the scaler works correctly when standardize == false") {
    Scaler s(A, false);
    mat sA = s.scale(A);
    mat uA = s.unscale(sA);
    expect_true( all(vectorise(sA) == vectorise(A)) );
    expect_true( all(vectorise(uA) == vectorise(A)) );
    expect_true( sA.size() == A.size() );
    expect_true( uA.size() == A.size() );
  }
  
  test_that("the scaler works correctly when standardize == true") {
    Scaler s(A, true);
    mat sA = s.scale(A);
    mat uA = s.unscale(sA);
    expect_false( approx_equal(sA, A, "both", 0.00001, 0.001) );
    expect_true( approx_equal(uA, A, "both", 0.00001, 0.001) );
    expect_true( approx_equal(mean(sA), zerovec, "absdiff", 0.00001) );
    expect_true( approx_equal(stddev(sA), onesvec, "absdiff", 0.00001) );
    expect_true( sA.size() == A.size() );
    expect_true( uA.size() == A.size() );
  }
  
}

// Tests for Sampler class
context("UTILS - Sampler class") {
  
  // Keep at these values! 
  int n_rows = 32;
  int n_cols = 4;
  double val_prop_non_zero = 0.5;
  int batch_size = 24;
  
  mat X = RNG_gaussian(n_rows, n_cols);
  mat y = RNG_gaussian(n_rows, n_cols);
  
  test_that("the samplers validation set logic works with val_prop == 0") {
    // Training parameters and construct sampler
    List train_param = List::create(Named("batch_size") = batch_size, 
                                    Named("val_prop") = 0, 
                                    Named("drop_last") = true);
    Sampler s(X, y, train_param);
    // Validation set logic
    mat val_X = s.get_Xv();
    mat val_y = s.get_yv();
    expect_false( s.validate );
    expect_true( val_X.n_rows == 0 );
    expect_true( val_X.n_cols == 0 );
    expect_true( val_y.n_rows == 0 );
    expect_true( val_y.n_cols == 0 );
    
  }
  
  test_that("the samplers validation set logic works with val_prop > 0") {
    // Training parameters and construct sampler
    List train_param = List::create(Named("batch_size") = batch_size, 
                                    Named("val_prop") = val_prop_non_zero, 
                                    Named("drop_last") = true);
    Sampler s(X, y, train_param);
    // Validation set logic
    mat val_X = s.get_Xv();
    mat val_y = s.get_yv();
    expect_true( s.validate );
    expect_true( val_X.n_rows == n_rows * val_prop_non_zero );
    expect_true( val_X.n_cols == n_cols );
    expect_true( val_y.n_rows == n_rows * val_prop_non_zero );
    expect_true( val_y.n_cols == n_cols );
    
  }
  
  test_that("the sampler works with drop_last == true") {
    
    // Training parameters and construct sampler
    List train_param = List::create(Named("batch_size") = batch_size, 
                                    Named("val_prop") = 0, 
                                    Named("drop_last") = true);
    Sampler s(X, y, train_param);
    
    // Number of batches
    expect_true( s.n_batch == 1 );
    
    // First batch
    mat batch_X = s.next_Xb();
    mat batch_y = s.next_yb();
    expect_true( batch_X.size() == batch_y.size() );
    expect_true( batch_X.n_rows == batch_size );
    expect_true( batch_X.n_cols == n_cols );
    
  }
  
  test_that("the sampler works correctly with drop_last == false") {
    
    // Training parameters and construct sampler
    List train_param = List::create(Named("batch_size") = batch_size, 
                                    Named("val_prop") = 0,
                                    Named("drop_last") = false);
    Sampler s(X, y, train_param);
    
    // Number of batches
    expect_true( s.n_batch == 2 );
    
    // First batch
    mat batch_X = s.next_Xb();
    mat batch_y = s.next_yb();
    expect_true( batch_X.size() == batch_y.size() );
    expect_true( batch_X.n_rows == batch_size );
    expect_true( batch_X.n_cols == n_cols );
    
    // Second batch
    batch_X = s.next_Xb();
    batch_y = s.next_yb();
    expect_true( batch_X.size() == batch_y.size() );
    expect_true( batch_X.n_rows == n_rows - batch_size );
    expect_true( batch_X.n_cols == n_cols );
  }
  
}
