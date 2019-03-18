
// All test files should include the <testthat.h> header file.
#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_utils.h"

using namespace Rcpp;
using namespace arma;

// Include code to test
#include "utils.h"

// ---------------------------------------------------------------------------//
// ANN
// ---------------------------------------------------------------------------//

// ---------------------------------------------------------------------------//
// LAYER
// ---------------------------------------------------------------------------//

// ---------------------------------------------------------------------------//
// OPTIMIZERS
// ---------------------------------------------------------------------------//

// Tests for optimizers
context("OPTIMIZERS") {
  
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
    
    mat W = SGDTester.W;
    vec b = SGDTester.b;
    
    // Run tests
    expect_true( std::abs(SGDTester.rosenbrock_eval(W)) < abs_tol );
    expect_true( std::abs(SGDTester.rosenbrock_eval(b)) < abs_tol );
    double x = W(0,0), y = W(0,1);
    expect_true( std::abs(x-1) < abs_tol );
    expect_true( std::abs(y-1) < abs_tol );
    x = b(0), y = b(1);
    expect_true( std::abs(x-1) < abs_tol );
    expect_true( std::abs(y-1) < abs_tol );
  }
  
  // RMSprop Loss
  test_that("the RMSprop optimizer works correctly") {
    
    // Construct optimizer tester
    OptimizerTester RMSPropTester("rmsprop", abs_tol);
    
    for (int i; i < n_steps; i++) {
      RMSPropTester.step_W();
      RMSPropTester.step_b();
    }
    
    mat W = RMSPropTester.W;
    vec b = RMSPropTester.b;
    
    // Run tests
    expect_true( std::abs(RMSPropTester.rosenbrock_eval(W)) < abs_tol );
    expect_true( std::abs(RMSPropTester.rosenbrock_eval(b)) < abs_tol );
    double x = W(0,0), y = W(0,1);
    expect_true( std::abs(x-1) < abs_tol );
    expect_true( std::abs(y-1) < abs_tol );
    x = b(0), y = b(1);
    expect_true( std::abs(x-1) < abs_tol );
    expect_true( std::abs(y-1) < abs_tol );
  }
  
  // Adam optimizer
  test_that("the ADAM optimizer works correctly") {
    
    // Construct optimizer tester
    OptimizerTester AdamTester("adam", abs_tol);
    
    for (int i; i < n_steps; i++) {
      AdamTester.step_W();
      AdamTester.step_b();
    }
    
    mat W = AdamTester.W;
    vec b = AdamTester.b;
    
    // Run tests
    expect_true( std::abs(AdamTester.rosenbrock_eval(W)) < abs_tol );
    expect_true( std::abs(AdamTester.rosenbrock_eval(b)) < abs_tol );
    double x = W(0,0), y = W(0,1);
    expect_true( std::abs(x-1) < abs_tol );
    expect_true( std::abs(y-1) < abs_tol );
    x = b(0), y = b(1);
    expect_true( std::abs(x-1) < abs_tol );
    expect_true( std::abs(y-1) < abs_tol );
  }
}

// ---------------------------------------------------------------------------//
// LOSS FUNCTIONS
// ---------------------------------------------------------------------------//

// Tests for Loss functions
context("LOSS") {

  int n_rows = 32; // No of observations
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


// ---------------------------------------------------------------------------//
// UTILS
// ---------------------------------------------------------------------------//

// Tests for Scaler class
context("UTILS - Scaler class") {
  
  int n_rows = 32;
  int n_cols = 4;
  double rel_tol = 1e-4;
  double abs_tol = 1e-7;
  
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
    expect_false( approx_equal(sA, A, "both", abs_tol, rel_tol) );
    expect_true( approx_equal(uA, A, "both", abs_tol, rel_tol) );
    expect_true( approx_equal(mean(sA), zerovec, "absdiff", abs_tol) );
    expect_true( approx_equal(stddev(sA), onesvec, "absdiff", abs_tol) );
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
