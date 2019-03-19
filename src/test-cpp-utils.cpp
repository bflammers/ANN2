
// All test files should include the <testthat.h> header file.
#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_utils.h"

using namespace Rcpp;
using namespace arma;

// Include code to test
#include "utils.h"


// ---------------------------------------------------------------------------//
// UTILS
// ---------------------------------------------------------------------------//

// Tests for Scaler class
context("Tests for utils: Scaler class") {
  
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
context("Tests for utils: Sampler class") {
  
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
