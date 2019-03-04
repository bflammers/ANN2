
// All test files should include the <testthat.h> header file.
#include <testthat.h>
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// Include code to test
#include "utils.h"

// This is needed for random number generation since Armadillo's RNG does not 
// seem to work nicely with testthat. Not sure what is going wrong. The first 
// time the test runs, the number are generated as expected by randu or randn
// but the second time, it generates all values close to zero
std::mt19937 engine;  // Mersenne twister random number engine
std::normal_distribution<double> distr(0.0, 1.0);

// Tests for Scaler class
context("Scaler class") {
  
  int n_rows = 32;
  int n_cols = 4;
  
  rowvec zerovec = zeros<rowvec>(n_cols);
  rowvec onesvec = ones<rowvec>(n_cols);
  
  mat A(n_rows, n_cols);
  A.imbue( [&]() { return distr(engine); } );
  
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
context("Sampler class") {
  
  // Keep at these values! 
  int n_rows = 32;
  int n_cols = 4;
  double val_prop_non_zero = 0.5;
  int batch_size = 24;
  
  mat X(n_rows, n_cols);
  X.imbue( [&]() { return distr(engine); } );
  
  mat y(n_rows, n_cols);
  y.imbue( [&]() { return distr(engine); } );
  
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
