
// All test files should include the <testthat.h> header file.
#include <testthat.h>
#include <RcppArmadillo.h>

#define ARMA_USE_WRAPPER

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

int n_rows = 32;
int n_cols = 4;

rowvec zerovec = zeros<rowvec>(n_cols);
rowvec onesvec = ones<rowvec>(n_cols);

// Tests for Scaler class
context("Scaler class") {
  
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
