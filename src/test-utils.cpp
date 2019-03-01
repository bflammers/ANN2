
// All test files should include the <testthat.h> header file.
#include <testthat.h>
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// Include code to test
#include "utils.h"

// Tests for Scaler class
context("Scaler") {
  
  test_that("the scaler works correctly when standardize == false") {
    mat A = randn<mat>(20,5);
    Scaler s(A, false);
    mat sA = s.scale(A);
    mat uA = s.unscale(sA);
    expect_true(all(vectorise(sA) == vectorise(A)));
    expect_true(all(vectorise(uA) == vectorise(A)));
    expect_true(sA.size() == A.size());
    expect_true(uA.size() == A.size());
  }
  
  test_that("the scaler works correctly when standardize == true") {
    mat A = randn<mat>(20,5);
    rowvec zerovec = zeros<rowvec>(5);
    rowvec onesvec = ones<rowvec>(5);
    Scaler s(A, true);
    mat sA = s.scale(A);
    mat uA = s.unscale(sA);
    expect_true(approx_equal(uA, A, "both", 0.00001, 0.001));
    expect_false(approx_equal(sA, A, "both", 0.00001, 0.001));
    expect_true(approx_equal(mean(sA), zerovec, "absdiff", 0.00001));
    expect_true(approx_equal(stddev(sA), onesvec, "absdiff", 0.00001));
    expect_true(sA.size() == A.size());
    expect_true(uA.size() == A.size());
  }

}
