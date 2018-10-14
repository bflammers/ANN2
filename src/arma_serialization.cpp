// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "arma_serialization.h"

using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Armadillo matrix serialization
// ---------------------------------------------------------------------------//
MatSerializer::MatSerializer () {}

MatSerializer::MatSerializer (mat X) : ncol(X.n_cols), nrow(X.n_rows) {
  X_holder.resize(ncol);
  for (size_t i = 0; i < ncol; ++i) {
    X_holder[i] = arma::conv_to< std::vector<double> >::from( X.col(i) );
  };
}

mat MatSerializer::getMat () {
  mat X(nrow, ncol);
  for (size_t i = 0; i < ncol; ++i) {
    X.col(i) = arma::conv_to< colvec >::from( X_holder[i] );
  };
  return X;
}

// ---------------------------------------------------------------------------//
// Armadillo vector serialization
// ---------------------------------------------------------------------------//

VecSerializer::VecSerializer () {}

VecSerializer::VecSerializer (vec X) {
  X_holder = arma::conv_to< std::vector<double> >::from( X );
}

vec VecSerializer::getVec () {
  vec X = arma::conv_to< vec >::from( X_holder );
  return X;
}

// ---------------------------------------------------------------------------//
// Armadillo row vector serialization
// ---------------------------------------------------------------------------//

RowVecSerializer::RowVecSerializer () {}

RowVecSerializer::RowVecSerializer (rowvec X) {
  X_holder = arma::conv_to< std::vector<double> >::from( X );
}

rowvec RowVecSerializer::getRowVec () {
  rowvec X = arma::conv_to< rowvec >::from( X_holder );
  return X;
}
