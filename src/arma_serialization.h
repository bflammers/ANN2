#ifndef ARMA_SERIALIZATION_H 
#define ARMA_SERIALIZATION_H

#include <RcppArmadillo.h>
#include <cereal/types/vector.hpp>

// ---------------------------------------------------------------------------//
// Armadillo matrix serialization
// ---------------------------------------------------------------------------//
class MatSerializer 
{
private:
  int ncol, nrow;
  std::vector< std::vector<double> > X_holder;
  
public:
  MatSerializer ();
  MatSerializer (arma::mat X);
  arma::mat getMat ();
  
  template<typename Archive>
  void serialize(Archive& ar) {
    ar( nrow , ncol , X_holder );
  }
};

// ---------------------------------------------------------------------------//
// Armadillo vector serialization
// ---------------------------------------------------------------------------//
class VecSerializer 
{
private:
  std::vector< double > X_holder;
  
public:
  VecSerializer ();
  VecSerializer (arma::vec X);
  arma::vec getVec ();
  
  template<typename Archive>
  void serialize(Archive& ar) {
    ar( X_holder );
  }
};

// ---------------------------------------------------------------------------//
// Armadillo row vector serialization
// ---------------------------------------------------------------------------//
class RowVecSerializer 
{
private:
  std::vector< double > X_holder;
  
public:
  RowVecSerializer ();
  RowVecSerializer (arma::rowvec X);
  arma::rowvec getRowVec ();
  
  template<typename Archive>
  void serialize(Archive& ar) {
    ar( X_holder );
  }
};

#endif
