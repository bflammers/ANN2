#ifndef OPTIMIZER
#define OPTIMIZER

#include <RcppArmadillo.h>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/polymorphic.hpp>
#include <memory>

#include "arma_serialization.h"

// ---------------------------------------------------------------------------//
// Base optimizer class
// ---------------------------------------------------------------------------//

class Optimizer {
public:
  std::string type;
  virtual arma::mat updateW(arma::mat W, arma::mat dW) = 0;
  virtual arma::vec updateb(arma::vec b, arma::vec db) = 0;
};

// ---------------------------------------------------------------------------//
// Stochastic Gradient Descent optimizer
// ---------------------------------------------------------------------------//

class SGD : public Optimizer
{
private:
  double learn_rate, m, L1, L2;
  arma::mat mW;
  arma::vec mb;
public:
  SGD ();
  SGD (arma::mat W_templ_, arma::vec b_templ_, Rcpp::List optim_param_);
  arma::mat updateW(arma::mat W, arma::mat dW);
  arma::vec updateb(arma::vec b, arma::vec db);
  
  // Serialize
  template<class Archive>
  void save(Archive & archive) const
  {
    MatSerializer ser_mW(mW);
    VecSerializer ser_mb(mb);
    archive( ser_mW, ser_mb, learn_rate, m, L1, L2 );
  }
  
  // Deserialze
  template<class Archive>
  void load(Archive & archive)
  {
    MatSerializer ser_mW;
    VecSerializer ser_mb;
    archive( ser_mW, ser_mb, learn_rate, m, L1, L2 );
    mW = ser_mW.getMat();
    mb = ser_mb.getVec();
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(SGD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Optimizer, SGD);

// ---------------------------------------------------------------------------//
// Optimizer factory 
// ---------------------------------------------------------------------------//

std::unique_ptr<Optimizer> OptimizerFactory (arma::mat W_templ, arma::mat b_templ, Rcpp::List optim_param);


#endif