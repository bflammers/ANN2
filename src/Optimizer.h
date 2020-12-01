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
  virtual ~Optimizer () {};
  int n_train;
  std::string type;
  virtual arma::mat updateW(arma::mat W, arma::mat dW, int batch_size) = 0;
  virtual arma::vec updateb(arma::vec b, arma::vec db) = 0;
};

// ---------------------------------------------------------------------------//
// Stochastic Gradient Descent optimizer
// ---------------------------------------------------------------------------//

class SGD : public Optimizer
{
private:
  double learn_rate, L1, L2, momentum;
  arma::mat mW;
  arma::vec mb;
public:
  SGD ();
  SGD (arma::mat W_templ_, arma::vec b_templ_, Rcpp::List optim_param_);
  arma::mat updateW(arma::mat W, arma::mat dW, int batch_size);
  arma::vec updateb(arma::vec b, arma::vec db);
  
  // Serialize
  template<class Archive>
  void save(Archive & archive) const
  {
    MatSerializer ser_mW(mW);
    VecSerializer ser_mb(mb);
    archive( ser_mW, ser_mb, learn_rate, momentum, L1, L2 );
  }
  
  // Deserialze
  template<class Archive>
  void load(Archive & archive)
  {
    MatSerializer ser_mW;
    VecSerializer ser_mb;
    archive( ser_mW, ser_mb, learn_rate, momentum, L1, L2 );
    mW = ser_mW.getMat();
    mb = ser_mb.getVec();
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(SGD)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Optimizer, SGD)

// ---------------------------------------------------------------------------//
// RMSprop optimizer
// ---------------------------------------------------------------------------//

class RMSprop : public Optimizer
{
private:
  double learn_rate, L1, L2, decay, epsilon;
  arma::mat rmsW;
  arma::vec rmsb;
public:
  RMSprop ();
  RMSprop (arma::mat W_templ_, arma::vec b_templ_, Rcpp::List optim_param_);
  arma::mat updateW(arma::mat W, arma::mat dW, int batch_size);
  arma::vec updateb(arma::vec b, arma::vec db);
  
  // Serialize
  template<class Archive>
  void save(Archive & archive) const
  {
    MatSerializer ser_rmsW(rmsW);
    VecSerializer ser_rmsb(rmsb);
    archive( ser_rmsW, ser_rmsb, learn_rate, decay, epsilon, L1, L2 );
  }
  
  // Deserialze
  template<class Archive>
  void load(Archive & archive)
  {
    MatSerializer ser_rmsW;
    VecSerializer ser_rmsb;
    archive( ser_rmsW, ser_rmsb, learn_rate, decay, epsilon, L1, L2 );
    rmsW = ser_rmsW.getMat();
    rmsb = ser_rmsb.getVec();
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(RMSprop)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Optimizer, RMSprop)

// ---------------------------------------------------------------------------//
// Adam optimizer
// ---------------------------------------------------------------------------//

class Adam : public Optimizer
{
private:
  double learn_rate, L1, L2, beta1, beta2, epsilon;
  int tW, tb; // Number of updates counters
  arma::mat mW, vW;
  arma::vec mb, vb;
public:
  Adam ();
  Adam (arma::mat W_templ_, arma::vec b_templ_, Rcpp::List optim_param_);
  arma::mat updateW(arma::mat W, arma::mat dW, int batch_size);
  arma::vec updateb(arma::vec b, arma::vec db);
  
  // Serialize
  template<class Archive>
  void save(Archive & archive) const
  {
    MatSerializer ser_mW(mW), ser_vW(vW);
    VecSerializer ser_mb(mb), ser_vb(vb);
    archive( ser_mW, ser_vW, ser_mb, ser_vb, learn_rate, beta1, beta2, epsilon, 
             L1, L2 );
  }
  
  // Deserialze
  template<class Archive>
  void load(Archive & archive)
  {
    MatSerializer ser_mW(mW), ser_vW(vW);
    VecSerializer ser_mb(mb), ser_vb(vb);
    archive( ser_mW, ser_vW, ser_mb, ser_vb, learn_rate, beta1, beta2, epsilon, 
             L1, L2 );
    mW = ser_mW.getMat();
    vW = ser_vW.getMat();
    mb = ser_mb.getVec();
    vb = ser_vb.getVec();
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(Adam)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Optimizer, Adam)

// ---------------------------------------------------------------------------//
// Optimizer factory 
// ---------------------------------------------------------------------------//

std::unique_ptr<Optimizer> OptimizerFactory (arma::mat W_templ, 
                                             arma::mat b_templ, 
                                             Rcpp::List optim_param);


#endif
