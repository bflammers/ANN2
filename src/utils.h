#ifndef UTILS_H 
#define UTILS_H

#include <RcppArmadillo.h>
#include <cereal/archives/portable_binary.hpp>
#include "arma_serialization.h"

// ---------------------------------------------------------------------------//
// Scaler class
// ---------------------------------------------------------------------------//
class Scaler 
{
private:
  arma::rowvec z_mu, z_sd;
  bool standardize;
  
public:
  int n_col;
  Scaler(); // Default constructor needed for serialization
  Scaler (arma::mat z, bool standardize_);
  arma::mat scale(arma::mat z);
  arma::mat unscale(arma::mat z);
  
  template<class Archive>
  void save(Archive & archive) const
  {
    RowVecSerializer ser_z_mu(z_mu), ser_z_sd(z_sd);
    archive( ser_z_mu, ser_z_sd, standardize ); 
  }
  
  template<class Archive>
  void load(Archive & archive)
  {
    RowVecSerializer ser_z_mu, ser_z_sd;
    archive( ser_z_mu, ser_z_sd, standardize ); 
    z_mu = ser_z_mu.getRowVec();
    z_sd = ser_z_sd.getRowVec();
  }
};

// ---------------------------------------------------------------------------//
// Sampler class
// ---------------------------------------------------------------------------//
class Sampler 
{
private:
  arma::mat X_train, y_train, X_val, y_val;
  std::list<arma::uvec> indices;
  std::list<arma::uvec>::iterator X_it;
  std::list<arma::uvec>::iterator y_it;
public:
  int n_batch, n_train;
  bool validate;
  Sampler (arma::mat X, arma::mat y, Rcpp::List train_param);
  void shuffle();
  arma::mat next_Xb();
  arma::mat next_yb();
  arma::mat get_Xv();
  arma::mat get_yv();
};

// ---------------------------------------------------------------------------//
// Tracker class
// ---------------------------------------------------------------------------//
class Tracker {
private:
  int k, curr_progress;
  double one_percent;
  bool verbose;
  std::string progressBar(int progress);
  
public:
  bool validate;
  int n_passes;
  
  Tracker(); // Default constructor needed for serialization
  Tracker(bool verbose_);
  arma::mat train_history;
  void setTracker(int n_passes_, bool validate_, Rcpp::List train_param_);
  void track (int epoch, double train_loss, double val_loss);
  void endLine ();
  
  // Serialize
  template<class Archive>
  void save(Archive & archive) const
  {
    MatSerializer ser_train_history(train_history);
    archive( ser_train_history, verbose, k, n_passes, validate ); 
  }
  
  // Deserialze
  template<class Archive>
  void load(Archive & archive)
  {
    MatSerializer ser_train_history(train_history);
    archive( ser_train_history, verbose, k, n_passes, validate );
    train_history = ser_train_history.getMat();
  }
  
};

#endif