#ifndef LOSS
#define LOSS

#include <RcppArmadillo.h>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <memory>

// ---------------------------------------------------------------------------//
// Base loss class
// ---------------------------------------------------------------------------//

class Loss {
public: 
  std::string type;
  virtual double eval(arma::mat y, arma::mat y_fit) = 0;
  virtual arma::mat grad(arma::mat y, arma::mat y_fit) = 0;
};

// ---------------------------------------------------------------------------//
// Log loss class
// ---------------------------------------------------------------------------//

class LogLoss : public Loss
{
public:
  LogLoss ();
  double eval (arma::mat y, arma::mat y_fit);
  arma::mat grad (arma::mat y, arma::mat y_fit);
  
  template<class Archive>
  void serialize( Archive & archive ) {
    archive( type ); 
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(LogLoss);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Loss, LogLoss);


// ---------------------------------------------------------------------------//
// Squared loss class
// ---------------------------------------------------------------------------//

class SquaredLoss : public Loss 
{
public:
  SquaredLoss ();
  double eval(arma::mat y, arma::mat y_fit);
  arma::mat grad(arma::mat y, arma::mat y_fit);
  
  template<class Archive>
  void serialize( Archive & archive ) {
    archive( type ); 
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(SquaredLoss);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Loss, SquaredLoss);

// ---------------------------------------------------------------------------//
// Absolute loss class
// ---------------------------------------------------------------------------//

class AbsoluteLoss : public Loss
{
public:
  AbsoluteLoss ();
  double eval(arma::mat y, arma::mat y_fit);
  arma::mat grad(arma::mat y, arma::mat y_fit);
  
  template<class Archive>
  void serialize( Archive & archive ) {
    archive( type ); 
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(AbsoluteLoss);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Loss, AbsoluteLoss);

// ---------------------------------------------------------------------------//
// Huber loss class
// ---------------------------------------------------------------------------//

class HuberLoss : public Loss
{
private:
  double delta_huber;
public:
  HuberLoss();
  HuberLoss(Rcpp::List loss_param_);
  double eval(arma::mat y, arma::mat y_fit);
  arma::mat grad(arma::mat y, arma::mat y_fit);
  
  template<class Archive>
  void serialize( Archive & archive ) {
    archive( type, delta_huber ); 
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(HuberLoss);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Loss, HuberLoss);

// ---------------------------------------------------------------------------//
// Pseudo-Huber loss class
// ---------------------------------------------------------------------------//

class PseudoHuberLoss : public Loss
{
private:
  double delta_huber;
public:
  PseudoHuberLoss();
  PseudoHuberLoss(Rcpp::List loss_param_);
  double eval(arma::mat y, arma::mat y_fit);
  arma::mat grad(arma::mat y, arma::mat y_fit);
  
  template<class Archive>
  void serialize( Archive & archive ) {
    archive( type, delta_huber ); 
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(PseudoHuberLoss);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Loss, PseudoHuberLoss);

// ---------------------------------------------------------------------------//
// Loss factory
// ---------------------------------------------------------------------------//

std::unique_ptr<Loss> LossFactory (Rcpp::List loss_param_);

#endif