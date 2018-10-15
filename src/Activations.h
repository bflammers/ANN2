#ifndef ACTIVATIONS
#define ACTIVATIONS

#include <RcppArmadillo.h>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>
#include <memory>

#include "arma_serialization.h"

// ---------------------------------------------------------------------------//
// Base activation class
// ---------------------------------------------------------------------------//

class Activation {
public: 
  std::string type;
  virtual arma::mat eval(arma::mat X) = 0;
  virtual arma::mat grad(arma::mat X) = 0;
};

// ---------------------------------------------------------------------------//
// Tanh activation class
// ---------------------------------------------------------------------------//

class TanhActivation : public Activation
{
private:
  arma::mat A;
public:
  // Constructor
  TanhActivation();
  // Evaluate tanh
  arma::mat eval(arma::mat X);
  // Derivative tanh
  arma::mat grad(arma::mat X);
  
  // Serialize
  template<class Archive>
  void serialize( Archive & archive ) {
    archive( type ); 
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(TanhActivation);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Activation, TanhActivation);

// ---------------------------------------------------------------------------//
// Sigmoid activation class
// ---------------------------------------------------------------------------//

class SigmoidActivation : public Activation
{
private:
  arma::mat A;
public:
  // Constructor
  SigmoidActivation ();
  // Evaluate sigmoid
  arma::mat eval(arma::mat X) ;
  // Derivative sigmoid
  arma::mat grad(arma::mat X);
  
  // Serialize
  template<class Archive>
  void serialize( Archive & archive ) {
    archive( type ); 
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(SigmoidActivation);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Activation, SigmoidActivation);

// ---------------------------------------------------------------------------//
// Rectifier activation class
// ---------------------------------------------------------------------------//

class ReluActivation : public Activation
{
public:
  // Constructor
  ReluActivation ();
   // Evaluate relu
  arma::mat eval(arma::mat X);
  // Derivative relu
  arma::mat grad(arma::mat X);
  
  // Serialize
  template<class Archive>
  void serialize( Archive & archive ) {
    archive( type ); 
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(ReluActivation);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Activation, ReluActivation);

// ---------------------------------------------------------------------------//
// Linear activation class
// ---------------------------------------------------------------------------//

class LinearActivation : public Activation
{
public:
  // Constructor
  LinearActivation ();
  // Evaluate
  arma::mat eval(arma::mat X);
  // Derivative
  arma::mat grad(arma::mat X);
  
  // Serialize
  template<class Archive>
  void serialize( Archive & archive ) {
    archive( type ); 
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(LinearActivation);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Activation, LinearActivation);

// ---------------------------------------------------------------------------//
// Softmax activation class
// ---------------------------------------------------------------------------//

class SoftMaxActivation : public Activation
{
private:
  arma::mat A;
public:
  // Constructor
  SoftMaxActivation ();
  // Evaluate softmax
  arma::mat eval(arma::mat X);
  // Derivative softmax
  arma::mat grad(arma::mat X);
  
  // Serialize
  template<class Archive>
  void serialize( Archive & archive ) {
    archive( type ); 
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(SoftMaxActivation);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Activation, SoftMaxActivation);

// ---------------------------------------------------------------------------//
// Step function activation class
// ---------------------------------------------------------------------------//

class StepActivation : public Activation
{
private:
  int H; 
  double k;
  arma::vec seqH;
  arma::mat::iterator mit;
public:
  // Default constructor needed for serialization
  StepActivation ();
  // Constructor 
  StepActivation (Rcpp::List activ_param_);
  // Evaluate stepfunction
  arma::mat eval(arma::mat X);
  // Derivative stepfunction
  arma::mat grad(arma::mat X);
  
  // Saving step serialization
  template<class Archive>
  void save(Archive & archive) const
  {
    VecSerializer ser_seqH(seqH);
    archive( type, H, k, ser_seqH ); 
  }
  
  // Loading step serialization
  template<class Archive>
  void load(Archive & archive)
  {
    RowVecSerializer ser_seqH;
    archive( type, H, k, ser_seqH ); 
    seqH = ser_seqH.getRowVec();
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(StepActivation);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Activation, StepActivation);

// ---------------------------------------------------------------------------//
// Ramp function activation class
// ---------------------------------------------------------------------------//

class RampActivation : public Activation
{
public:
  // Constructor
  RampActivation ();
  // Evaluate ramp
  arma::mat eval(arma::mat X);
  // Derivative ramp
  arma::mat grad(arma::mat X);
  
  // Serialize
  template<class Archive>
  void serialize( Archive & archive ) {
    archive( type ); 
  }
};

// Register class for serialization
CEREAL_REGISTER_TYPE(RampActivation);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Activation, RampActivation);

// ---------------------------------------------------------------------------//
// Activation factory
// ---------------------------------------------------------------------------//

std::shared_ptr<Activation> ActivationFactory (Rcpp::List activ_param);

#endif