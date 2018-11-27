#ifndef ANN_H
#define ANN_H

#include <RcppArmadillo.h>
#include "utils.h"
#include "Loss.h"
#include "Layer.h"

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/vector.hpp>

// Class ANN
//' @export
class ANN 
{
private:
  
  // STD list containing the hidden layers and the output layer as well as the 
  // iterators needed to traverse this list in the forward and backward pass
  std::list<Layer> layers;
  std::list<Layer>::iterator it;
  std::list<Layer>::reverse_iterator rit;
  
  // Loss object pointer - dynamically assigned using a factory design pattern
  std::unique_ptr<Loss> L;
  
  // Scaler objects for scaling and unscaling the data
  Scaler sX, sY;
  
  // Tracker object and epoch integer for keeping track of the training process
  // Epoch integer as a member of ANN class (and not in tracker object) because 
  // I prefer to use a for loop over the epochs during training and this is most 
  // transparant 
  Tracker tracker;
  int epoch;
  
  // Members storing meta info
  std::vector<std::string> y_names;
  std::vector<int> num_nodes;
  bool regression;
  
public:
  
  // Constructors
  ANN(); // Default constructor needed for serialization
  ANN(Rcpp::List data_, Rcpp::List net_param_, Rcpp::List loss_param_, 
      Rcpp::List activ_param_, Rcpp::List optim_param_);
  
  // Forward and backward pass
  arma::mat forwardPass (arma::mat X);
  void backwardPass (arma::mat Y, arma::mat Y_fit);
  
  // Partial forward, used to get hidden layer representation
  arma::mat partialForward (arma::mat X, int i_start, int i_stop);
  
  // Predict method - calls forwardPass() after scaling
  arma::mat predict (arma::mat X);
  
  // Evaluates loss for a given input matrix
  double evalLoss(arma::mat Y, arma::mat X);
  
  // Train the network
  void train (Rcpp::List data, Rcpp::List train_param);
  
  // Print method
  void print (bool print_epochs);
  
  // Get methods for accessing training and meta info
  Rcpp::List getTrainHistory ();
  Rcpp::List getMeta();
  
  // Methods used to read/write the network to/from file
  // These methods make a call to the serialize() method
  void write (const char* fileName);
  void read (const char* fileName);
  
  // Serialize
  template<class Archive>
  void serialize(Archive & archive) {
    archive( epoch, tracker, sX, sY, L, layers, num_nodes, y_names, regression ); 
  }
  
};

RCPP_MODULE(ANN) {
  using namespace Rcpp ;
  class_<ANN>( "ANN" )
    .constructor<List, List, List, List, List>()
    .method( "predict", &ANN::predict)
    .method( "partialForward", &ANN::partialForward)
    .method( "train", &ANN::train)
    .method( "print", &ANN::print)
    .method( "getTrainHistory", &ANN::getTrainHistory)
    .method( "write", &ANN::write)
    .method( "read", &ANN::read)
    .method( "getMeta", &ANN::getMeta)
  ;
}

#endif
