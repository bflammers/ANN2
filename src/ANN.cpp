// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "ANN.h"

using namespace Rcpp;
using namespace arma;

// Default constructor needed for serialization
ANN::ANN() {};

// ANN class constructor
ANN::ANN(List data_, List net_param_, List optim_param_, List loss_param_, List activ_param_)
  : scaler_X(as<mat>(data_["X"]), as<bool>(net_param_["stand_X"])),
    scaler_y(as<mat>(data_["y"]), as<bool>(net_param_["stand_y"])),
    tracker(as<bool>(net_param_["verbose"])),
    epoch(0)
{
  
  // Set loss using factory design pattern based on user input
  L = LossFactory(loss_param_);
  
  // Set meta data
  num_nodes = as< std::vector<int> >(net_param_["num_nodes"]);
  y_names = as< std::vector<std::string> >(data_["y_names"]);
  regression = as<bool>(net_param_["regression"]);
  autoencoder = as<bool>(net_param_["autoencoder"]);
  
  // Set iterable vectors for activation type and learn_rates
  StringVector activ_types = activ_param_["types"];
  vec learn_rates = optim_param_["learn_rates"];
  
  // Set parameter Lists to be passed to layer()
  List activ_param = activ_param_;
  List optim_param = optim_param_;
  
  // Set layers
  for(int i = 1; i!=num_nodes.size(); i++){
    activ_param["type"] = activ_types(i);
    optim_param["learn_rate"] = learn_rates(i);
    Layer l(num_nodes[i-1], num_nodes[i], activ_param, optim_param);
    layers.push_back(std::move(l)); // std::move needed with use of unique_ptr
  }
  
  // Print NN info if verbose
  if ( as<bool>(net_param_["verbose"]) ) print( false );
}

// Forward pass through the network 
// Does not take care of data scaling: this happens before calling forwardPass()
// in train() and predict() methods
mat ANN::forwardPass (mat X) 
{
  X = X.t();
  for(it = layers.begin(); it != layers.end(); ++it) {
    X = it->forward(X);
  }
  return X.t();
}

// Forward pass through the network 
// Error matrix E is propagated backwards through the network
// Layer member method backward() also performs parameter updates
mat ANN::backwardPass (mat y, mat y_fit) 
{
  mat E = L->grad(y, y_fit);
  for(rit = layers.rbegin(); rit != layers.rend(); ++rit) {
    E = rit->backward(E);
  }
  // Return error matrix. This is only used for gradient checking
  return E;
}

// Update pass through the network 
// Invokes the update() method for all layers in the network
void ANN::updatePass ()
{
  for(it = layers.begin(); it != layers.end(); ++it) {
    it->update();
  }
}

// Method to make a partial forward pass
// This is useful for calculating the hidden layer representation for a given
// input matrix but also for going from a given hidden layer representation
// to the fitted values (neural network output)
mat ANN::partialForward (mat X, int i_start, int i_stop) 
{
  
  // Set start & stop point iterators
  std::list<Layer>::iterator start_it = layers.begin();
  std::advance(start_it, i_start);
  std::list<Layer>::iterator stop_it = layers.begin();
  std::advance(stop_it, i_stop);
  
  // If input layer: standardize
  if ( i_start == 0 ) {
    X = scaler_X.scale(X); // Check if start at 0
  }
  
  X = X.t();
  // Loop from start_it to stop_it
  for(it = start_it; it != stop_it; ++it) {
    X = it->forward(X);
  }
  X = X.t();
  
  // If output layer: undo standardize
  if ( i_stop == layers.size() ) {
    X = scaler_y.unscale(X);
  }
  
  return X;
}

mat ANN::predict (mat X) 
{
  X = scaler_X.scale(X);
  X = forwardPass(X);
  mat y_pred = scaler_y.unscale(X);
  return y_pred;
}

// Train the network!
void ANN::train (List data, List train_param)
{
  
  // Training parameters
  int n_epochs = train_param["n_epochs"];
  int max_epochs = epoch + n_epochs;
  
  // Scale data
  mat X = scaler_X.scale(as<mat>(data["X"]));
  mat y = scaler_y.scale(as<mat>(data["y"]));

  // Set sampler and tracker
  Sampler sampler(X, y, train_param);
  int n_new_passes = n_epochs * sampler.n_batch;
  tracker.setTracker(n_new_passes, sampler.validate, train_param);
  
  // Set training set size in optimizer for scaling the regularization
  for(it = layers.begin(); it != layers.end(); ++it) {
    it->O->n_train = sampler.n_train;
  }

  for (; epoch != max_epochs; epoch++) {

    // Shuffle data
    sampler.shuffle();

    for (int b = 0; b != sampler.n_batch; b++) {
      
      // Sample new batch
      mat Xb = sampler.next_Xb();
      mat yb = sampler.next_yb();

      // Forward pass
      mat yb_fit = forwardPass(Xb);
      
      // Backward pass, calculates gradients wrt. weights and biases
      // Returns a Armadillo matrix. This is only used for gradient checking
      mat _ = backwardPass(yb, yb_fit);
      
      // Update parameters
      updatePass();
      
      // Track loss on scaled data
      double batch_loss = accu( L->eval(yb, yb_fit) ) / yb.n_rows;
      
      // Track validation loss on scaled data
      double val_loss = 0;
      if (sampler.validate) {
        mat y_val_fit = forwardPass(sampler.get_Xv());
        val_loss = accu( L->eval(sampler.get_yv(), y_val_fit) );
        val_loss /= y_val_fit.n_rows;
      } 
      
      // Add loss to tracker
      tracker.track(epoch, batch_loss, val_loss);
      
      // Check for interrupt
      Rcpp::checkUserInterrupt();
      
    }
  }
  
  // End printing on the same line
  tracker.endLine();
  
}

// Class method for printing NN information
void ANN::print ( bool print_epochs ) {
  
  // Use stringstream to pass only one string to Rcout
  std::stringstream print_stream;
  
  // Add first line and input layer line (input layer not in layers List)
  print_stream << "Artificial Neural Network: \n";
  print_stream << "  Layer - " << num_nodes[0] << " nodes - input \n";
  
  // Get number of nodes and activation type for each layer and add to stream
  for(it = layers.begin(); it != layers.end(); ++it) {
    print_stream << it->print();
  }
  
  // Print loss type and optimizer type
  print_stream << "With " << L->type << " loss and " << 
    layers.front().O->type << " optimizer \n";
  
  // Add the amount of training (in epochs) to stream
  if ( print_epochs ) print_stream << "Trained for " << epoch << " epochs \n"; 
  
  // Pass stream as string to Rcout to print
  Rcout << print_stream.str();
}

// Class method for accessing training history
List ANN::getTrainHistory ( ) {
  
  // Collect loss vectors in list and return
  return List::create(Named("n_epoch") = epoch, 
                      Named("n_eval") = tracker.n_passes, 
                      Named("validate") = tracker.validate,
                      Named("epoch") = tracker.train_history.col(0),
                      Named("train_loss") = tracker.train_history.col(1),
                      Named("val_loss") = tracker.train_history.col(2));
  
}

// Method for accessing network meta info
// It is nice to store all this information in the C++ object, and not at the 
// R level because now we can use pure C++ serialization when writing the 
// ANN object to a file 
List ANN::getMeta()
{
  int n_layers = num_nodes.size();

  return List::create(Named("no_hidden") = n_layers == 2,
                      Named("n_hidden") = n_layers - 2,
                      Named("n_in") = num_nodes[0], 
                      Named("n_out") = num_nodes[n_layers - 1], 
                      Named("regression") = regression,
                      Named("y_names") = y_names,
                      Named("num_nodes") = num_nodes, 
                      Named("autoencoder") = autoencoder);
}

// Method for obtaining the weights and biases
List ANN::getParams()
{
  List weights, biases;
  for(it = layers.begin(); it != layers.end(); ++it) {
    weights.push_back(it->W);
    biases.push_back(it->b);
  }
  return List::create(Named("weights") = weights,
                      Named("biases") = biases);
}

// Method to evaluate the loss function - exposed & used for gradient checking
mat ANN::evalLoss(mat y, mat y_fit) 
{
  return L->eval(y, y_fit);
}

// Method for scaling X, exposed to R level
mat ANN::scale_X(mat X, bool inverse) 
{
  if (inverse) {
    return scaler_X.unscale(X);
  }
  return scaler_X.scale(X);
}

// Method for scaling y, exposed to R level
mat ANN::scale_y(mat y, bool inverse)
{
  if (inverse) {
    return scaler_y.unscale(y);
  }
  return scaler_y.scale(y);
}

void ANN::write (const char* fileName) {
  
  {
    std::ofstream ofs(fileName, std::ios::binary);
    cereal::PortableBinaryOutputArchive oarchive(ofs);
    ANN::serialize(oarchive);
  }
  
}

void ANN::read (const char* fileName) {
  
  {
    std::ifstream ifs(fileName, std::ios::binary);
    cereal::PortableBinaryInputArchive iarchive(ifs);
    ANN::serialize(iarchive);
  }
}

