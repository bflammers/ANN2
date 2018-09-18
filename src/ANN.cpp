// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include "utils.h"
#include "loss.h"
#include "layer.h"
using namespace Rcpp;
using namespace arma;

// Class ANN
//' @export ANN
class ANN 
{
private:
  std::list<layer> layers;
  std::list<layer>::iterator it;
  std::list<layer>::reverse_iterator rit;
  Scaler sX, sY;
  loss *L;
  Tracker tracker;
  int epoch;
  
public:
  ANN(List data_, List net_param_, List optim_param_, List loss_param_, 
      List activ_param_);
  mat forwardPass (mat X);
  void backwardPass (mat Y, mat Y_fit);
  mat partialForward (mat X, int i_start, int i_stop);
  mat predict (mat X);
  double evalLoss(mat Y, mat X);
  void train (List data, List train_param);

};

// ANN class constructor
ANN::ANN(List data_, List net_param_, List optim_param_, List loss_param_, 
    List activ_param_)
  : sX(data_["X"], net_param_["stand_X"]),
    sY(data_["Y"], net_param_["stand_Y"]),
    epoch(0)
{
  
  // Set loss
  lossFactory lFact(loss_param_); 
  L = lFact.createLoss();
  
  // Set iterable vectors for number of nodes, activation type and learn_rates
  ivec num_nodes = net_param_["num_nodes"];
  StringVector activ_types = activ_param_["types"];
  vec learn_rates = optim_param_["learn_rates"];
  
  // Set parameter Lists to be passed to layer()
  List activ_param = activ_param_;
  List optim_param = optim_param_;
  
  // Set layers
  for(int i = 1; i!=num_nodes.size(); i++){
    activ_param["type"] = activ_types(i);
    optim_param["learn_rate"] = learn_rates(i);
    layer l(num_nodes(i-1), num_nodes(i), activ_param, optim_param);
    layers.push_back(l);
  }
}

mat ANN::forwardPass (mat X) 
{
  X = X.t();
  for(it = layers.begin(); it != layers.end(); ++it) {
    X = it->forward(X);
  }
  return X.t();
}

void ANN::backwardPass (mat Y, mat Y_fit) 
{
  mat E = L->grad(Y, Y_fit);
  for(rit = layers.rbegin(); rit != layers.rend(); ++rit) {
    E = rit->backward(E);
  }
}

mat ANN::partialForward (mat X, int i_start, int i_stop) 
{
  
  // Set start & stop point iterators
  std::list<layer>::iterator start_it = layers.begin();
  std::advance(start_it, i_start - 1);
  std::list<layer>::iterator stop_it = layers.begin();
  std::advance(stop_it, i_stop - 1);
  
  // If input layer: standardize and transpose
  if ( i_start == 1 ) {
    X = sX.scale(X); // Check if start at 0
    X = X.t();
  }
  
  // Loop from start_it to stop_it
  for(it = start_it; it != stop_it; ++it) {
    X = it->forward(X);
  }
  
  // If output layer: transpose and standardize
  if ( i_stop == layers.size() ) {
    X = X.t();
    X = sY.unscale(X);
  }
  
  return X;
}

mat ANN::predict (mat X) 
{
  X = sX.scale(X);
  X = forwardPass(X);
  mat Y_pred = sY.unscale(X);
  return Y_pred;
}

// Evaluate loss, input should be scaled data
double ANN::evalLoss(mat Y, mat X)
{
  return accu( L->eval(Y, forwardPass(X)) );
}

void ANN::train (List data, List train_param)
{
  
  // Training parameters
  int n_epochs = train_param["n_epochs"];
  int max_epochs = epoch + n_epochs;
  
  // Scale data
  mat X = sX.scale(data["X"]);
  mat Y = sY.scale(data["Y"]);

  // Set sampler and tracker
  Sampler sampler(X, Y, train_param);
  int n_new_passes = n_epochs * sampler.n_batch;
  tracker.setTracker(n_new_passes, sampler.validate, train_param);

  for (; epoch != max_epochs; epoch++) {

    // Shuffle data
    sampler.shuffle();

    for (int b = 0; b != sampler.n_batch; b++) {
      
      // Sample new batch
      mat Xb = sampler.nextXb();
      mat Yb = sampler.nextYb();

      // Forward pass
      mat Yb_fit = forwardPass(Xb);
      
      // Backward pass, also includes update
      backwardPass(Yb, Yb_fit);
      
      // Track loss on scaled data
      double batch_loss = L->eval(Yb, Yb_fit);
      double val_loss = (sampler.validate) ? evalLoss(sampler.getYv(), sampler.getXv()) : 0;
      tracker.track(batch_loss, val_loss);
      
      // Check for interrupt
      checkUserInterrupt();
      
    }
  }
  
  // End printing on the same line
  tracker.endLine();
}

RCPP_MODULE(ANN) {
  using namespace Rcpp ;
  class_<ANN>( "ANN" )
  .constructor<List, List, List, List, List>()
  .method( "predict", &ANN::predict)
  .method( "partialForward", &ANN::partialForward)
  .method( "train", &ANN::train)
  ;
}

