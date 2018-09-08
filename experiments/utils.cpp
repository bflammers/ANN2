// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "utils.h"
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Constants
// ---------------------------------------------------------------------------//

double double_epsilon = std::numeric_limits<double>::epsilon();

// ---------------------------------------------------------------------------//
// Functions
// ---------------------------------------------------------------------------//

// Make a matrix consisting of one repeated column
mat repColVec(vec colvec, int n)
{
  mat result(colvec.size(), n);
  result.each_col() = colvec;
  return result;
}

// // Armadillo modulo function
// template<typename T>
// T modulo (T a, int n)
// {
//   return a - arma::floor(a/n) * n;
// }

// Make progress bar string
std::string progressBar(int progress) {
  int bar_width = 50;
  std::stringstream progress_string;
  int pos = bar_width * progress / 100;
  
  progress_string << "[";
  for (int i = 0; i != bar_width; ++i) {
    if (i < pos) progress_string << "+";
    else if (i == pos) progress_string << "|";
    else progress_string << "-";
  }
  progress_string << "] " << progress << "%";
  return progress_string.str();
}

// ---------------------------------------------------------------------------//
// Tracker class
// ---------------------------------------------------------------------------//
tracker::tracker () : k(0), one_percent(100) {}
tracker::~tracker () { Rcout << std::endl; }

void tracker::setTracker (int n_passes_, bool validate_, List train_param_)
{
  verbose = train_param_["verbose"];
  validate = validate_;
  n_passes = k + n_passes_;
  one_percent = std::max( double(n_passes - 1) / 100, double_epsilon );
  train_history.resize(n_passes, 2);
}

void tracker::track (double train_loss, double val_loss) {

  // Update progress bar and loss
  if ( verbose ) {

    double progress = k / one_percent;
    int progress_perc = Rcpp::ceiling( progress );
    if ( true ) {
      
      std::stringstream progress_stream;
      progress_stream << progressBar( progress_perc );
      
      if ( validate ) {
        progress_stream << " - Validation loss: " << val_loss;
      } else {
        progress_stream << " - Training loss: " << train_loss;
      }
      
      Rcout << "\r" << progress_stream.str();
      Rcout.flush();
      
    }
  }

  // Add train and validation loss to matrix
  rowvec loss_vec = {train_loss, val_loss};
  train_history.row(k) = loss_vec;
  
  // Increment counter
  k++;
}

// ---------------------------------------------------------------------------//
// Scaler class
// ---------------------------------------------------------------------------//

scaler::scaler (mat z, bool standardize_, List net_param_)
  : standardize(standardize_)
{
  ivec num_nodes = net_param_["num_nodes"];
  if ( standardize ) {
    z_mu = mean(z);
    z_sd = stddev(z);
  } else {
    z_mu = zeros<rowvec>(num_nodes[0]);
    z_sd = ones<rowvec>(num_nodes[num_nodes.size() - 1]);
  }
}

mat scaler::scale(mat z) 
{ 
  if ( standardize ) {
    z.each_row() -= z_mu;
    z.each_row() /= z_sd;
  }
  return z;
}

mat scaler::unscale(mat z) 
{
  if ( standardize ) {
    z.each_row() %= z_sd;
    z.each_row() += z_mu;
  }
  return z;
}

// ---------------------------------------------------------------------------//
// Sampler class
// ---------------------------------------------------------------------------//
sampler::sampler (mat X_, mat Y_, List train_param)
{
  // Training parameters 
  int batch_size = train_param["batch_size"];
  double val_prop = train_param["val_prop"];
  bool drop_last = train_param["drop_last"];
  
  // Derived parameters
  int n_obs = X_.n_rows;
  n_train = ceil ( (1 - val_prop ) * n_obs );
  n_batch = ceil ( double( n_train ) / batch_size );
  n_batch = ( modulo(n_train, n_batch) == 0 ) ? n_batch : n_batch - drop_last;
  validate = ( n_train < n_obs );
  
  // Randomly shuffle X and Y for train/validation split
  uvec rand_perm = arma::shuffle(regspace<uvec>(0, n_obs - 1));
  mat X = X_.rows(rand_perm);
  mat Y = Y_.rows(rand_perm);
  
  // Divide X and Y in training and validation sets
  X_train = X.rows(0, n_train - 1);
  Y_train = Y.rows(0, n_train - 1);
  if ( validate ) {
    X_val = X.rows(n_train, n_obs - 1);
    Y_val = Y.rows(n_train, n_obs - 1);
  }
  
  // Fill indices list with uvecs to subset X_train and Y_train for batches
  for (int i = 0; i != n_batch; i++) {
    int batch_start = i * batch_size;
    int batch_stop = std::min( n_train, (i+1) * batch_size ) - 1;
    uvec batch_range = regspace<uvec>(batch_start, batch_stop);
    indices.push_back ( batch_range );
  }
  
  // Set list iterators to begin
  Xit = indices.begin();
  Yit = indices.begin();
}

void sampler::shuffle () 
{
  // Randomly shuffle X and Y for train/validation split
  uvec rand_perm = arma::shuffle(regspace<uvec>(0, n_train - 1));
  X_train = X_train.rows(rand_perm);
  Y_train = Y_train.rows(rand_perm);
  
  // Set list iterators to begin
  Xit = indices.begin();
  Yit = indices.begin();
}

mat sampler::nextXb () 
{ 
  mat X_batch = X_train.rows( (*Xit) );
  std::advance(Xit, 1);
  return X_batch; 
}

mat sampler::nextYb () 
{ 
  mat Y_batch = Y_train.rows( (*Yit) );
  std::advance(Yit, 1);
  return Y_batch; 
}

mat sampler::getXv () 
{ 
  return X_val; 
}

mat sampler::getYv () 
{ 
  return Y_val; 
}


