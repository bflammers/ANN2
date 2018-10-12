// Enable C++11 via this plugin 
// [[Rcpp::plugins("cpp11")]]

// [[Rcpp::depends(Rcereal)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <cereal/types/vector.hpp>
#include <cereal/archives/portable_binary.hpp>
#include "utils.h"
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------//
// Armadillo matrix wrapper for serialization
// ---------------------------------------------------------------------------//
MatSerializer::MatSerializer () {}

MatSerializer::MatSerializer (mat X) : ncol(X.n_cols), nrow(X.n_rows) {
  X_holder.resize(ncol);
  for (size_t i = 0; i < ncol; ++i) {
    X_holder[i] = arma::conv_to< std::vector<double> >::from( X.col(i) );
  };
}

mat MatSerializer::getMat () {
  mat X(nrow, ncol);
  for (size_t i = 0; i < ncol; ++i) {
    X.col(i) = arma::conv_to< colvec >::from( X_holder[i] );
  };
  return X;
}

template<typename Archive>
void MatSerializer::serialize(Archive& ar) {
  ar( nrow , ncol , X_holder );
}

// ---------------------------------------------------------------------------//
// Tracker class
// ---------------------------------------------------------------------------//
Tracker::Tracker () {}

Tracker::Tracker (bool verbose_)
  : k(0), curr_progress(0), one_percent(100), verbose(verbose_) {}

// Set tracker parameters of training run, allocate memory
void Tracker::setTracker (int n_passes_, bool validate_, List train_param_)
{
  validate = validate_;
  n_passes = k + n_passes_;
  one_percent = std::max( double(n_passes - 1) / 100, 
                          std::numeric_limits<double>::epsilon());
  train_history.resize(n_passes, 3);
  
  if ( verbose ) Rcout << "Training progress:\n";
}

// Private method for creating progress bar string
std::string Tracker::progressBar(int progress) {
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

void Tracker::track (int epoch, double train_loss, double val_loss) {

  // Update progress bar and loss
  if ( verbose ) {

    int progress_perc = std::ceil( k / one_percent );
    if ( progress_perc != curr_progress || k % 10 == 0 ) {
      
      curr_progress = progress_perc;
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
  rowvec loss_vec = {double(epoch), train_loss, val_loss};
  train_history.row(k) = loss_vec;
  
  // Increment counter
  k++;
}

// End the line after
void Tracker::endLine () { if ( verbose ) Rcout << std::endl; }

// Serialize
template<class Archive>
void Tracker::save(Archive & archive) const
{
  MatSerializer ser_train_history(train_history);
  archive( ser_train_history , verbose, k ); 
}

// Deserialze
template<class Archive>
void Tracker::load(Archive & archive)
{
  MatSerializer ser_train_history(train_history);
  archive( ser_train_history , verbose, k );
  train_history = ser_train_history.getMat();
}

// ---------------------------------------------------------------------------//
// Scaler class
// ---------------------------------------------------------------------------//

Scaler::Scaler () {}

Scaler::Scaler (mat z, bool standardize_)
  : standardize(standardize_), 
    n_col(z.n_cols)
{
  if ( standardize ) {
    z_mu = mean(z);
    z_sd = stddev(z);
  } else {
    z_mu = zeros<rowvec>(z.n_cols);
    z_sd = ones<rowvec>(z.n_cols);
  }
}

mat Scaler::scale(mat z) 
{ 
  if ( standardize ) {
    z.each_row() -= z_mu;
    z.each_row() /= z_sd;
  }
  return z;
}

mat Scaler::unscale(mat z) 
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
Sampler::Sampler (mat X_, mat Y_, List train_param)
{
  // Training parameters 
  int batch_size = train_param["batch_size"];
  double val_prop = train_param["val_prop"];
  bool drop_last = train_param["drop_last"];
  
  // Derived parameters
  int n_obs = X_.n_rows;
  n_train = std::ceil ( (1 - val_prop ) * n_obs );
  n_batch = std::ceil ( double( n_train ) / batch_size );
  n_batch = ( (n_train % n_batch) == 0 ) ? n_batch : n_batch - drop_last;
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

void Sampler::shuffle () 
{
  // Randomly shuffle X and Y for train/validation split
  uvec rand_perm = arma::shuffle(regspace<uvec>(0, n_train - 1));
  X_train = X_train.rows(rand_perm);
  Y_train = Y_train.rows(rand_perm);
  
  // Set list iterators to begin
  Xit = indices.begin();
  Yit = indices.begin();
}

mat Sampler::nextXb () 
{ 
  mat X_batch = X_train.rows( (*Xit) );
  std::advance(Xit, 1);
  return X_batch; 
}

mat Sampler::nextYb () 
{ 
  mat Y_batch = Y_train.rows( (*Yit) );
  std::advance(Yit, 1);
  return Y_batch; 
}

mat Sampler::getXv () 
{ 
  return X_val; 
}

mat Sampler::getYv () 
{ 
  return Y_val; 
}

// 
// // [[Rcpp::export]]
// void write (const char* fileName) {
//   // Create an output archive
//   Tracker t(true);
//   t.setTracker(2, true, List::create(Rcpp::Named("vec") = 1));
//   {
//     std::ofstream ofs(fileName, std::ios::binary);
//     cereal::PortableBinaryOutputArchive oarchive(ofs);
//     t.save(oarchive);
//   }
// }
// 
// // [[Rcpp::export]]
// void read (const char* fileName) {
//   Tracker tt; {
//     std::ifstream ifs(fileName, std::ios::binary);
//     cereal::PortableBinaryInputArchive iarchive(ifs);
//     tt.load(iarchive);
//   }
//   Rcout << tt.train_history;
// }
// 
// 
// /*** R
// write("/home/bart/Documents/R/ANN2/src/test.bin")
// read("/home/bart/Documents/R/ANN2/src/test.bin")
// */
// 


