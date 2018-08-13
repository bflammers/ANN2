#include <Rcpp.h>
using namespace Rcpp;

#include <iostream>
using namespace std;

class printData {
public:
  void print(int i) {
    cout << "Printing int: " << i << endl;
  }
  void print(double  f, int i) {
    cout << "Printing float: " << f << i << endl;
  }
  void print(String* c) {
    cout << "Printing character: " << c << endl;
  }
};

// [[Rcpp::export]]
int fn() {
  printData pd;
  
  // Call print to print integer
  pd.print(5);
  
  // Call print to print float
  pd.print(500.263, 2);
  
  // Call print to print character
  pd.print(1);
  
  return 0;
}

/*** R
fn()
*/
