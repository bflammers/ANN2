Rcpp::loadModule("ANN", TRUE)

ignoreMe <- setMethod("show", "Rcpp_ANN", function (object) {
  cat("\n Hi, I am an ANN object!\n")
})