library(testthat)
library(ANN2)

# Only test if not on mac
if (tolower(Sys.info()[["sysname"]]) != "darwin") {
  test_check("ANN2")
}
