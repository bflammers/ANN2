// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>

arma::mat repVec(arma::vec vector, int ntimes){
  arma::mat result(vector.size(), ntimes);
  for(int i = 0; i!=ntimes; i++){
    result.col(i) =vector;
  }
  return(result);
}

arma::vec huberLoss(arma::mat y_fit, arma::mat y, double dHuber){
  arma::mat a            = abs(y_fit - y);
  arma::mat elemGradLoss = dHuber * (a - dHuber/2);
  arma::uvec indQuad     = find(a <= dHuber);
  elemGradLoss(indQuad)  = pow(a(indQuad), 2)/2;
  return sum(elemGradLoss,1);
}

arma::mat huberGradLoss(arma::mat y_fit, arma::mat y, double dHuber){
  arma::mat err          = y_fit - y;
  arma::mat elemGradLoss = dHuber * sign(err);
  arma::uvec indQuad     = find(abs(err) <= dHuber);
  elemGradLoss(indQuad)  = err(indQuad);
  return elemGradLoss.t();
}

// [[Rcpp::export]]
arma::mat stepFun(arma::mat x, int nSteps, int smoothSteps){
  arma::uword nElements = x.n_elem;
  arma::mat result(size(x));
  arma::vec seqN = arma::linspace(1, nSteps, nSteps)/ nSteps;
  for(arma::uword i = 0; i!=nElements; i++){
    result(i) = sum( tanh( smoothSteps * (x(i) - seqN) ) );
  }
  result = 0.5 + result / (2*(nSteps-1));
  return result;
}

// [[Rcpp::export]]
arma::mat stepGradFun(arma::mat x, int nSteps, int smoothSteps){
  arma::uword nElements = x.n_elem;
  arma::mat result(size(x));
  arma::vec seqN = arma::linspace(1, nSteps, nSteps)/ nSteps;
  for(arma::uword i = 0; i!=nElements; i++){
    result(i) = sum(1 - pow( tanh( smoothSteps * (x(i) - seqN) ), 2) );
  }
  result = smoothSteps * result / (2*(nSteps-1));
  return result;
}

arma::mat colSoftMax(arma::mat x) {
  int ncol = x.n_cols;
  arma::mat out(x.n_rows, ncol);
  for (int i = 0; i < ncol; i++) {
    arma::vec exp_x = exp( x.col(i) - max(x.col(i)) );
    out.col(i)      = exp_x / sum(exp_x);
  }
  return out;
}

arma::uvec sample_index(const int &size){
  arma::uvec sequence = arma::linspace<arma::uvec>(0, size-1, size);
  arma::uvec out = Rcpp::RcppArmadillo::sample(sequence, size, false);
  return out;
}

// Loss functions
arma::vec lossFunction(arma::mat y, arma::mat y_fit, Rcpp::String lossType, double dHuber){
  int nTrain = y.n_rows;
  arma::vec result(nTrain);
  if(lossType == "log"){
    result    = -log(y_fit.elem(find(y.t() == 1)));
    result.elem( find_nonfinite(result) ).fill(1e100); // Machine tolerance largest number? R_PosInf
  }else if(lossType == "quadratic"){
    result = sum(pow(y_fit.t() - y, 2), 1);
  }else if(lossType == "absolute"){
    result = sum(abs(y_fit.t() - y), 1);
  }else if(lossType == "huber"){
    result = huberLoss(y_fit.t(), y, 1);
  }else if(lossType == "pseudo-huber"){
    result = sum(sqrt(1 + pow((y_fit.t() - y)/dHuber, 2))-1, 1);
  }
  return result;
}

// Gradients of loss functions
arma::mat lossGradFunction(arma::mat y, arma::mat y_fit, Rcpp::String lossType, double dHuber){
  arma::mat result(y.n_rows, y.n_rows);
  if(lossType == "log") result = y_fit-y.t();
  else if(lossType == "quadratic") result = 2*(y_fit-y.t());
  else if(lossType == "absolute")  result = sign(y_fit-y.t());
  else if(lossType == "huber") result = huberGradLoss(y_fit.t(), y, 1);
  else if(lossType == "pseudo-huber"){
    arma::mat err = y_fit - y.t();
    result = err % (1/sqrt(1 + pow(err/dHuber, 2)));
  }
  return result;
}

// Activation functions
arma::mat activFunction(arma::mat x, Rcpp::String activType, int nSteps, int smoothSteps){
  arma::mat result(x.n_rows, x.n_cols);
  if(activType == "tanh")           result = 1.725*tanh(2*x/3);
  else if(activType == "sigmoid")   result = 1/(1+exp(-x));
  else if(activType == "rectifier") result = max(result.zeros(), x);
  else if(activType == "linear")    result = x;
  else if(activType == "softMax")   result = colSoftMax(x);
  else if(activType == "step")      result = stepFun(x, nSteps, smoothSteps);
  else if(activType == "ramp"){
    result = x;
    result.elem(find(x>=1)).fill(1);
    result.elem(find(x<=0)).fill(0);
  }
  return result;
}

// Gradients of activation 
arma::mat activGradFunction(arma::mat x, Rcpp::String activType, int nSteps, int smoothSteps){
  arma::mat result(x.n_rows, x.n_cols); result.zeros();
  if(activType == "tanh")           result = 1.15*(1-pow(tanh(2*x/3), 2));
  else if(activType == "sigmoid"){ arma::mat s = 1/(1+exp(-x)); result = s % (1-s); }
  else if(activType == "rectifier") result.elem(find(x > 0)).fill(1);
  else if(activType == "linear")    result.ones();
  else if(activType == "step")      result = stepGradFun(x, nSteps, smoothSteps);
  else if(activType == "ramp")      result.elem(find(0<x && x<1)).fill(1);
  else if(activType == "softMax"){
    arma::mat softMaxMat = colSoftMax(x);
    result               = softMaxMat % (1-softMaxMat);
  }
  return result;
}

// [[Rcpp::export]]
Rcpp::List scaleData(arma::mat X){
  int nCol = X.n_cols;
  arma::vec Xcenter(nCol), Xscale(nCol);
  arma::mat scaledX(size(X));
  for (int i = 0; i != nCol; i ++) {
    Xcenter[i] = arma::mean(X.col(i));
    Xscale[i]  = arma::stddev(X.col(i));
    if (Xscale[i] == 0) {
      Xscale[i] = 1;
    }
    scaledX.col(i) = (X.col(i) -  Xcenter[i]) / Xscale[i];
  }
  return Rcpp::List::create(
    Rcpp::Named("scaled") = scaledX,
    Rcpp::Named("center") = Xcenter,
    Rcpp::Named("scale")  = Xscale
  );
}

arma::mat scaleNewData(arma::mat X, arma::vec Xcenter, arma::vec Xscale){
  int nCol = X.n_cols;
  arma::mat scaledX(size(X));
  for(int i = 0; i != nCol; i ++){
    scaledX.col(i) = (X.col(i) -  Xcenter[i]) / Xscale[i];
  }
  return scaledX;
}

arma::mat reScaleData(arma::mat X, arma::vec Xcenter, arma::vec Xscale){
  int nCol = X.n_cols;
  arma::mat reScaledX(size(X));
  for(int i = 0; i != nCol; i ++){
    reScaledX.col(i) = X.col(i) * Xscale[i] + Xcenter[i];
  }
  return reScaledX;
}

Rcpp::List forwardPass(Rcpp::List upOut, arma::mat Xbatch, Rcpp::CharacterVector activTypes, int nSteps, int smoothSteps) {
  arma::field<arma::vec> biasVecs    = upOut["biasVecs"];
  arma::field<arma::mat> weightMats  = upOut["weightMats"];

  int nLayers = biasVecs.size(), nTrain = Xbatch.n_rows;
  arma::mat prevActivation = Xbatch.t();
  arma::field<arma::mat> inputLayers(nLayers), activLayers(nLayers);

  for(int i = 0; i!=nLayers; i++){

    inputLayers[i] = weightMats[i] * prevActivation + repVec(biasVecs[i], nTrain);
    activLayers[i] = prevActivation = activFunction(inputLayers[i], activTypes[i], nSteps, smoothSteps);
  }

  return Rcpp::List::create(
    Rcpp::Named("inputLayers")    = inputLayers,
    Rcpp::Named("activLayers")    = activLayers
  );
}

Rcpp::List backwardPass(Rcpp::List upOut, Rcpp::List fpOut, arma::mat c_X, arma::mat c_y,
                        Rcpp::CharacterVector activTypes, Rcpp::String lossType, double dHuber,
                        int nSteps, int smoothSteps){

  arma::field<arma::mat> weightMats  = upOut["weightMats"];
  arma::field<arma::mat> activLayers = fpOut["activLayers"];
  arma::field<arma::mat> inputLayers = fpOut["inputLayers"];

  int nLayers = activLayers.size();
  arma::field<arma::mat> errorLayers(nLayers), weightGradMats(nLayers);
  arma::mat lossGradVec  = lossGradFunction(c_y, activLayers[nLayers-1], lossType, dHuber);
  errorLayers[nLayers-1] = lossGradVec % activGradFunction(inputLayers[nLayers-1], activTypes(nLayers-1), nSteps, smoothSteps);
  
  for(int i = nLayers-2; i!=-1; i--){
    weightGradMats[i+1] = errorLayers[i+1] * activLayers[i].t();
    errorLayers[i]      = weightMats[i+1].t() * errorLayers[i+1] % activGradFunction(inputLayers[i], activTypes(i), nSteps, smoothSteps);
  }

  weightGradMats[0] = errorLayers[0] * c_X;

  return Rcpp::List::create(
    Rcpp::Named("errorLayers")    = errorLayers,
    Rcpp::Named("weightGradMats") = weightGradMats
  );
}

Rcpp::List updateParams(Rcpp::List upOut, Rcpp::List bpOut, double momentum, double L1, double L2, double stepSize){
  arma::field<arma::mat> weightMats     = upOut["weightMats"];
  arma::field<arma::mat> weightGradMats = bpOut["weightGradMats"];
  arma::field<arma::vec> biasVecs       = upOut["biasVecs"];
  arma::field<arma::mat> errorLayers    = bpOut["errorLayers"];
  arma::field<arma::mat> weightMomMats  = upOut["weightMomMats"];
  arma::field<arma::vec> biasMomVecs    = upOut["biasMomVecs"];

  int nLayers = weightMats.size();
  for(int i = nLayers-1; i!=-1; i--){
    weightMomMats[i] = momentum * weightMomMats[i] - stepSize * weightGradMats[i];
    weightMats[i]    = (1 - stepSize * L2) * weightMats[i] - stepSize * L1 * sign(weightMats[i]) + weightMomMats[i];
    biasMomVecs[i]   = momentum * biasMomVecs[i] - stepSize * sum(errorLayers[i],1);
    biasVecs[i]      = biasVecs[i] + biasMomVecs[i];
  }

  return Rcpp::List::create(
    Rcpp::Named("biasVecs")       = biasVecs,
    Rcpp::Named("biasMomVecs")    = biasMomVecs,
    Rcpp::Named("weightMats")     = weightMats,
    Rcpp::Named("weightMomMats")  = weightMomMats
  );
}

Rcpp::List createNN(Rcpp::List upOut, bool regression, bool standardize, arma::vec y_center, arma::vec y_scale,
                    arma::vec X_center, arma::vec X_scale, Rcpp::CharacterVector y_names, Rcpp::CharacterVector activTypes, 
                    int nEpochs, arma::mat descentDetails, bool validLoss, bool plotExample, Rcpp::List fpOut, 
                    Rcpp::List bpOut, int nSteps, int smoothSteps){
  Rcpp::List NN_pred =  Rcpp::List::create(
    Rcpp::Named("biasVecs")       = upOut["biasVecs"],
    Rcpp::Named("weightMats")     = upOut["weightMats"],
    Rcpp::Named("activTypes")     = activTypes,
    Rcpp::Named("nSteps")         = nSteps,
    Rcpp::Named("smoothSteps")    = smoothSteps,
    Rcpp::Named("regression")     = regression,
    Rcpp::Named("standardize")    = standardize,
    Rcpp::Named("y_center")       = y_center,
    Rcpp::Named("y_scale")        = y_scale,
    Rcpp::Named("X_center")       = X_center,
    Rcpp::Named("X_scale")        = X_scale,
    Rcpp::Named("y_names")        = y_names
  );
  Rcpp::List NN_plot = Rcpp::List::create(
    Rcpp::Named("nEpochs")        = nEpochs,
    Rcpp::Named("descentDetails") = descentDetails,
    Rcpp::Named("validLoss")      = validLoss
  );
  if(!plotExample){
    return Rcpp::List::create(
      Rcpp::Named("NN_pred") = NN_pred,
      Rcpp::Named("NN_plot") = NN_plot
    );
  }
  Rcpp::List NN_example = Rcpp::List::create(
    Rcpp::Named("upOut")          = upOut,
    Rcpp::Named("fpOut")          = fpOut,
    Rcpp::Named("bpOut")          = bpOut,
    Rcpp::Named("nEpochs")        = nEpochs,
    Rcpp::Named("descentDetails") = descentDetails
  );
  return Rcpp::List::create(
    Rcpp::Named("NN_pred")    = NN_pred,
    Rcpp::Named("NN_plot")    = NN_plot,
    Rcpp::Named("NN_example") = NN_example
  );
}

// [[Rcpp::export]]
Rcpp::List partialForward(Rcpp::List NN, arma::mat nodesIn, bool standardizeIn, bool standardizeOut, 
                         int layerStart, int layerStop) {
  arma::field<arma::vec> biasVecs   = NN["biasVecs"];
  arma::field<arma::mat> weightMats = NN["weightMats"];
  Rcpp::CharacterVector activTypes  = NN["activTypes"];
  int nSteps                        = NN["nSteps"];
  int smoothSteps                   = NN["smoothSteps"];
  arma::mat X(size(nodesIn)), inputLayer;
  if (standardizeIn) {
    arma::vec X_center = NN["X_center"];
    arma::vec X_scale  = NN["X_scale"];
    X = scaleNewData(nodesIn, X_center, X_scale);
  } else {
    X = nodesIn;
  }
  
  int nX    = X.n_rows;
  arma::mat prevActivation = X.t();
  
  for (int i = layerStart; i!=layerStop; i++) {
    inputLayer     = weightMats[i] * prevActivation + repVec(biasVecs[i], nX);
    prevActivation = activFunction(inputLayer, activTypes[i], nSteps, smoothSteps);
  }
  
  arma::mat nodesOut(size(prevActivation.t()));
  if (standardizeOut) {
    arma::vec y_center = NN["y_center"];
    arma::vec y_scale  = NN["y_scale"];
    nodesOut = reScaleData(prevActivation.t(), y_center, y_scale);
  } else {
    nodesOut = prevActivation.t();
  }
  return Rcpp::List::create(
      Rcpp::Named("input")       = inputLayer.t(),
      Rcpp::Named("activation")  = nodesOut
  );
}

// [[Rcpp::export]]
arma::mat predictC(Rcpp::List NN, arma::mat newdata, bool standardize) {
  arma::field<arma::vec> biasVecs   = NN["biasVecs"];
  arma::field<arma::mat> weightMats = NN["weightMats"];
  Rcpp::CharacterVector activTypes  = NN["activTypes"];
  bool regression                   = NN["regression"];
  int nSteps                        = NN["nSteps"];
  int smoothSteps                   = NN["smoothSteps"];
  arma::mat X(size(newdata));
  if(standardize) {
    arma::vec X_center = NN["X_center"];
    arma::vec X_scale  = NN["X_scale"];
    X = scaleNewData(newdata, X_center, X_scale);
  } else {
    X = newdata;
  }

  int nLayers = biasVecs.size();
  int nTrain    = X.n_rows;
  arma::mat prevActivation = X.t();

  for(int i = 0; i!=nLayers; i++){
    arma::mat inputLayer = weightMats[i] * prevActivation + repVec(biasVecs[i], nTrain);
    prevActivation = activFunction(inputLayer, activTypes[i], nSteps, smoothSteps);
  }

  arma::mat prediction(size(prevActivation.t()));
  if(regression & standardize){
    arma::vec y_center = NN["y_center"];
    arma::vec y_scale  = NN["y_scale"];
    prediction = reScaleData(prevActivation.t(), y_center, y_scale);
  }else{
    prediction = prevActivation.t();
  }
  return prediction;
}

double checkLoss(arma::mat X, arma::mat y, Rcpp::List upOut, bool regression, Rcpp::CharacterVector activTypes,
                 Rcpp::String lossType, double dHuber, int nSteps, int smoothSteps){
  Rcpp::List NN =  Rcpp::List::create(
    Rcpp::Named("biasVecs")    = upOut["biasVecs"],
    Rcpp::Named("weightMats")  = upOut["weightMats"],
    Rcpp::Named("activTypes")  = activTypes,
    Rcpp::Named("regression")  = regression,
    Rcpp::Named("nSteps")      = nSteps,
    Rcpp::Named("smoothSteps") = smoothSteps
  );
  arma::mat Xpred = predictC(NN, X, FALSE);
  arma::vec XLoss = lossFunction(y, Xpred.t(), lossType, dHuber);
  return mean(XLoss);
}

Rcpp::List checkValidLoss(arma::mat X_val, arma::mat y_val, Rcpp::List upOut, arma::mat descentDetails,
                    bool regression, Rcpp::CharacterVector activTypes, bool earlyStop, int earlyStopEpochs, double earlyStopTol,
                    bool lrSched, arma::vec lrSchedEpochs, arma::vec lrSchedLearnRates, double dHuber, int nSteps,
                    int smoothSteps, int iEpoch, double learnRate, Rcpp::String lossType){
  bool DoBreak = FALSE;

  Rcpp::List NNval =  Rcpp::List::create(
    Rcpp::Named("biasVecs")    = upOut["biasVecs"],
    Rcpp::Named("weightMats")  = upOut["weightMats"],
    Rcpp::Named("activTypes")  = activTypes,
    Rcpp::Named("regression")  = regression,
    Rcpp::Named("nSteps")      = nSteps,
    Rcpp::Named("smoothSteps") = smoothSteps
  );

  arma::mat valPred    = predictC(NNval, X_val, FALSE);
  arma::vec valLoss    = lossFunction(y_val, valPred.t(), lossType, dHuber);
  arma::rowvec rowDescentDetails = descentDetails.row(iEpoch);
  rowDescentDetails[1] = mean(valLoss);

  if(earlyStop & (iEpoch > 0)){
    if(std::fmod(iEpoch, earlyStopEpochs)==0){
      arma::vec validLossVec = descentDetails.col(1);
      arma::vec earlyStopVec = validLossVec.elem(arma::linspace<arma::uvec>(iEpoch - earlyStopEpochs, iEpoch, earlyStopEpochs + 1));
      if(all(diff(earlyStopVec) > earlyStopTol)) DoBreak = TRUE;
    }
  }

  if(lrSched){
    if(any(iEpoch == lrSchedEpochs)){
      learnRate = max(lrSchedLearnRates.elem(find(lrSchedEpochs == iEpoch)));
      rowDescentDetails[3] = 1;
    }
  }
  rowDescentDetails[2] = learnRate;
  return Rcpp::List::create(
    Rcpp::Named("DoBreak")           = DoBreak,
    Rcpp::Named("learnRate")         = learnRate,
    Rcpp::Named("rowDescentDetails") = rowDescentDetails
  );
}

// [[Rcpp::export]]
Rcpp::List stochGD(Rcpp::List dataList, int nTrain, bool standardize, Rcpp::CharacterVector activTypes, 
                   Rcpp::String lossType, double dHuber, int nSteps, int smoothSteps, int batchSize, int maxEpochs, 
                   double learnRate, double momentum, double L1, double L2, bool earlyStop, int earlyStopEpochs,
                   double earlyStopTol, bool lrSched, arma::vec lrSchedEpochs, arma::vec lrSchedLearnRates, Rcpp::List fpOut,
                   Rcpp::List bpOut, Rcpp::List upOut, bool validLoss, bool verbose, bool regression, bool plotExample){
  arma::mat X = dataList["X_train"], y = dataList["y_train"], X_val = dataList["X_val"], y_val = dataList["y_val"];
  arma::vec y_center = dataList["y_center"], y_scale= dataList["y_scale"];
  arma::vec X_center = dataList["X_center"], X_scale= dataList["X_scale"]; 
  Rcpp::CharacterVector y_names = dataList["y_names"];
  arma::mat Xperm(size(X)), yperm(size(y)), descentDetails(maxEpochs,4); descentDetails.fill(0);
  arma::vec  epochLossVec(nTrain);
  arma::uvec randPerm(nTrain);
  int nBatch   = (int)std::ceil(double(nTrain)/double(batchSize));
  int nEpochs  = maxEpochs;
  bool doBreak = FALSE;
  
  for(int iEpoch = 0; iEpoch!=maxEpochs; iEpoch++){
    randPerm = sample_index(nTrain);
    Xperm    = X.rows(randPerm);
    yperm    = y.rows(randPerm);
    for(int iBatch = 0; iBatch!=nBatch; iBatch++){
      int firstRow = iBatch * batchSize;
      int lastRow  = std::min(firstRow + batchSize -1 , nTrain-1);

      arma::mat Xbatch = Xperm.rows(firstRow, lastRow);
      arma::mat ybatch = yperm.rows(firstRow, lastRow);
      
      fpOut = forwardPass(upOut, Xbatch, activTypes, nSteps, smoothSteps);
      bpOut = backwardPass(upOut, fpOut, Xbatch, ybatch, activTypes, lossType, dHuber, nSteps, smoothSteps);
      upOut = updateParams(upOut, bpOut, momentum, L1, L2, learnRate);
      
    }
    
    descentDetails(iEpoch,0) = checkLoss(X, y, upOut, regression, activTypes, lossType, dHuber, nSteps, smoothSteps);
    if(validLoss){
      Rcpp::List validLossList = checkValidLoss(X_val, y_val, upOut, descentDetails, regression, activTypes, earlyStop,
                                 earlyStopEpochs, earlyStopTol, lrSched, lrSchedEpochs, lrSchedLearnRates,
                                 dHuber, nSteps, smoothSteps, iEpoch, learnRate, lossType);
      doBreak        = validLossList["DoBreak"];
      learnRate      = validLossList["learnRate"];
      descentDetails.row(iEpoch) = Rcpp::as<arma::rowvec>(validLossList["rowDescentDetails"]);
      if(doBreak){
        if(verbose) Rcpp::Rcout << std::endl << "---> Early stop after " << iEpoch << " epochs" << std::endl;
        nEpochs = iEpoch;
        break;
      }
    }
    if(verbose){
      double percProgress = round(100*double(iEpoch)/double(maxEpochs));
      Rcpp::Rcout << "\r" << "Training: " << percProgress << "%" << std::flush;
    }
    Rcpp::checkUserInterrupt();
  }
  if (!plotExample) {
    Rcpp::Rcout << std::endl;
  }
  Rcpp::List NN = createNN(upOut, regression, standardize, y_center, y_scale, X_center, X_scale, y_names, 
                           activTypes, nEpochs, descentDetails, validLoss, plotExample, fpOut, upOut, nSteps, smoothSteps);
  return NN;
}

