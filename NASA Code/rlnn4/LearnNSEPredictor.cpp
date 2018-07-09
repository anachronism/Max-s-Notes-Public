//TODO:
///(1.) Remove testing set from NN.  Just have training set.

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>

#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/identity_output_layer.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/core/optimizers/gradient_descent/gradient_descent.hpp>
#include <mlpack/core/optimizers/adadelta/ada_delta.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>

#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/ThreeLayerNetwork.cpp"
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/TwoLayerNetwork.cpp"
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/Logging.hpp"
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/FeedForwardNetwork.cpp"
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/NeuralNetworkPredictor.cpp"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <mlpack/prereqs.hpp>
#include <fstream>
#include <boost/archive/tmpdir.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/serialization.hpp>

#include <iostream>

#ifndef LOGGING
#define LOGGING
#endif

using namespace mlpack;

namespace boost {
namespace serialization {
class access;
}
}

void circularShift(arma::colvec vecToShift,arma::colvec& retVec, int shiftNum){
	
	for (int i = 0; i < vecToShift.n_elem; i++){
		retVec[i] = vecToShift((i+shiftNum)%vecToShift.n_elem);
	}
}

arma::rowvec squaredError(arma::rowvec vector1, arma::rowvec vector2){
	arma::rowvec err = vector1 - vector2;
	return err % err;
}

template <class NetType, template <class T> class OptType>
class LearnNSEPredictor {
	public:
		std::vector<NetType *> nets;

		void initializeRMSProp(double stepSize, double alpha, double eps, size_t maxIterations, double tolerance, bool shuffle);
		void train(const arma::mat &trainData, const arma::mat &trainLabels, double trainDataFrac);
		void predict(arma::mat &inputData, arma::mat &prediction);
		void exportWeights(arma::mat &weights);
		void importWeights(arma::mat &weights);
		void loadOldRun(std::string filename);
		void saveCurrentRun(std::string filename);
		
		LearnNSEPredictor(int nNets,
						 int inputVectorSize, 
						 const std::vector<int> hiddenLayerSize, 
						 int outputVectorSize,
						 double sigmoidSlope,
						 double sigmoidThresh,
						 double errThresh
						 );
	
	private:
		std::vector<OptType<decltype(((NetType*)nullptr)->net)> *> _opts;
		arma::rowvec _mse_perfs;
		arma::mat _weights;
		arma::colvec _initWeights;
		arma::colvec netWeights;
		arma::mat beta;
		int betaInd;
		double sigmoidSlope,sigmoidThresh,errThresh;
		bool initialized;
		long trainingCounter;
		friend class boost::serialization::access;
		
		template<class Archive>
		void serialize(Archive & ar, const int version) {
			ar & _mse_perfs;
			ar & _weights;
			ar & _initWeights;
		}

		std::vector<FeedForwardNetwork *> nnFFNVec;
};


template <class NetType, template <class T> class OptType>
LearnNSEPredictor<NetType,OptType>::LearnNSEPredictor(int nNets, 
													int inputVectorSize, 
													const std::vector<int> hiddenLayerSize, 
													int outputVectorSize,
													double sigmoidSlopeTmp,
						 							double sigmoidThreshTmp,
													double errThreshTmp) 
{
	//Set LearnNSE Parameters.
	initialized = false;
	trainingCounter = 0;
	betaInd = 0;
	sigmoidThresh = sigmoidThreshTmp;
	sigmoidSlope = sigmoidSlopeTmp;
	errThresh = errThreshTmp;
	beta = arma::ones(nNets,nNets);
	netWeights = arma::ones(nNets); 
	// Intiialize list of networks.
	for(int i=0; i<nNets; i++) {
		NetType * t = new NetType(inputVectorSize,hiddenLayerSize,outputVectorSize);
		OptType<decltype(t->net)> * op = new OptType<decltype(t->net)>(t->net);
		nets.push_back(t);
		_opts.push_back(op);

		FeedForwardNetwork * pFFN = new FeedForwardNetwork;
		nnFFNVec.push_back(pFFN);
	}

	_mse_perfs.set_size(nNets);

	//initialize my NNs for LM training
	std::vector<int> networkSize;
	networkSize.resize(hiddenLayerSize.size());
	for(int i=0; i<hiddenLayerSize.size(); i++) {
		networkSize[i] = hiddenLayerSize[i];
	}
	networkSize.push_back(outputVectorSize);

	//init first network to grab initial weights
	if(hiddenLayerSize.size()==2) {
		const std::vector<std::string> actTypePerLayer = {"logsig","logsig","linear"};
		nnFFNVec[0]->initNetwork(inputVectorSize,networkSize.size(),networkSize,actTypePerLayer);
	}
	else {
		const std::vector<std::string> actTypePerLayer = {"logsig","linear"};
		nnFFNVec[0]->initNetwork(inputVectorSize,networkSize.size(),networkSize,actTypePerLayer);
	}
	nnFFNVec[0]->exportWeights(_initWeights);

	//build rest of parallel networks.  All parallel NNs have the same init weights.  the "randomness"
	//comes from uniquely shuffled train/validation data.
	if(hiddenLayerSize.size()==2) {
		const std::vector<std::string> actTypePerLayer = {"logsig","logsig","linear"};
		for(int i=1; i<nnFFNVec.size(); i++) {
			nnFFNVec[i]->initNetwork(inputVectorSize,networkSize.size(),networkSize,actTypePerLayer);
			nnFFNVec[i]->importWeights(_initWeights);
		}
	}
	else {
		const std::vector<std::string> actTypePerLayer = {"logsig","linear"};
		for(int i=1; i<nnFFNVec.size(); i++) {
			nnFFNVec[i]->initNetwork(inputVectorSize,networkSize.size(),networkSize,actTypePerLayer);
			nnFFNVec[i]->importWeights(_initWeights);
		}	
	}
}

template <class NetType, template <class T> class OptType>
void LearnNSEPredictor<NetType,OptType>::initializeRMSProp(double stepSize, double alpha, double eps, size_t maxIterations, double tolerance, bool shuffle) {
	for(int i=0; i<_opts.size(); i++) {
		_opts[i]->Alpha() = alpha;
		_opts[i]->Epsilon() = eps;
		_opts[i]->MaxIterations() = maxIterations;
		_opts[i]->Shuffle() = shuffle;
		_opts[i]->StepSize() = stepSize;
		_opts[i]->Tolerance() = tolerance;
	}
}

template <class NetType, template <class T> class OptType>
void LearnNSEPredictor<NetType,OptType>::train(const arma::mat &trainData, const arma::mat &trainLabels, double trainDataFrac){
	arma::mat prediction,prediction_eval,prediction_tmp;
	arma::mat shuffledTrainData;
	arma::mat shuffledTrainLabels;
	arma::mat shuffledValData;
	arma::mat shuffledValLabels;
	arma::colvec Dt,Bt_sampBySamp,Dt_sampBySamp, Wt, Wt_sampBySamp;
	arma::colvec weightsCol;
	arma::colvec omega;
	arma::rowvec betaVec;
	arma::colvec betaTmp;

	// arma::colvec b;
	arma::colvec sigSlopeVec,b,mseInit;
	double Bt, Et;
	double epsilon_tk,beta_hat;
	int mt,netInd,indMax,i;
	int nNets = nets.size();
	bool firstTrainingBatch;

	if(trainingCounter >= nNets)
		firstTrainingBatch = false;
	else 
		firstTrainingBatch = true;

	//resize shuffle buffers
	shuffledTrainData.set_size(trainData.n_rows,
		(int) floor(trainData.n_cols * trainDataFrac));
	shuffledTrainLabels.set_size(trainLabels.n_rows,
		(int) floor(trainLabels.n_cols * trainDataFrac));
	shuffledValData.set_size(trainData.n_rows,
		(int) (trainData.n_cols - floor(trainData.n_cols * trainDataFrac)));
	shuffledValLabels.set_size(trainLabels.n_rows,
		(int) (trainLabels.n_cols - floor(trainLabels.n_cols * trainDataFrac)));
	//shuffle data and split into train/val sets
	arma::colvec shuffledOrder = arma::regspace(0,1,trainData.n_cols -1);
	shuffledOrder = arma::shuffle(shuffledOrder);
	for(i=0; i<trainData.n_cols; i++) {
		if(i < (int) floor(trainData.n_cols * trainDataFrac)) {
			shuffledTrainData.col(i) = trainData.col(shuffledOrder(i));
			shuffledTrainLabels.col(i) = trainLabels.col(shuffledOrder(i));
		} else {
			shuffledValData.col(i-((int) floor(trainData.n_cols * trainDataFrac))) =
				trainData.col(shuffledOrder(i));
			shuffledValLabels.col(i-((int) floor(trainLabels.n_cols * trainDataFrac))) = 
				trainLabels.col(shuffledOrder(i));				
		}
	}
	

  //if net.initialized == false, net.beta = []; end
	mt = shuffledTrainData.n_cols; /***CHECK IF THIS SHOULD BE ROWS OR COLS***/
	// logFile << "mt = "<< mt <<"N cols: "<<shuffledTrainData.n_cols<<std::endl;

	Dt = arma::ones<arma::colvec>(mt)/mt; // Initialize instance weight distribution.
	Dt_sampBySamp = Dt;

	if(initialized){
		// Step 1: COmpute error of existing ensemble on new data.
		predict(shuffledTrainData,prediction);
		//compute mean squared error
		//mseInit = calcMSE(testLabels,prediction);
		mseInit = squaredError(prediction,shuffledTrainLabels) / shuffledTrainData.n_cols;
		//logFile << squaredError(prediction,shuffledTrainLabels) << "  ";
		Et = arma::accu(mseInit);
		// Get beta for each network.
		Bt = Et / (1-Et);
		Bt_sampBySamp = mseInit/(1-mseInit);
		if(Bt == 0) Bt = 1/mt;
		//Update and normalize instance weights.
		Wt = 1/mt * Bt;
		Dt = Wt/arma::accu(Wt);
		Wt_sampBySamp = 1/mt * Bt_sampBySamp;
		Dt_sampBySamp = Wt_sampBySamp/arma::accu(Wt_sampBySamp);
	}

	// Step 2: Create new classifier
	if (firstTrainingBatch)
		netInd = trainingCounter;
	else
		netInd = trainingCounter % nNets; //*****MAYBE WORK ON PROPER PRUNING*****//
	
	arma::mat weightsMat;
	exportWeights(weightsMat);
	logFile <<"F2: EXPORTED WEIGHTS"<<std::endl;
	nnFFNVec[netInd]->importWeights(_initWeights);
	predict(shuffledTrainData,prediction_tmp);
	nnFFNVec[netInd]->runLM(shuffledTrainData,shuffledTrainLabels,shuffledValData,shuffledValLabels,0.0,1e-12,1e10,500,20);
	
	/*******NOT SURE THAT WEIGHTS NEED TO BE EXPORTED**********/
	
	// nnFFNVec[netInd]->exportWeights(weightsCol);
	// if(netInd==0) {
	// 	weightsMat.set_size(weightsCol.n_elem,nets.size());
	// }
	// weightsMat.col(i) = weightsCol;
	
	// importWeights(weightsMat);

	// Step 3: Evaluate on testdata.

	predict(shuffledTrainData,prediction_eval);
	#ifdef LOGGING
	// 	/**********NOT GETTING HERE**********/
	// 	//logFile << "epsilon_tk: " << epsilon_tk << std::endl;
	// 	logFile <<"SEARCH FOR THIS"<<std::endl;
	// 	shuffledTrainLabels.print(logFile);
		logFile <<"post training:";
		prediction_eval.print(logFile);
		logFile <<"pre training:";
		prediction_tmp.print(logFile);

	#endif
	if(firstTrainingBatch)
		indMax = trainingCounter + 1;
	else
		indMax = nNets;
	float tmpFloat;

	for( i = 0; i < indMax; i++){
		epsilon_tk = arma::accu(Dt_sampBySamp % squaredError(prediction_eval,shuffledTrainLabels))/mt;
		tmpFloat = arma::accu(squaredError(prediction_eval,shuffledTrainLabels))/mt;
		#ifdef LOGGING 
		logFile <<"Search for this: " <<epsilon_tk<<" tmpfloat: "<<tmpFloat << std::endl;
		//Dt_sampBySamp.print(logFile);
		#endif
		if(epsilon_tk > 0.5){
			
			if((i<netInd && firstTrainingBatch) || (i != netInd && !firstTrainingBatch)) 
				epsilon_tk = 0.5; // if old network keep 
			else if(i == netInd) { //Retrain
				nnFFNVec[netInd]->importWeights(_initWeights);
				nnFFNVec[netInd]->runLM(shuffledTrainData,shuffledTrainLabels,shuffledValData,shuffledValLabels,0.0,1e-12,1e10,500,20);

				//arma::colvec weightsCol;
				// nnFFNVec[netInd]->exportWeights(weightsCol);
				/******************/
				// if(i==0) {
				// 	weightsMat.set_size(weightsCol.n_elem,nets.size());
				// }
				// weightsMat.col(i) = weightsCol;
				
				// importWeights(weightsMat);
			}
		}

		beta(betaInd,i) = epsilon_tk /(1-epsilon_tk);
	}
	// Step 4: Compute classifier Weights.
	if (trainingCounter == 0){
		if (beta(betaInd,trainingCounter) < errThresh)
			beta(betaInd,trainingCounter) = errThresh;

		netWeights[i] = log(1/beta(betaInd,trainingCounter));
	}
	else{
		for(i = 0; i < indMax; i++){
			if (trainingCounter < nNets)
				omega = arma::regspace(0,trainingCounter-i);
			else
				omega = arma::regspace(0,nNets - 1);


			b = (nNets - i - sigmoidThresh) * arma::ones(omega.n_elem); 
			sigSlopeVec = sigmoidSlope* arma::ones(omega.n_elem);
			omega = 1/(1+exp(sigSlopeVec % (omega - b))); //For vectors, arma overloads % to be elt multiply.
			omega = omega/arma::accu(omega);
			omega = omega.t();

			betaVec = beta.col(i);	
			if(trainingCounter < nNets)
				beta_hat = arma::sum(omega % (betaVec.subvec(i,trainingCounter))); /****NOT PROPER, NEED TO SUBSET****/
			else 
				/**************NEED TO USE CIRCULAR INDEXING*******************/
				betaVec = arma::zeros(betaVec.n_elem);
				circularShift(betaVec,betaTmp,nNets-betaInd);
				beta_hat = arma::accu(omega % betaTmp); //betaVec.span(nNets-[end-numClassifiers+1:end,i]);

			if (beta_hat < errThresh)
				beta_hat = errThresh;
			netWeights[i] = log(1/beta_hat);
		}
	}

	trainingCounter += 1;
	betaInd = (betaInd + 1) % nNets;
	// Step 5: Get error from validation set.


}



/************GOING TO HAVE TO MODIFY THIS TO USE ENSEMBLE PROPERLY************/
template <class NetType, template <class T> class OptType>
void LearnNSEPredictor<NetType,OptType>::predict(arma::mat &inputData, arma::mat &prediction) {
	arma::mat tmpPrediction;
	int nNets = nets.size();
	int indMax;
	if (trainingCounter < nNets)
		indMax = trainingCounter + 1;
	else
		indMax = nNets;
	
	for(int i=0; i<indMax; i++) {
		nets[i]->net.Predict(inputData,tmpPrediction);
		
		if(i==0) {
			prediction = tmpPrediction * netWeights[i];
		} else {
			prediction = tmpPrediction * netWeights[i] + prediction;
		}
	}
	prediction = prediction / nNets;

	//logFile << "in predict, prediction size: "<< arma::size(prediction)<<std::endl;
/*
	arma::colvec inputDataCol;
	arma::colvec tmpPredictionCol;
	for(int i=0; i<nnFFNVec.size(); i++) {
		for(int j=0; j<inputData.n_cols; j++) {
			inputDataCol = inputData.col(j);
			nnFFNVec[i]->forwardPropagate(inputDataCol,tmpPredictionCol);
			if(j==0) {
				tmpPrediction.set_size(tmpPredictionCol.n_rows,inputData.n_cols);
			}
			tmpPrediction.col(j) = tmpPredictionCol;
		}

		if(i==0) {
			prediction = tmpPrediction;
		} else {
			prediction = tmpPrediction + prediction;
		}
	}
	prediction = prediction / nets.size();
*/
}

template <class NetType, template <class T> class OptType>
void LearnNSEPredictor<NetType,OptType>::exportWeights(arma::mat &weights) {
	arma::mat tmpWeights = nets[0]->net.Parameters();
	weights.set_size(tmpWeights.n_elem,nets.size());
	for(int i=0; i<nets.size(); i++) {
		weights.col(i) = nets[i]->net.Parameters();
	}

}

template <class NetType, template <class T> class OptType>
void LearnNSEPredictor<NetType,OptType>::importWeights(arma::mat &weights) {

	for(int i=0; i<nets.size(); i++) {
		arma::mat &currentWeights = nets[i]->net.Parameters();
		currentWeights = weights.col(i);
	}

}

template <class NetType, template <class T> class OptType>
void LearnNSEPredictor<NetType,OptType>::saveCurrentRun(std::string filename) {
	exportWeights(_weights);
	std::ofstream ofs(filename);
	//save data to archvie
	boost::archive::text_oarchive oa(ofs);
	//write class instance to archive
	oa & *this;
	//archive and stream closed with destructors are called
};

template <class NetType, template <class T> class OptType>
void LearnNSEPredictor<NetType,OptType>::loadOldRun(std::string filename) {

	std::ifstream ifs(filename);
	boost::archive::text_iarchive ia(ifs);
	//read class state from archive
	ia & *this;
	//archive and stream closed with destructors are called.

	importWeights(_weights);
}
/*
int main() {
	//params
	const int nNets = 10;
	const int inputVectorSize = 2;
	const std::vector<int> hiddenLayerSize = {2,3};
	const int outputVectorSize = 1;

	const size_t maxEpochs = 10000;	

	arma::mat trainData =  { {0.0,0.0,1.0,1.0},
							 //{0.0,1.0,0.0,1.0},
							 //{0.0,0.0,1.0,1.0},
							 //{0.0,1.0,0.0,1.0},
							 //{0.0,0.0,1.0,1.0},
							 {0.0,1.0,0.0,1.0} };

	arma::mat trainLabels =  {0.0,1.0,1.0,0.0};

	arma::mat testData =  {  {0.0,0.0,1.0,1.0},
							 //{0.0,1.0,0.0,1.0},
							 //{0.0,0.0,1.0,1.0},
							 //{0.0,1.0,0.0,1.0},
							 //{0.0,0.0,1.0,1.0},
							 {0.0,1.0,0.0,1.0} };

	arma::mat testLabels =   {0.0,1.0,1.0,0.0};

	arma::mat inputLabel(2,1);
	arma::mat predOutput(1,1);

	arma::mat weights;

	//instantiate NN module
	LearnNSEPredictor<ThreeLayerNetwork,optimization::RMSprop> nnPred( nNets,
																			inputVectorSize,
																			hiddenLayerSize,
																			outputVectorSize
																		  );
	
	//init optimizer
	nnPred.initializeRMSProp(0.01,0.88,1e-8,maxEpochs*trainData.n_cols, 1e-18,true);
	//train NNs
	nnPred.train(trainData,trainLabels,testData,testLabels);
	nnPred.exportWeights(weights);
	std::cout << "trained weights: " << std::endl;
	std::cout << weights << std::endl;
	//predict NN
	inputLabel(0,0) = 0.0;
	inputLabel(1,0) = 1.0;
	nnPred.predict(inputLabel,predOutput);
	std::cout << "Input Label: " << std::endl << inputLabel << std::endl;
	std::cout << "Predicted Output: " << predOutput << std::endl;
	//import new weights
	arma::mat newWeights = arma::zeros((2*2)+(2*2)+(1*2),10);
	nnPred.importWeights(newWeights);
	nnPred.exportWeights(weights);
	std::cout << "weights (mod): " << std::endl;
	std::cout << weights << std::endl;
	
}*/

/*int main() {
	//params
	const int nNets = 100;
	const int inputVectorSize = 2;
	const std::vector<int> hiddenLayerSize = {2,2};
	const int outputVectorSize = 1;

	const size_t maxEpochs = 10000;	

	arma::mat trainData =  { {0.0,0.0,1.0,1.0},
							 //{0.0,1.0,0.0,1.0},
							 //{0.0,0.0,1.0,1.0},
							 //{0.0,1.0,0.0,1.0},
							 //{0.0,0.0,1.0,1.0},
							 {0.0,1.0,0.0,1.0} };

	arma::mat trainLabels =  {0.0,1.0,1.0,0.0};

	arma::mat testData =  {  {0.0,0.0,1.0,1.0},
							 //{0.0,1.0,0.0,1.0},
							 //{0.0,0.0,1.0,1.0},
							 //{0.0,1.0,0.0,1.0},
							 //{0.0,0.0,1.0,1.0},
							 {0.0,1.0,0.0,1.0} };

	arma::mat testLabels =   {0.0,1.0,1.0,0.0};

	arma::mat inputLabel(2,1);
	arma::mat predOutput(1,1);


	//instantiate NN module
	LearnNSEPredictor<ThreeLayerNetwork,optimization::RMSprop> nnPred( nNets,
																			inputVectorSize,
																			hiddenLayerSize,
																			outputVectorSize
																		  );
	
	//init optimizer
	nnPred.initializeRMSProp(0.01,0.88,1e-8,maxEpochs*trainData.n_cols, 1e-18,true);
	//train NNs
	nnPred.train(trainData,trainLabels,testData,testLabels);
	//predict NN
	inputLabel(0,0) = 0.0;
	inputLabel(1,0) = 1.0;
	nnPred.predict(inputLabel,predOutput);
	std::cout << "Input Label: " << std::endl << inputLabel << std::endl;
	std::cout << "Predicted Output: " << predOutput << std::endl;
}*/
/*
int main() {
	//params
	const int nNets = 100;
	const int inputVectorSize = 2;
	const std::vector<int> hiddenLayerSize = {2};
	const int outputVectorSize = 1;

	const size_t maxEpochs = 10000;	

	arma::mat trainData =  { {0.0,0.0,1.0,1.0},
							 //{0.0,1.0,0.0,1.0},
							 //{0.0,0.0,1.0,1.0},
							 //{0.0,1.0,0.0,1.0},
							 //{0.0,0.0,1.0,1.0},
							 {0.0,1.0,0.0,1.0} };

	arma::mat trainLabels =  {0.0,1.0,1.0,0.0};

	arma::mat testData =  {  {0.0,0.0,1.0,1.0},
							 //{0.0,1.0,0.0,1.0},
							 //{0.0,0.0,1.0,1.0},
							 //{0.0,1.0,0.0,1.0},
							 //{0.0,0.0,1.0,1.0},
							 {0.0,1.0,0.0,1.0} };

	arma::mat testLabels =   {0.0,1.0,1.0,0.0};

	arma::mat inputLabel(2,1);
	arma::mat predOutput(1,1);


	//instantiate NN module
	LearnNSEPredictor<TwoLayerNetwork,optimization::RMSprop> nnPred( nNets,
																		  inputVectorSize,
																		  hiddenLayerSize,
																		  outputVectorSize
																		  );
	
	//init optimizer
	nnPred.initializeRMSProp(0.01,0.88,1e-8,maxEpochs*trainData.n_cols, 1e-18,true);
	//train NNs
	nnPred.train(trainData,trainLabels,testData,testLabels);
	//predict NN
	inputLabel(0,0) = 0.0;
	inputLabel(1,0) = 1.0;
	nnPred.predict(inputLabel,predOutput);
	std::cout << "Input Label: " << std::endl << inputLabel << std::endl;
	std::cout << "Predicted Output: " << predOutput << std::endl;
}*/