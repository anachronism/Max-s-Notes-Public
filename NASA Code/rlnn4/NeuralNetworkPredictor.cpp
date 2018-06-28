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

using namespace mlpack;

namespace boost {
namespace serialization {
class access;
}
}

template <class NetType, template <class T> class OptType>
class NeuralNetworkPredictor {
	public:
		std::vector<NetType *> nets;

		void initializeRMSProp(double stepSize, double alpha, double eps, size_t maxIterations, double tolerance, bool shuffle);

		void train(const arma::mat &trainData, const arma::mat &trainLabels, double trainDataFrac);
		void predict(arma::mat &inputData, arma::mat &prediction);
		void exportWeights(arma::mat &weights);
		void importWeights(arma::mat &weights);
		void loadOldRun(std::string filename);
		void saveCurrentRun(std::string filename);
		
		NeuralNetworkPredictor(int nNets, int inputVectorSize, const std::vector<int> hiddenLayerSize, int outputVectorSize);
	
	private:
		std::vector<OptType<decltype(((NetType*)nullptr)->net)> *> _opts;
		arma::rowvec _mse_perfs;
		arma::mat _weights;
		arma::colvec _initWeights;

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
NeuralNetworkPredictor<NetType,OptType>::NeuralNetworkPredictor(int nNets, int inputVectorSize, const std::vector<int> hiddenLayerSize, int outputVectorSize) {
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
void NeuralNetworkPredictor<NetType,OptType>::initializeRMSProp(double stepSize, double alpha, double eps, size_t maxIterations, double tolerance, bool shuffle) {
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
void NeuralNetworkPredictor<NetType,OptType>::train(const arma::mat &trainData, const arma::mat &trainLabels, double trainDataFrac) {
	arma::mat prediction;
	arma::mat shuffledTrainData;
	arma::mat shuffledTrainLabels;
	arma::mat shuffledValData;
	arma::mat shuffledValLabels;

	//resize shuffle buffers
	shuffledTrainData.set_size(trainData.n_rows,
		(int) floor(trainData.n_cols * trainDataFrac));
	shuffledTrainLabels.set_size(trainLabels.n_rows,
		(int) floor(trainLabels.n_cols * trainDataFrac));
	shuffledValData.set_size(trainData.n_rows,
		(int) (trainData.n_cols - floor(trainData.n_cols * trainDataFrac)));
	shuffledValLabels.set_size(trainLabels.n_rows,
		(int) (trainLabels.n_cols - floor(trainLabels.n_cols * trainDataFrac)));

	//train NNs
	arma::mat weightsMat;
	for(int i=0; i<nets.size(); i++) {
		//nets[i]->net.Train(trainData, trainLabels, *_opts[i]);

		//shuffle data and split into train/val sets
		arma::colvec shuffledOrder = arma::regspace(0,1,trainData.n_cols -1);
		shuffledOrder = arma::shuffle(shuffledOrder);
		for(int i=0; i<trainData.n_cols; i++) {
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

		
		//init weights to same init values each time and train using LM
		nnFFNVec[i]->importWeights(_initWeights);
		/***********************This is the function that will be changed.***************************/
		//nnFFNVec[i]->runLM(shuffledTrainData,shuffledTrainLabels,shuffledValData,shuffledValLabels,0.0,1e-12,1e10,500,20);
		nnFFNVec[i]->runRLM(shuffledTrainData,shuffledTrainLabels,shuffledValData,shuffledValLabels,0.0,1e-12,1e10,500,20);

		//update weights in MLPack NN
		arma::colvec weightsCol;
		nnFFNVec[i]->exportWeights(weightsCol);
		if(i==0) {
			weightsMat.set_size(weightsCol.n_elem,nets.size());
		}
		weightsMat.col(i) = weightsCol;
	}

	importWeights(weightsMat);

/*	//test NNs
	for(int i=0; i<nets.size(); i++) {
		//run NN
		nets[i]->net.Predict(testData,prediction);
		
		//compute mean squared error
		_mse_perfs[i] = 0;
		for(int j=0; j<testData.n_cols; j++) {
			_mse_perfs[i] = norm(prediction.col(j)-testLabels.col(j),2) + _mse_perfs[i];
		}
		_mse_perfs[i] = _mse_perfs[i] / testData.n_cols;
	}
*/
}

template <class NetType, template <class T> class OptType>
void NeuralNetworkPredictor<NetType,OptType>::predict(arma::mat &inputData, arma::mat &prediction) {
	arma::mat tmpPrediction;
	
	for(int i=0; i<nets.size(); i++) {
		nets[i]->net.Predict(inputData,tmpPrediction);
		if(i==0) {
			prediction = tmpPrediction;
		} else {
			prediction = tmpPrediction + prediction;
		}
	}
	prediction = prediction / nets.size();
	
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
void NeuralNetworkPredictor<NetType,OptType>::exportWeights(arma::mat &weights) {
	arma::mat tmpWeights = nets[0]->net.Parameters();
	weights.set_size(tmpWeights.n_elem,nets.size());
	for(int i=0; i<nets.size(); i++) {
		weights.col(i) = nets[i]->net.Parameters();
	}

}

template <class NetType, template <class T> class OptType>
void NeuralNetworkPredictor<NetType,OptType>::importWeights(arma::mat &weights) {

	for(int i=0; i<nets.size(); i++) {
		arma::mat &currentWeights = nets[i]->net.Parameters();
		currentWeights = weights.col(i);
	}

}

template <class NetType, template <class T> class OptType>
void NeuralNetworkPredictor<NetType,OptType>::saveCurrentRun(std::string filename) {
	exportWeights(_weights);
	std::ofstream ofs(filename);
	//save data to archvie
	boost::archive::text_oarchive oa(ofs);
	//write class instance to archive
	oa & *this;
	//archive and stream closed with destructors are called
};

template <class NetType, template <class T> class OptType>
void NeuralNetworkPredictor<NetType,OptType>::loadOldRun(std::string filename) {

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
	NeuralNetworkPredictor<ThreeLayerNetwork,optimization::RMSprop> nnPred( nNets,
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
	NeuralNetworkPredictor<ThreeLayerNetwork,optimization::RMSprop> nnPred( nNets,
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
	NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop> nnPred( nNets,
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