#ifndef NEURALNETWORKPREDICTOR 
#define NEURALNETWORKPREDICTOR

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

#include "/home/tim/Desktop/rlnn5/ThreeLayerNetwork.cpp"
#include "/home/tim/Desktop/rlnn5/TwoLayerNetwork.cpp"
#include "/home/tim/Desktop/rlnn5/Logging.hpp"
#include "/home/tim/Desktop/rlnn5/FeedForwardNetwork.cpp"
#include "/home/tim/Desktop/rlnn5/MemoryManagement.hpp"
#include "/home/tim/Desktop/rlnn5/RecursiveLMHelper.cpp"

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
#include <boost/date_time/posix_time/posix_time.hpp>

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

		// Serialization used for saving runs after completed.
		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const int version) {
			ar & _mse_perfs;
			ar & _weights;
			ar & _initWeights;
		}

		std::vector<FeedForwardNetwork *> nnFFNVec;
};
// Initialization of NeuralNetPredictor type.
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

	//initialize NNs for LM training
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

// Initialization of RMSProp necessary for using MLPacks network. Never actually used.
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

// Train ensemble.
template <class NetType, template <class T> class OptType>
void NeuralNetworkPredictor<NetType,OptType>::train(const arma::mat &trainData, const arma::mat &trainLabels, double trainDataFrac) {
	arma::mat prediction;
	arma::mat shuffledTrainData;
	arma::mat shuffledTrainLabels;
	arma::mat shuffledValData;
	arma::mat shuffledValLabels;

	//Shuffle data
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
		//Run LM or RLM depending on configuration.
		#if LM==1
			nnFFNVec[i]->runLM(shuffledTrainData,shuffledTrainLabels,shuffledValData,shuffledValLabels,0.0,1e-12,1e10,500,20);
		
		#elif RLM==1
			nnFFNVec[i]->runRLM(shuffledTrainData,shuffledTrainLabels,shuffledValData,shuffledValLabels,0.0,1e-12,1e10,500,20);
		#endif

		//update weights in MLPack NN
		arma::colvec weightsCol;
		nnFFNVec[i]->exportWeights(weightsCol);
		if(i==0) {
			weightsMat.set_size(weightsCol.n_elem,nets.size());
		}
		weightsMat.col(i) = weightsCol;
	}

	importWeights(weightsMat);
}

// Predict with network.
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
	prediction = prediction/nets.size();
	prediction = arma::clamp(prediction,0,1);

}

/*********Functions that enable serialization***********/
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
#endif