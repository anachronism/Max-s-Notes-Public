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

#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/NeuralNetworkPredictor.cpp"
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/TrainingDataBuffer.cpp"
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/Logging.hpp"
#include "/home/max/Documents/Max-s-Notes/NASA Code/rlnn4/ApplicationSpecificHelper.cpp"

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
#include <thread>
#include <iostream>
#include <condition_variable>
#include <boost/date_time/posix_time/posix_time.hpp>


using namespace mlpack;

struct CogEngParams {
	//RLNNCognitiveEngine Params
	double cogeng_epsilonResetLim;
	double cogeng_nnExploreMaxPerfThresh;
	double cogeng_nnRejectionRate;
	double cogeng_trainFrac;
	double cogeng_pruneFrac;
	arma::rowvec cogeng_fitnessWeights;
	double cogeng_forceExploreThreshold;
	//NeuralNetworkPredictorExplore Params
	int nnExplore_nNets;
	int nnExplore_inputVectorSize;
	std::vector<int> nnExplore_hiddenLayerSizes;
	int nnExplore_outputVectorSize;
	double nnExplore_rmsProp_stepSize;
	double nnExplore_rmsProp_alpha;
	double nnExplore_rmsProp_eps;
	size_t nnExplore_rmsProp_maxEpochs;
	double nnExplore_rmsProp_tolerance;
	bool nnExplore_rmsProp_shuffle;
	//NeuralNetworkPredictor Params
	int nnExploit_nNets;
	int nnExploit_inputVectorSize;
	std::vector<int> nnExploit_hiddenLayerSizes;
	int nnExploit_outputVectorSize;
	double nnExploit_rmsProp_stepSize;
	double nnExploit_rmsProp_alpha;
	double nnExploit_rmsProp_eps;
	size_t nnExploit_rmsProp_maxEpochs;
	double nnExploit_rmsProp_tolerance;
	bool nnExploit_rmsProp_shuffle;
	//Application Specific Object Params
	int nnAppSpec_nOutVecFeatures;
	double nnAppSpec_frameSize;
	double nnAppSpec_maxEsN0;
	double nnAppSpec_minEsN0;
	double nnAppSpec_maxBER;
	arma::Row<int> nnAppSpec_modList;
	arma::Row<double> nnAppSpec_codList;
	arma::Row<double> nnAppSpec_rollOffList;
	arma::Row<double> nnAppSpec_symbolRateList;
	arma::Row<double> nnAppSpec_transmitPowerList;
	arma::Row<int> nnAppSpec_modCodList;

	//TrainingDataBuffer Params
	int buf_nTrainTestSamples;
	arma::mat buf_actionList;
};

namespace boost {
namespace serialization {
class access;
}
}

class RLNNCognitiveEngine {
	friend std::ostream & operator<<(std::ostream &os, const RLNNCognitiveEngine &rlnnCogEng);
	public:
		NeuralNetworkPredictor<ThreeLayerNetwork,optimization::RMSprop> nnExplore;
		NeuralNetworkPredictor<ThreeLayerNetwork,optimization::RMSprop> nnExploreTrainer;

		std::vector<NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop> *> nnExploit;
		std::vector<NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop> *> nnExploitTrainer;

		TrainingDataBuffer trBuf;
		ApplicationSpecificHelper appSpecObj;

		RLNNCognitiveEngine(const CogEngParams & inpParams);
		RLNNCognitiveEngine();
		int chooseAction();
		void recordResponse(int actionID, const arma::rowvec &measurementVec);

		//void loadPreviousRun(Archive & ar, const unsigned int version);
		void saveCurrentRun(std::string filename) {
			//save nn weights
			nnExplore.saveCurrentRun(filename + "_nnExplore" + ".txt");
			for(int i=0; i<nnExploit.size(); i++) {
				nnExploit[i]->saveCurrentRun(filename + "_nnExploit_" + std::to_string(i) + ".txt");
			}
			//save training buffers
			trBuf.saveCurrentRun(filename + "_trBuf" + ".txt");
			//save app specific vars
			appSpecObj.saveCurrentRun(filename + "_appSpecObj" + ".txt");

			std::ofstream ofs(filename + "_RLNNCogEng" ".txt");
			//save data to archvie
			boost::archive::text_oarchive oa(ofs);
			//write class instance to archive
			oa & *this;
			//archive and stream closed with destructors are called
		};

		void loadOldRun(std::string filename) {
			std::ifstream ifs(filename + "_RLNNCogEng" ".txt");
			boost::archive::text_iarchive ia(ifs);
			//read class state from archive
			ia & *this;
			//archive and stream closed with destructors are called.

			//load nn weights
			nnExplore.loadOldRun(filename + "_nnExplore" + ".txt");
			for(int i=0; i<nnExploit.size(); i++) {
				nnExploit[i]->loadOldRun(filename + "_nnExploit_" + std::to_string(i) + ".txt");
			}
			//save training buffers
			trBuf.loadOldRun(filename + "_trBuf" + ".txt");
			//save app specific vars
			appSpecObj.loadOldRun(filename + "_appSpecObj" + ".txt");			
		}

	private:
		double _trainFrac;
		double _pruneFrac;
		double _nnRejectionRate;
		double _epsilon;
		int _epsilonIter;
		double _epsilonResetLim;
		double _nnExploreMaxPerfThresh;
		bool _nnTrained;
		arma::mat _actionList;
		int _nActions;
		bool _exploitFlag;
		bool _forceExplore;
		bool _firstExploreAfterNNTrained;
		arma::rowvec _fitnessWeights;
		double _fitObservedMax;
		double _lastExploitFitObserved;
		arma::mat _nnExploreInputs;
		arma::mat _nnExploitInputs;
		int _histRollBackIdx;
		int _histRollBackCnt;

		bool _currentlyTraining;
		std::atomic<bool> _trainingComplete;

		std::vector<std::thread> tTrainingVec;

		double _forceExploreThreshold;

		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const int version) {
			ar & _epsilon;
			ar & _epsilonIter;
			ar & _nnTrained;
			ar & _firstExploreAfterNNTrained;
			ar & _fitObservedMax;
			ar & _lastExploitFitObserved;
			ar & _nnExploreInputs;
			ar & _nnExploitInputs;
			ar & _exploitFlag;
			ar & _forceExplore;
			ar & _histRollBackIdx;
		}
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(RLNNCognitiveEngine)

/*std::ostream & operator<<(std::ostream &os, RLNNCognitiveEngine &rlnnCogEng) {
	return os << rlnnCogEng._epsilon << rlnnCogEng._epsilonIter << rlnnCogEng._nnTrained << rlnnCogEng._firstExploreAfterNNTrained
				<< rlnnCogEng._fitObservedMax << rlnnCogEng._lastExploitFitObserved << rlnnCogEng._nnExploreInputs << rlnnCogEng._nnExploitInputs << rlnnCogEng._exploitFlag
				<<rlnnCogEng._nnExploreWeights << rlnnCogEng._nnExploitWeights;
}*/

RLNNCognitiveEngine::RLNNCognitiveEngine(const CogEngParams & inpParams)
: 	nnExplore(inpParams.nnExplore_nNets,inpParams.nnExplore_inputVectorSize,inpParams.nnExplore_hiddenLayerSizes,inpParams.nnExplore_outputVectorSize),
	nnExploreTrainer(inpParams.nnExplore_nNets,inpParams.nnExplore_inputVectorSize,inpParams.nnExplore_hiddenLayerSizes,inpParams.nnExplore_outputVectorSize),
  	trBuf(inpParams.nnAppSpec_nOutVecFeatures,inpParams.buf_nTrainTestSamples,2), //2 NNs (explore and exploit)
  	appSpecObj(	inpParams.buf_actionList,
  				inpParams.nnAppSpec_frameSize,
  			   	inpParams.nnAppSpec_maxEsN0,
  			   	inpParams.nnAppSpec_minEsN0,
  			   	inpParams.nnAppSpec_modList,
  			   	inpParams.nnAppSpec_codList,
  			   	inpParams.nnAppSpec_rollOffList,
  			   	inpParams.nnAppSpec_symbolRateList,
  			   	inpParams.nnAppSpec_transmitPowerList,
  			   	inpParams.nnAppSpec_modCodList,
  			   	inpParams.nnAppSpec_maxBER)
{

	//RLNNCognitiveEngine
	_epsilonResetLim = inpParams.cogeng_epsilonResetLim;
	_nnExploreMaxPerfThresh = inpParams.cogeng_nnExploreMaxPerfThresh;
	_nnRejectionRate = inpParams.cogeng_nnRejectionRate;
	_trainFrac = inpParams.cogeng_trainFrac;
	_pruneFrac = inpParams.cogeng_pruneFrac;
	_forceExploreThreshold = inpParams.cogeng_forceExploreThreshold;

	_epsilon = 1.0;
	_epsilonIter = 1;
	_nnTrained = false;
	_firstExploreAfterNNTrained = false;
	_actionList = inpParams.buf_actionList;
	_nActions = _actionList.n_cols;
	_exploitFlag = false;
	_fitnessWeights = inpParams.cogeng_fitnessWeights;
	_fitObservedMax = 0.0;
	_lastExploitFitObserved = 0.0;
	_forceExplore = true;
	_histRollBackIdx = -1;
	_histRollBackCnt = 0;

	_currentlyTraining = false;
	_trainingComplete = false;

	//NeuralNetworkPredictors
	//configure RMS prop for nnExplore
	nnExplore.initializeRMSProp(inpParams.nnExplore_rmsProp_stepSize,
							 inpParams.nnExplore_rmsProp_alpha,
							 inpParams.nnExplore_rmsProp_eps,
							 inpParams.nnExplore_rmsProp_maxEpochs*inpParams.buf_nTrainTestSamples,
							 inpParams.nnExplore_rmsProp_tolerance,
							 inpParams.nnExplore_rmsProp_shuffle
							 );
	nnExploreTrainer.initializeRMSProp(inpParams.nnExplore_rmsProp_stepSize,
							 inpParams.nnExplore_rmsProp_alpha,
							 inpParams.nnExplore_rmsProp_eps,
							 inpParams.nnExplore_rmsProp_maxEpochs*inpParams.buf_nTrainTestSamples,
							 inpParams.nnExplore_rmsProp_tolerance,
							 inpParams.nnExplore_rmsProp_shuffle
							 );
	//generate array of NNs for nnExploit (one for each type of action)
	for(int i=0; i<_actionList.n_rows; i++) {
		NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop> * pNNExploit =
			new NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop>(
				inpParams.nnExploit_nNets,
				inpParams.nnExploit_inputVectorSize,
				inpParams.nnExploit_hiddenLayerSizes,
				inpParams.nnExploit_outputVectorSize
				);
		pNNExploit->initializeRMSProp(inpParams.nnExploit_rmsProp_stepSize,
							 inpParams.nnExploit_rmsProp_alpha,
							 inpParams.nnExploit_rmsProp_eps,
							 inpParams.nnExploit_rmsProp_maxEpochs*inpParams.buf_nTrainTestSamples,
							 inpParams.nnExploit_rmsProp_tolerance,
							 inpParams.nnExploit_rmsProp_shuffle
							 );		
		nnExploit.push_back(pNNExploit);

		NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop> * pNNExploitTrainer =
			new NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop>(
				inpParams.nnExploit_nNets,
				inpParams.nnExploit_inputVectorSize,
				inpParams.nnExploit_hiddenLayerSizes,
				inpParams.nnExploit_outputVectorSize
				);
		pNNExploitTrainer->initializeRMSProp(inpParams.nnExploit_rmsProp_stepSize,
							 inpParams.nnExploit_rmsProp_alpha,
							 inpParams.nnExploit_rmsProp_eps,
							 inpParams.nnExploit_rmsProp_maxEpochs*inpParams.buf_nTrainTestSamples,
							 inpParams.nnExploit_rmsProp_tolerance,
							 inpParams.nnExploit_rmsProp_shuffle
							 );		
		nnExploitTrainer.push_back(pNNExploitTrainer);
	}

}

int RLNNCognitiveEngine::chooseAction() {
	double eProb;
	arma::mat predictions;
	std::vector<arma::mat> predictedActionVec;
	double perfThresh;
	int actionID;

	//set epsilon, nearbyhopidx
	if(_epsilon > _epsilonResetLim) {
		_epsilon = 1.0/((double) _epsilonIter);
		_epsilonIter++;
	} else { //reset epsilon
		_epsilon = 1.0;
		_epsilonIter=1;
	}

	//choose action
	if(_nnTrained && !_forceExplore) {
		eProb = math::Random();
		if(eProb <= _epsilon) { //exploration mode
			//generate the input vectors for prediction
			std::cout << "Exploring" << std::endl;
			#ifdef LOGGING
			logFile << "::Action Type: Exploring " << std::endl;
			#endif
			appSpecObj.genNNExploreInputs(_nnExploreInputs);
			//predict using input vectors. returns mean of all
			//parallel nets
			nnExplore.predict(_nnExploreInputs, predictions);
			//calculate current max threshold
			perfThresh = predictions.max() * (_nnExploreMaxPerfThresh);
			//if first explore after NN trained, then choose max perf.
			//randomly choose one of the actions (based on rejection rate)
			//and perfThresh
			if(_firstExploreAfterNNTrained) {
				std::cout << "Random Explore" << std::endl;
				actionID = predictions.index_max();
				_firstExploreAfterNNTrained = false;
			} else {
				arma::uvec idxsLessThresh = arma::find(predictions < perfThresh);
				arma::uvec idxsGreatThresh = arma::find(predictions >= perfThresh);
				//std::cout << "idxs: " << idxsLessThresh.n_elem << " " << idxsGreatThresh.n_elem << std::endl;			

				//check edge case: if true, randomly choose (as they are all in the same
				//subset). else choose off of rejection rate.
				if((idxsLessThresh.n_elem==0) || (idxsGreatThresh.n_elem==0)) {
					std::cout << "Random Explore" << std::endl;
					actionID = _nActions;
					while(actionID >= _nActions) {
						actionID = (int) floor((double) _nActions*math::Random());
					}				
				} else {
					if(math::Random() >= _nnRejectionRate) {
						//std::cout << "Less Than Threshold Explore" << std::endl;
						int idx = idxsLessThresh.n_elem;
						while(idx >= idxsLessThresh.n_elem) {
							idx = (int) floor((double) idxsLessThresh.n_elem*math::Random());
						}
						actionID = idxsLessThresh(idx);
					} else { //choose action better than perfThresh
						//std::cout << "Greater Than Threshold Explore" << std::endl;
						int idx = idxsGreatThresh.n_elem;
						while(idx >= idxsGreatThresh.n_elem) {
							idx = (int) floor((double) idxsGreatThresh.n_elem*math::Random());
						}
						actionID = idxsGreatThresh(idx);
					}					
				}
			}
			_exploitFlag = false;
		} else { //exploit mode
			//generate the input vectors for prediction
			std::cout << "Exploiting" << std::endl;
			#ifdef LOGGING
			logFile << "::Action Type: Exploiting " << std::endl;
			#endif
			appSpecObj.genNNExploitInputs(_nnExploitInputs);
			//predict using input vectors. returns mean of all
			//parallel nets
			//std::cout << "nnExploit Inputs: " << _nnExploitInputs << std::endl;
			//std::cout << "nnExploit Outputs: " << std::endl;
			//std::cout << "nnExploit.size(): " << nnExploit.size() << std::endl;
			for(int i=0; i<nnExploit.size(); i++) {
				nnExploit[i]->predict(_nnExploitInputs, predictions);
				predictedActionVec.push_back(predictions);
				//std::cout << predictions << std::endl;
			}
			//std::cout << std::endl;
			//the outputs of the nn's are processed using the application
			//specific object and the actionID corresponding to their outputs
			//is returned back.
			actionID = appSpecObj.processNNExploitOutputs(predictedActionVec);

			_exploitFlag = true;
		}
	} else {
		//choose random action

		if(_currentlyTraining) {
			#ifdef LOGGING
			logFile << "::Action Type: Exploiting " << std::endl;
			#endif

			std::cout << "Exploiting" << std::endl;
			actionID = appSpecObj.returnFallBackAction();
			_exploitFlag = true;
		} else {
			#ifdef LOGGING
			logFile << "::Action Type: Exploring " << std::endl;
			#endif

			std::cout << "Random Explore" << std::endl;
			actionID = _nActions;
			while(actionID >= _nActions) {
				actionID = (int) floor((float) _nActions*math::Random());
			}
			_exploitFlag = false;
		}
	}

/*
	#ifdef LOGGING
	logFile << "Action Chosen: " << actionID << std::endl;
	if(_exploitFlag) {
		logFile << "Action Type: " << "Exploit" << std::endl;
	} else {
		logFile << "Action Type: " << "Explore" << std::endl;
	}
	#endif
*/
	return actionID;

}

void RLNNCognitiveEngine::recordResponse(int actionID, const arma::rowvec &measurementVec) {
	bool bufferFull;
	arma::rowvec fitParams;
	arma::colvec outVec;
	double fitObserved;

	//process and save measurements from receiver
	appSpecObj.processMeasurements(measurementVec);

	//compute new fitness observed
	appSpecObj.getFitnessParams(fitParams);
	//ONLY HERE FOR DEMO PURPOSES!!!
	std::cout << "Normalized Objective Scores:" << std::endl;
	std::cout << "Throughput: " << fitParams(0) << std::endl;
	std::cout << "BER: " << fitParams(1) << std::endl;
	std::cout << "Target BW: " << fitParams(2) << std::endl;
	std::cout << "Spectral Efficiency: " << fitParams(3) << std::endl;
	std::cout << "TX Power Efficiency: " << fitParams(4) << std::endl;
	std::cout << "DC Power Consumed: " << fitParams(5) << std::endl;
	//std::cout << "fitParams" << std::endl;
	//std::cout << fitParams << std::endl;
	fitObserved = arma::dot(_fitnessWeights,fitParams);
	std::cout << "Multiobjective Fitness Score: " << fitObserved << std::endl;

	#ifdef LOGGING
//	logFile << boost::posix_time::microsec_clock::local_time() << std::endl;
	logFile << "::Objective Fitnesses Observed: ";
	logFile << fitParams;
	logFile << "::Fitness Observed: " << fitObserved <<std::endl;
	#endif

	appSpecObj.setFitnessObserved(fitObserved,_exploitFlag);

	//Adapt/updates NN_exploit input whenever forced exploration finds a better performance
	if(_forceExplore) {
		if(trBuf.getBufCntr()==0) {
			_lastExploitFitObserved = fitObserved;
		} else { //bufCtnr>0
			if(fitObserved > _lastExploitFitObserved) {
				_lastExploitFitObserved = fitObserved;
			}
		}
	}

	//update nnExploit input whenever exploration finds a better performance
	if(fitObserved >= _fitObservedMax) { //we found a new performance maximum
		std::cout << "We found a new fObserved max." << std::endl;
//		#ifdef LOGGING
//		logFile << boost::posix_time::microsec_clock::local_time() << std::endl;
//		logFile << "::New Max Fitness Observed" <<std::endl;
//		logFile << std::endl;
//		#endif
		_fitObservedMax = fitObserved;
		if(!_exploitFlag) {
			appSpecObj.updateNNExploitInputs();
		}
	} else { //not a maximum
//		std::cout << "We found a worse fObserved." << std::endl;
		if(_exploitFlag) {
			if(fitObserved<_lastExploitFitObserved) {
				//Reset "More Efficient Mode". Threshold value is a designer parameter (0.5 for specific missions, 0.1 for general)
				if(  (((_lastExploitFitObserved-fitObserved)>_forceExploreThreshold) &&
					 (appSpecObj.lastAndNewNNExploitInputEqual()) &&
					 (appSpecObj.lastNNExploitInputEmpty()==false)) 
					 ||
					 (((_histRollBackCnt>=trBuf.getBufferSize()) &&
					 (appSpecObj.lastAndNewNNExploitInputEqual()) &&
					 (appSpecObj.lastNNExploitInputEmpty()==false))) 

					&& (!_currentlyTraining)
				  ) {
				  	//std::cout <<"HERE1"<<std::endl;
				  	_forceExplore = 1; //enter explore mode
				  	_fitObservedMax = 0; //reset max tracking
				  	trBuf.resetBuffer(); //reset NN history
				  	_histRollBackCnt = 0;
				}
				// Quick "recover mode" using performances from the buffer.  Triggers when 90% below previous exploration level
				else if (fitObserved < _lastExploitFitObserved*0.9 || /*fitObserved < _fitObservedMax*0.9*/ fitObserved < appSpecObj.getRollBackThreshold()) {
					//std::cout <<"HERE2"<<std::endl;
					_histRollBackIdx++; //increment ptr
					_histRollBackCnt++;
					if(_histRollBackIdx==(trBuf.getBufCntr()-1)) { //wrap ptr around
						_histRollBackIdx=0;
					}

					//get column in training buffer sorted by fitObserved, ascending order.
					TrainingDataBuffer::InpOutBuffParams *par;
					arma::colvec trainingInCol;
					arma::colvec trainingOutCol;
					par = appSpecObj.getNNInpOutBuffParms(1); //get exploit NN params
					trBuf.buildTrainingColumn(1,_actionList,
									   par->inpBuffIdxs,par->inpBuffExpandActions,par->inpNormParams,
									   par->outBuffIdxs,par->outBuffExpandActions,par->outNormParams,
									   appSpecObj.getFitObservedOutVecIdx(),(trBuf.getBufferSize()-1-_histRollBackIdx),
									   trainingInCol, trainingOutCol);
					//force update nnExploit Input
					appSpecObj.forceSetNNExploitInputs(trainingInCol);
				}
				//accepts new exploitation peformance 90% above last exploittation threshold
				else if(fitObserved > _lastExploitFitObserved*0.9 && _histRollBackIdx>-1) {
					//std::cout <<"HERE3"<<std::endl;
					_lastExploitFitObserved = fitObserved;
					appSpecObj.updateLastNNExploitInputs();
				  	_histRollBackCnt = 0;
				}
				//if exploiting and current exploitation performance is worse than previous exploitation perf;
				//roll back nn exploit input
				else {
					//std::cout <<"HERE4"<<std::endl;
					appSpecObj.rollBackExploitInputs();
				  	_histRollBackCnt = 0;
				}
			} else { //update last exploitation performance
				//std::cout <<"HERE5"<<std::endl;
				_lastExploitFitObserved = fitObserved;
				appSpecObj.updateLastNNExploitInputs();
				_histRollBackCnt = 0;
			}
		}
	}

		//	if(fitObserved < _lastExploitFitObserved) {
		//		appSpecObj.rollBackExploitInputs();
		//	} else {
		//		_lastExploitFitObserved = fitObserved;
		//		appSpecObj.updateLastNNExploitInputs();
		//	}
		//}
	//}

	//push action/result into training data buffer
	appSpecObj.genTrainingSample(outVec);
	bufferFull = trBuf.addTrainingSample(actionID, outVec);

	//train NNs
	if(bufferFull) {

//		if(_nnTrained==true && _forceExplore==false) {
//		if(false) {
		if(true) {
			if(!_currentlyTraining) {
				std::cout << "Training Occurring" << std::endl;
				_currentlyTraining = true;
				_trainingComplete = false;

				//build the nnExplore(0) and nnExploit(1) training sets
				std::vector<arma::mat> * nnInputs;
				std::vector<arma::mat> * nnOutputs;
				std::cout << "Buffer Full Thread" << std::endl;
				for(int i=0; i<2; i++) {
					TrainingDataBuffer::InpOutBuffParams *par;
					par = appSpecObj.getNNInpOutBuffParms(i);
					trBuf.buildTrainingSet(i,_actionList,
									   par->inpBuffIdxs,par->inpBuffExpandActions,par->inpNormParams,
									   par->outBuffIdxs,par->outBuffExpandActions,par->outNormParams
									   );
				}
				//retrieve the buffers
				nnInputs = trBuf.getTrainTestInput();
				nnOutputs = trBuf.getTrainTestOutput();


				tTrainingVec.push_back(std::thread ([this,nnInputs,nnOutputs](){
					
					//TRAIN NN_EXPLORE
					//std::cout << "Training NN Explore" << std::endl;
					//nnExploreTrainer.train(shuffledTrainData,shuffledTrainLabels,
					nnExploreTrainer.train((*nnInputs)[0],(*nnOutputs)[0],_trainFrac);			
					//nnExplore.train((*nnInputs)[0],(*nnOutputs)[0],_trainFrac);		
						
					//train NN array
					//std::cout << "Training NN Exploit" << std::endl;
					for(int i=0; i<nnExploit.size(); i++) {
						nnExploitTrainer[i]->train((*nnInputs)[1],(*nnOutputs)[1].row(i),_trainFrac);
						//nnExploit[i]->train((*nnInputs)[1],(*nnOutputs)[1].row(i),_trainFrac);
					}

					_trainingComplete = true;
				}));

			}
			//join thread
			//tTraining.join();

			if(_trainingComplete) {
				//join thread
				tTrainingVec[0].join();
				tTrainingVec.pop_back();
				_currentlyTraining = false;

				//transfer weights
				arma::mat trainedWeights;
				nnExploreTrainer.exportWeights(trainedWeights);
				nnExplore.importWeights(trainedWeights);
				for(int i=0; i<nnExploit.size(); i++) {
					nnExploitTrainer[i]->exportWeights(trainedWeights);
					nnExploit[i]->importWeights(trainedWeights);
				}	

				//flag as initially trained
				_nnTrained = true;
				_forceExplore = false; //allow us to use NNs
				_histRollBackIdx = 0;
				_firstExploreAfterNNTrained=true;
				//prune buffer
				trBuf.pruneDataBuffer(_pruneFrac);

			} else {
				std::cout << "Training Occurring" << std::endl;

				#ifdef LOGGING
				logFile << "::Training: Yes" <<std::endl;
				#endif
			}
		} else { /*************CHECK IF THIS IS EVER VISITED****************/
			std::cout << "Training Occurring" << std::endl;
			#ifdef LOGGING
			logFile << "::Training: Yes" <<std::endl;
			#endif
			std::vector<arma::mat> * nnInputs;
			std::vector<arma::mat> * nnOutputs;

			std::cout << "Buffer Full" << std::endl;
			//build the nnExplore(0) and nnExploit(1) training sets
			for(int i=0; i<2; i++) {
				TrainingDataBuffer::InpOutBuffParams *par;
				par = appSpecObj.getNNInpOutBuffParms(i);
				trBuf.buildTrainingSet(i,_actionList,
								   par->inpBuffIdxs,par->inpBuffExpandActions,par->inpNormParams,
								   par->outBuffIdxs,par->outBuffExpandActions,par->outNormParams
								   );
			}
			//retrieve the buffers
			nnInputs = trBuf.getTrainTestInput();
			nnOutputs = trBuf.getTrainTestOutput();

			//TRAIN NN_EXPLORE
			std::cout << "Training NN Explore" << std::endl;
			//nnExploreTrainer.train(shuffledTrainData,shuffledTrainLabels,
			nnExploreTrainer.train((*nnInputs)[0],(*nnOutputs)[0],_trainFrac);			
			//nnExplore.train((*nnInputs)[0],(*nnOutputs)[0],_trainFrac);			

			//train NN array
			std::cout << "Training NN Exploit" << std::endl;
			for(int i=0; i<nnExploit.size(); i++) {
				nnExploitTrainer[i]->train((*nnInputs)[1],(*nnOutputs)[1].row(i),_trainFrac);
				//nnExploit[i]->train((*nnInputs)[1],(*nnOutputs)[1].row(i),_trainFrac);
			}

			//transfer weights
			arma::mat trainedWeights;
			nnExploreTrainer.exportWeights(trainedWeights);
			nnExplore.importWeights(trainedWeights);
			for(int i=0; i<nnExploit.size(); i++) {
				nnExploitTrainer[i]->exportWeights(trainedWeights);
				nnExploit[i]->importWeights(trainedWeights);
			}		
			
			//flag as initially trained
			_nnTrained = true;
			_forceExplore = false; //allow us to use NNs
			_histRollBackIdx = 0;
			_firstExploreAfterNNTrained=true;
			//prune buffer
			trBuf.pruneDataBuffer(_pruneFrac);
		}			
	} else {
		#ifdef LOGGING
		logFile << "::Training: No" <<std::endl;
		#endif
	}

/*	#ifdef LOGGING
	logFile << "NN Training Occurred: " << bufferFull << std::endl;
	#endif
*/
}

//void RLNNCognitiveEngine::loadPreviousRun(Archive & ar, const unsigned int version) {
//
//}


/*
int main() {
	int actionID;

	arma::mat actionList;

	//---------------------------------------------------//
	//input parameters
	std::cout<<"Setting Cog Engine Parameters" << std::endl;
	CogEngParams cogEngParams;
	arma::rowvec fitnessWeights = {1.0/6.0, 1.0/6.0, 1.0/6.0,
									1.0/6.0, 1.0/6.0, 1.0/6.0};

	//RLNNCognitiveEngine Params
	cogEngParams.cogeng_epsilonResetLim = 4e-3;
	cogEngParams.cogeng_nnExploreMaxPerfThresh = 0.9;
	cogEngParams.cogeng_nnRejectionRate = 0.95;
	cogEngParams.cogeng_trainFrac = 0.9;
	cogEngParams.cogeng_pruneFrac = 0.5;
	cogEngParams.cogeng_fitnessWeights = fitnessWeights;

	//NeuralNetworkPredictor Params
	cogEngParams.nnExplore_nNets = 20;
	cogEngParams.nnExplore_inputVectorSize = 7;
	cogEngParams.nnExplore_hiddenLayerSizes.push_back(7);
	cogEngParams.nnExplore_hiddenLayerSizes.push_back(50);	
	cogEngParams.nnExplore_outputVectorSize = 1;
	cogEngParams.nnExplore_rmsProp_stepSize = 0.01;
	cogEngParams.nnExplore_rmsProp_alpha = 0.88;
	cogEngParams.nnExplore_rmsProp_eps = 1e-8;
	cogEngParams.nnExplore_rmsProp_maxEpochs = 10;
	cogEngParams.nnExplore_rmsProp_tolerance = 1e-18;
	cogEngParams.nnExplore_rmsProp_shuffle = true;
	cogEngParams.nnExploit_nNets = 10;
	cogEngParams.nnExploit_inputVectorSize = 7;
	cogEngParams.nnExploit_hiddenLayerSizes.push_back(20);	
	cogEngParams.nnExploit_outputVectorSize = 1;
	cogEngParams.nnExploit_rmsProp_stepSize = 0.01;
	cogEngParams.nnExploit_rmsProp_alpha = 0.88;
	cogEngParams.nnExploit_rmsProp_eps = 1e-8;
	cogEngParams.nnExploit_rmsProp_maxEpochs = 10;
	cogEngParams.nnExploit_rmsProp_tolerance = 1e-18;
	cogEngParams.nnExploit_rmsProp_shuffle = true;
	//Application Specific Object Params
	cogEngParams.nnAppSpec_nOutVecFeatures = 8;
	cogEngParams.nnAppSpec_frameSize = 16200.0;
	cogEngParams.nnAppSpec_maxEsN0 = 12.93; //dB
	cogEngParams.nnAppSpec_maxBER = pow(10,-12);
	cogEngParams.nnAppSpec_modList <<4<<4<<4<<4<<4<<4<<4<<4<<4
								  <<4<<4<<8<<8<<8<<8<<8<<8
								  <<16<<16<<16<<16<<16<<16
								  <<32<<32<<32<<32<<32
								  <<arma::endr;
	cogEngParams.nnAppSpec_codList <<(1.0/4.0)<<(1.0/3.0)<<(2.0/5.0)<<(1.0/2.0)
									<<(3.0/5.0)<<(2.0/3.0)<<(3.0/4.0)<<(4.0/5.0)
									<<(5.0/6.0)<<(8.0/9.0)<<(9.0/10.0)
									<<(3.0/5.0)<<(2.0/3.0)<<(3.0/4.0)<<(5.0/6.0)
									<<(8.0/9.0)<<(9.0/10.0)
								  	<<(2.0/3.0)<<(3.0/4.0)<<(4.0/5.0)<<(5.0/6.0)
								  	<<(8.0/9.0)<<(9.0/10.0)
								  	<<(3.0/4.0)<<(4.0/4.0)<<(5.0/6.0)<<(8.0/9.0)
								  	<<(9.0/10.0)
								  	<<arma::endr;
	cogEngParams.nnAppSpec_rollOffList << 0.2 << 0.3 << 0.35<<arma::endr;
	double RsMin = 0.5*pow(10,6)/(1+cogEngParams.nnAppSpec_rollOffList.min());//0.5
	double RsMax = 5*pow(10,6)/(1+cogEngParams.nnAppSpec_rollOffList.max());
	cogEngParams.nnAppSpec_symbolRateList = arma::trans(arma::regspace(RsMin,0.1*pow(10,6),RsMax)); //0.1MHz spacing
	cogEngParams.nnAppSpec_transmitPowerList = arma::trans(arma::regspace(0.0,1.0,10.0)); //1.0 dB spacing
	//TrainingDataBuffer Params
	cogEngParams.buf_nTrainTestSamples = 200;

	actionList.set_size(6,cogEngParams.nnAppSpec_symbolRateList.n_elem
							*cogEngParams.nnAppSpec_transmitPowerList.n_elem
							*cogEngParams.nnAppSpec_modList.n_elem
							*cogEngParams.nnAppSpec_rollOffList.n_elem);
	int id = 0;
	for(int i1=0; i1<cogEngParams.nnAppSpec_symbolRateList.n_elem; i1++) {
		for(int i2=0; i2<cogEngParams.nnAppSpec_transmitPowerList.n_elem; i2++) {
			for(int i3=0; i3<cogEngParams.nnAppSpec_modList.n_elem; i3++) {
				for(int i4=0; i4<cogEngParams.nnAppSpec_rollOffList.n_elem; i4++) {
						actionList(0,id) = cogEngParams.nnAppSpec_symbolRateList(i1);
						actionList(1,id) = cogEngParams.nnAppSpec_transmitPowerList(i2);
						actionList(2,id) = (double)i3; //modcod id
						actionList(3,id) = cogEngParams.nnAppSpec_rollOffList(i4);
						id++;
				}
			}
		}
	}
	for(int i=0; i<actionList.n_cols; i++) {
		actionList(4,i) = (double) cogEngParams.nnAppSpec_modList((int)actionList(2,i));
		actionList(5,i) = cogEngParams.nnAppSpec_codList((int)actionList(2,i));
		actionList(2,i) = log2(actionList(4,i));
	}


	cogEngParams.buf_actionList = actionList;

	//---------------------------------------------------//
	//instantiate cog engine
	std::cout<<"Instantiating Cog Engine" << std::endl;
	RLNNCognitiveEngine rlnnCogEng(cogEngParams);

	//choose an action
	arma::rowvec measurementVec(6);
	int i=0;
	for(int i=0; i<20000; i++) {
		std::cout<<i<<": Choosing Action"<<std::endl;
		actionID = rlnnCogEng.chooseAction();
		std::cout<<i<<": Action Chosen: " << actionID << std::endl;

		//apply action to environment
		measurementVec(0) = 6; //5.5-6.5 dB + transmit power
		measurementVec(1) = actionList(1,actionID); //tx power
		measurementVec(2) = actionList(0,actionID); //Rs
		measurementVec(3) = actionList(3,actionID); //rolloff
		measurementVec(4) = actionList(4,actionID); //mod
		measurementVec(5) = actionList(5,actionID); //cod
		//we don't need to measure log2(M) since we have M.
		std::cout <<i<<": measurementVec: " << std::endl;
		std::cout << measurementVec << std::endl;

		//record response of environment
		rlnnCogEng.recordResponse(actionID, measurementVec);
	}
}

8/
/*int main() {
	int actionID;

	//---------------------------------------------------//
	//input parameters
	CogEngParams cogEngParams;
	arma::mat actionList(6,500,arma::fill::randu);



	arma::rowvec fitnessWeights = {0.25, 0.25, 0.25, 0.25};

	//RLNNCognitiveEngine Params
	cogEngParams.cogeng_epsilonResetLim = 4e-3;
	cogEngParams.cogeng_nnPerfThresh = 0.5; //0.65
	cogEngParams.cogeng_nnRejectionRate = 1;
	cogEngParams.cogeng_learningRateAlphaResetLim = 1e-3;
	cogEngParams.cogeng_trainFrac = 0.9;
	cogEngParams.cogeng_pruneFrac = 0.5;

	//NeuralNetworkPredictor Params
	cogEngParams.nn_nNets = 100;
	cogEngParams.nn_hiddenLayerSize = 6;
	cogEngParams.nn_outputVectorSize = 1;
	cogEngParams.nn_setAlwaysChooseMinNN = true;

	cogEngParams.nn_rmsProp_stepSize = 0.01;
	cogEngParams.nn_rmsProp_alpha = 0.88;
	cogEngParams.nn_rmsProp_eps = 1e-8;
	cogEngParams.nn_rmsProp_maxEpochs = 10;
	cogEngParams.nn_rmsProp_tolerance = 1e-18;
	cogEngParams.nn_rmsProp_shuffle = true;
	//Reinforcement Learner Params
	cogEngParams.rl_nPerfValues = 200;
	cogEngParams.rl_fitnessWeights = fitnessWeights;
	cogEngParams.rl_rewardThreshold = 0;
	//TrainingDataBuffer Params
	cogEngParams.buf_nTrainTestSamples = 100;
	cogEngParams.buf_actionList = actionList;

	//---------------------------------------------------//
	//instantiate cog engine
	RLNNCognitiveEngine rlnnCogEng(cogEngParams);
	arma::rowvec fitParams(4);

	//choose an action
	for(int i=0; i<21600; i++) {
		actionID = rlnnCogEng.chooseAction();
		std::cout << "Action Chosen: " << actionID << std::endl;

		//apply action to environment
		fitParams.randu();

		//record response of environment
		rlnnCogEng.recordResponse(actionID,fitParams);
	}
}
*/