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

#include "/home/tim/Desktop/rlnn5/NeuralNetworkPredictor.cpp"
#include "/home/tim/Desktop/rlnn5/LearnNSEPredictor.cpp"
#include "/home/tim/Desktop/rlnn5/TrainingDataBuffer.cpp"
#include "/home/tim/Desktop/rlnn5/Logging.hpp"
#include "/home/tim/Desktop/rlnn5/MemoryManagement.hpp"
#include "/home/tim/Desktop/rlnn5/ApplicationSpecificHelper.cpp"

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
#include <chrono>

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
	//Learn NSE parameters
	#if NSE==1
		double sigmoidSlope;
		double sigmoidThresh;
		double errorThresh;
	#endif
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
		#if NSE==1
			LearnNSEPredictor<ThreeLayerNetwork,optimization::RMSprop> nnExplore;
			LearnNSEPredictor<ThreeLayerNetwork,optimization::RMSprop> nnExploreTrainer;
			
			std::vector<LearnNSEPredictor<TwoLayerNetwork,optimization::RMSprop> *> nnExploit;
			std::vector<LearnNSEPredictor<TwoLayerNetwork,optimization::RMSprop> *> nnExploitTrainer;
		#elif LM==1
			NeuralNetworkPredictor<ThreeLayerNetwork,optimization::RMSprop> nnExplore;
			NeuralNetworkPredictor<ThreeLayerNetwork,optimization::RMSprop> nnExploreTrainer;
		
			std::vector<NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop> *> nnExploit;
			std::vector<NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop> *> nnExploitTrainer;
		#elif RLM==1
			NeuralNetworkPredictor<ThreeLayerNetwork,optimization::RMSprop> nnExplore;
			NeuralNetworkPredictor<ThreeLayerNetwork,optimization::RMSprop> nnExploreTrainer;
		
			std::vector<NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop> *> nnExploit;
			std::vector<NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop> *> nnExploitTrainer;
		#endif 

		

		TrainingDataBuffer trBuf;
		ApplicationSpecificHelper appSpecObj;

		RLNNCognitiveEngine(const CogEngParams & inpParams);
		RLNNCognitiveEngine();
		int chooseAction();
		void recordResponse(int actionID, const arma::rowvec &measurementVec);

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
			//save data to archive
			boost::archive::text_oarchive oa(ofs);
			//write class instance to archive
			oa & *this;
			//archive and stream closed when destructors are called
		};

		void loadOldRun(std::string filename) {
			std::ifstream ifs(filename + "_RLNNCogEng" ".txt");
			boost::archive::text_iarchive ia(ifs);
			//read class state from archive
			ia & *this;
			//archive and stream closed when destructors are called.

			//load nn weights
			nnExplore.loadOldRun(filename + "_nnExplore" + ".txt");
			for(int i=0; i<nnExploit.size(); i++) {
				nnExploit[i]->loadOldRun(filename + "_nnExploit_" + std::to_string(i) + ".txt");
			}
			//load training buffers
			trBuf.loadOldRun(filename + "_trBuf" + ".txt");
			//load app specific vars
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

// Initialization
RLNNCognitiveEngine::RLNNCognitiveEngine(const CogEngParams & inpParams)
#if NSE==1
: 	nnExplore(inpParams.nnExplore_nNets,inpParams.nnExplore_inputVectorSize,inpParams.nnExplore_hiddenLayerSizes,inpParams.nnExplore_outputVectorSize,inpParams.sigmoidSlope,inpParams.sigmoidThresh,inpParams.errorThresh),
	nnExploreTrainer(inpParams.nnExplore_nNets,inpParams.nnExplore_inputVectorSize,inpParams.nnExplore_hiddenLayerSizes,inpParams.nnExplore_outputVectorSize,inpParams.sigmoidSlope,inpParams.sigmoidThresh,inpParams.errorThresh),
#else
: 	nnExplore(inpParams.nnExplore_nNets,inpParams.nnExplore_inputVectorSize,inpParams.nnExplore_hiddenLayerSizes,inpParams.nnExplore_outputVectorSize),
	nnExploreTrainer(inpParams.nnExplore_nNets,inpParams.nnExplore_inputVectorSize,inpParams.nnExplore_hiddenLayerSizes,inpParams.nnExplore_outputVectorSize),
#endif
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
	/*******TODO: Check if RMSProp is actually used at any point******/
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
		#if NSE==1
			LearnNSEPredictor<TwoLayerNetwork,optimization::RMSprop> * pNNExploit =
				new LearnNSEPredictor<TwoLayerNetwork,optimization::RMSprop>(
					inpParams.nnExploit_nNets,
					inpParams.nnExploit_inputVectorSize,
					inpParams.nnExploit_hiddenLayerSizes,
					inpParams.nnExploit_outputVectorSize,
					inpParams.sigmoidSlope,
					inpParams.sigmoidThresh,
					inpParams.errorThresh
					);
		#else
			NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop> * pNNExploit =
				new NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop>(
					inpParams.nnExploit_nNets,
					inpParams.nnExploit_inputVectorSize,
					inpParams.nnExploit_hiddenLayerSizes,
					inpParams.nnExploit_outputVectorSize
					);
		#endif

		pNNExploit->initializeRMSProp(inpParams.nnExploit_rmsProp_stepSize,
							 inpParams.nnExploit_rmsProp_alpha,
							 inpParams.nnExploit_rmsProp_eps,
							 inpParams.nnExploit_rmsProp_maxEpochs*inpParams.buf_nTrainTestSamples,
							 inpParams.nnExploit_rmsProp_tolerance,
							 inpParams.nnExploit_rmsProp_shuffle
							 );		
		nnExploit.push_back(pNNExploit);

		#if NSE == 1
		LearnNSEPredictor<TwoLayerNetwork,optimization::RMSprop> * pNNExploitTrainer =
			new LearnNSEPredictor<TwoLayerNetwork,optimization::RMSprop>(//new NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop>(
				inpParams.nnExploit_nNets,
				inpParams.nnExploit_inputVectorSize,
				inpParams.nnExploit_hiddenLayerSizes,
				inpParams.nnExploit_outputVectorSize,
				inpParams.sigmoidSlope,
				inpParams.sigmoidThresh,
				inpParams.errorThresh
				);
		#else 
			NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop> * pNNExploitTrainer =
			new NeuralNetworkPredictor<TwoLayerNetwork,optimization::RMSprop>(
					inpParams.nnExploit_nNets,
					inpParams.nnExploit_inputVectorSize,
					inpParams.nnExploit_hiddenLayerSizes,
					inpParams.nnExploit_outputVectorSize
					);
		#endif


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

// Logic to select action in cognitive engine.
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
			} 
			else {
				arma::uvec idxsLessThresh = arma::find(predictions < perfThresh);
				arma::uvec idxsGreatThresh = arma::find(predictions >= perfThresh);
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
						//choose action worse than _nnRejectionRate
						int idx = idxsLessThresh.n_elem;
						while(idx >= idxsLessThresh.n_elem) {
							idx = (int) floor((double) idxsLessThresh.n_elem*math::Random());
						}
						actionID = idxsLessThresh(idx);
					} else { //choose action better than _nnRejectionRate
						int idx = idxsGreatThresh.n_elem;
						while(idx >= idxsGreatThresh.n_elem) {
							idx = (int) floor((double) idxsGreatThresh.n_elem*math::Random());
						}
						actionID = idxsGreatThresh(idx);
					}					
				}
			}
			_exploitFlag = false;
		} 
		else { //exploit mode
			//generate the input vectors for prediction
			std::cout << "Exploiting" << std::endl;
			#ifdef LOGGING
			logFile << "::Action Type: Exploiting " << std::endl;
			#endif
			appSpecObj.genNNExploitInputs(_nnExploitInputs);
			
			//predict using input vectors. returns mean of all
			//parallel nets
			for(int i=0; i<nnExploit.size(); i++) {
				nnExploit[i]->predict(_nnExploitInputs, predictions);
				predictedActionVec.push_back(predictions);
			}
			//the outputs of the nn's are processed using the application
			//specific object and the actionID corresponding to their outputs
			//is returned back.
			actionID = appSpecObj.processNNExploitOutputs(predictedActionVec);

			_exploitFlag = true;
		}
	} 
	else { //If reset or if first history buffer
		//choose random action
		if(_currentlyTraining) {
			#ifdef LOGGING
			logFile << "::Action Type: Exploiting " << std::endl;
			#endif

			std::cout << "Exploiting" << std::endl;
			//std::cout<<"training"<< std::endl;
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

	return actionID;

}
// Logic for interpreting the evaluation metrics that are received. 
// Also encompassses training when history buffer is full.
void RLNNCognitiveEngine::recordResponse(int actionID, const arma::rowvec &measurementVec) {
	bool bufferFull;
	arma::rowvec fitParams;
	arma::colvec outVec;
	double fitObserved;
	static  boost::posix_time::ptime trainStartTime,trainEndTime;
	
	//process and save measurements from receiver
	appSpecObj.processMeasurements(measurementVec);

	//compute new fitness observed
	appSpecObj.getFitnessParams(fitParams);
	
	//This cout is here for demonstration purposes.
	std::cout << "Normalized Objective Scores:" << std::endl;
	std::cout << "Throughput: " << fitParams(0) << std::endl;
	std::cout << "BER: " << fitParams(1) << std::endl;
	std::cout << "Target BW: " << fitParams(2) << std::endl;
	std::cout << "Spectral Efficiency: " << fitParams(3) << std::endl;
	std::cout << "TX Power Efficiency: " << fitParams(4) << std::endl;
	std::cout << "DC Power Consumed: " << fitParams(5) << std::endl;
	fitObserved = arma::dot(_fitnessWeights,fitParams);
	std::cout << "Multiobjective Fitness Score: " << fitObserved << std::endl;

	#ifdef LOGGING
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
		_fitObservedMax = fitObserved;
		if(!_exploitFlag) {
			appSpecObj.updateNNExploitInputs();
		}
	} 
	else { //not a max observed fitness score
		if(_exploitFlag) {
			if(fitObserved<_lastExploitFitObserved) {
				//Reset "More Efficient Mode". Threshold value is a hyperparameter (0.5 for specific missions, 0.1 for general)
				if(  (((_lastExploitFitObserved-fitObserved)>_forceExploreThreshold) &&
					 (appSpecObj.lastAndNewNNExploitInputEqual()) &&
					 (appSpecObj.lastNNExploitInputEmpty()==false)) 
					 ||
					 (((_histRollBackCnt>=trBuf.getBufferSize()) &&
					 (appSpecObj.lastAndNewNNExploitInputEqual()) &&
					 (appSpecObj.lastNNExploitInputEmpty()==false))) 
					&& (!_currentlyTraining)
				  ) {
				  	_forceExplore = 1; //enter forced explore mode
				  	_fitObservedMax = 0; //reset max tracking
				  	trBuf.resetBuffer(); //reset NN history
				  	_histRollBackCnt = 0;
				}
				// Quick "recover mode" using performances from the buffer.  
				//Triggers when 90% below previous exploration level
				else if (fitObserved < _lastExploitFitObserved*0.9 ||  fitObserved < appSpecObj.getRollBackThreshold()) {
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
			} 
			else { //update last exploitation performance
				_lastExploitFitObserved = fitObserved;
				appSpecObj.updateLastNNExploitInputs();
				_histRollBackCnt = 0;
			}
		}
	}

	//push action/result into training data buffer
	appSpecObj.genTrainingSample(outVec);
	bufferFull = trBuf.addTrainingSample(actionID, outVec);

	//train NNs if history buffer is full.
	if(bufferFull) {
		if(true) {
			if(!_currentlyTraining) {
				std::cout << "Training Occurring" << std::endl;
				trainStartTime = boost::posix_time::microsec_clock::local_time();
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
					nnExploreTrainer.train((*nnInputs)[0],(*nnOutputs)[0],_trainFrac);			
						
					//train NN array
					for(int i=0; i<nnExploit.size(); i++) {
						nnExploitTrainer[i]->train((*nnInputs)[1],(*nnOutputs)[1].row(i),_trainFrac);
					}
					_trainingComplete = true;
				}));

			}
			if(_trainingComplete) {
				//join thread
				tTrainingVec[0].join();
				tTrainingVec.pop_back();
				_currentlyTraining = false; /**********************/

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
		}	
	} else {
		#ifdef LOGGING
		logFile << "::Training: No" <<std::endl;
		#endif
	}
}

