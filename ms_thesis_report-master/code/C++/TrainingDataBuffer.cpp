#include <mlpack/core.hpp>

#include "/home/tim/Desktop/rlnn5/Logging.hpp"

#include <iostream>

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
#include <algorithm>
#include <cmath>

using namespace mlpack;

namespace boost {
namespace serialization {
class access;
}
}

class TrainingDataBuffer {
	public:
		bool addTrainingSample(int actionID, const arma::colvec & outVec);
		void buildTrainingSet(	int nnIdx,
								const arma::mat & actionList,
								const std::vector<int> &inpBuffIdxs, const std::vector<int> &inpBuffExpandActions,
								const arma::mat &inpNormParams,
								const std::vector<int> &outBuffIdxs, const std::vector<int> &outBuffExpandActions,
								const arma::mat &outNormParams);

		bool pruneDataBuffer(double fracToKeep);

		void printBuffActionIDTime();
		void printBuffOutputVec();
		void printTrainTestInput();
		void printTrainTestOutput();

		std::vector<arma::mat> * getTrainTestInput();
		std::vector<arma::mat> * getTrainTestOutput();

		int getBufCntr();
		int getBufferSize();
		void resetBuffer();

	void buildTrainingColumn( 	int nnIdx,
								const arma::mat & actionList,
								const std::vector<int> &inpBuffIdxs, const std::vector<int> &inpBuffExpandActions,
								const arma::mat &inpNormParams,
								const std::vector<int> &outBuffIdxs, const std::vector<int> &outBuffExpandActions,
								const arma::mat &outNormParams,
								int paramToSortBy,
								int ascendingOrderIdx,
								arma::colvec &trainingInCol, arma::colvec &trainingOutCol);

		TrainingDataBuffer(int nOutVecFeatures, int nTrainTestSamples, int nNeuralNets); //constructor

		struct InpOutBuffParams {
			std::vector<int> inpBuffIdxs;
			std::vector<int> inpBuffExpandActions;
			arma::mat inpNormParams;
			std::vector<int> outBuffIdxs;
			std::vector<int> outBuffExpandActions;
			arma::mat outNormParams;
		};

		void saveCurrentRun(std::string filename) {

			std::ofstream ofs(filename);
			//save data to archvie
			boost::archive::text_oarchive oa(ofs);
			//write class instance to archive
			oa & *this;
			//archive and stream closed with destructors are called
		};

		void loadOldRun(std::string filename) {
			std::ifstream ifs(filename);
			boost::archive::text_iarchive ia(ifs);
			//read class state from archive
			ia & *this;
			//archive and stream closed with destructors are called.
		}

	private:
		arma::Mat<int> _buffActionIDTime;
		arma::mat _buffOutputVec;

		std::vector<arma::mat> _trainTestInput;
		std::vector<arma::mat> _trainTestOutput;

		double _bufCntr;

		int _lastActionID;

		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const int version) {
			ar & _buffActionIDTime;
			ar & _buffOutputVec;
			ar & _trainTestInput;
			ar & _trainTestOutput;
			ar & _bufCntr;
			ar & _lastActionID;
		}
};

TrainingDataBuffer::TrainingDataBuffer(int nOutVecFeatures, int nTrainTestSamples, int nNeuralNets) {
	_buffActionIDTime.set_size(2, nTrainTestSamples); //{actionID; lastModified}
	_buffActionIDTime.fill(-1);
	_buffOutputVec.set_size(nOutVecFeatures, nTrainTestSamples);

	_trainTestInput.resize(nNeuralNets);
	_trainTestOutput.resize(nNeuralNets);

	_bufCntr = 0;
}

std::vector<arma::mat> * TrainingDataBuffer::getTrainTestInput() {
	return &_trainTestInput;
}

std::vector<arma::mat> * TrainingDataBuffer::getTrainTestOutput() {
	return &_trainTestOutput;
}

int TrainingDataBuffer::getBufCntr() {
	return _bufCntr;
}

int TrainingDataBuffer::getBufferSize() {
	return _buffActionIDTime.n_cols;
}

void TrainingDataBuffer::resetBuffer() {
	_buffActionIDTime.fill(-1);
	_bufCntr = 0;	
}

void TrainingDataBuffer::printBuffActionIDTime() {
	std::cout << _buffActionIDTime <<  std::endl;
}

void TrainingDataBuffer::printBuffOutputVec() {
	std::cout << _buffOutputVec <<  std::endl;
}

void TrainingDataBuffer::printTrainTestInput() {
	for(int i=0; i<_trainTestInput.size(); i++) {
		std::cout << "NN_" << i << ":" << std::endl;
		std::cout << _trainTestInput[i] <<  std::endl;
	}
}

void TrainingDataBuffer::printTrainTestOutput() {
	for(int i=0; i<_trainTestOutput.size(); i++) {
		std::cout << "NN_" << i << ":" << std::endl;
		std::cout << _trainTestOutput[i] <<  std::endl;
	}
}

bool TrainingDataBuffer::addTrainingSample(int actionID, const arma::colvec & outVec) {
	//search for action ID
	arma::uvec currActionIdx = find(_buffActionIDTime.row(0)==actionID);
	if(currActionIdx.is_empty()) { //action not in buffer
		if(_bufCntr<_buffActionIDTime.n_cols) { //buffer not full yet
			//add to next free spot
			arma::uvec freeIdx = find(_buffActionIDTime.row(0)==-1,1,"first");
			_buffActionIDTime(0,freeIdx(0)) = actionID;
			_buffActionIDTime(1,freeIdx(0)) = 0;
			_buffOutputVec.col(freeIdx(0)) = outVec;
			_buffActionIDTime.row(1) = _buffActionIDTime.row(1) + arma::ones<arma::Row<unsigned int>>(_buffActionIDTime.n_cols);
			_bufCntr++;
		} else { //buffer full
			//replace oldest entry
			int oldestIdx = _buffActionIDTime.row(1).index_max();
			_buffActionIDTime(0,oldestIdx) = actionID;
			_buffActionIDTime(1,oldestIdx) = 0;		
			_buffOutputVec.col(oldestIdx) = outVec;
			_buffActionIDTime.row(1) = _buffActionIDTime.row(1) + arma::ones<arma::Row<unsigned int>>(_buffActionIDTime.n_cols);	
		}
	} else { //action is in buffer
		//only add if last action is different than current action
		if(_lastActionID != actionID) {
			arma::uvec newerTimestamps = find(_buffActionIDTime.row(1) < _buffActionIDTime(1,currActionIdx(0)));
			arma::Row<unsigned int> onesVec = arma::ones<arma::Row<unsigned int>>(newerTimestamps.n_elem);
			//add 1 to all newer timestamps than the found Action IDX
			_buffActionIDTime(onesVec,newerTimestamps) = _buffActionIDTime(onesVec,newerTimestamps) + 1;
			//add new timestamp
			_buffActionIDTime(1,currActionIdx(0)) = 1;
			//replace with new entry
			_buffOutputVec.col(currActionIdx(0)) = outVec;
			//_buffActionIDTime.row(1) = _buffActionIDTime.row(1) + arma::ones<arma::Row<unsigned int>>(_buffActionIDTime.n_cols);			
		}
	}
	
	_lastActionID = actionID;

	return _bufCntr == _buffActionIDTime.n_cols;
}

bool TrainingDataBuffer::pruneDataBuffer(double fracToKeep) {
	int nSampToKeep = (int) floor(fracToKeep*_buffActionIDTime.n_cols);
	for(int i=0; i<_buffActionIDTime.n_cols; i++) {
		if(_buffActionIDTime(1,i) > nSampToKeep) { //vals start at 1
			_buffActionIDTime(0,i) = -1;
			_buffActionIDTime(1,i) = 0;
			_buffOutputVec.col(i) = -1.0*arma::ones<arma::vec>(_buffOutputVec.n_rows);
		}
	}

	_bufCntr = nSampToKeep;
	return _bufCntr == _buffActionIDTime.n_cols;
}

void TrainingDataBuffer::buildTrainingSet( int nnIdx, //index of NN's buffer to build (starts at 0)
								const arma::mat & actionList, //n_action_types x n_permutations
								const std::vector<int> &inpBuffIdxs,	//idx -1 is actionID, idx0+ correspond to outVec passed into
																		//  addTrainingSample
								const std::vector<int> &inpBuffExpandActions, //same length as inpBuffIdxs. true if actionID should be
																				// expanded at this idx. False otherwise
								const arma::mat &inpNormParams,	//cols: [param min, param max, new range min, new range max].
																//rows: corresponds to idxs in inpBuffIdxs 
								const std::vector<int> &outBuffIdxs,	//idx -1 is actionID, idx0+ correspond to outVec passed into
																		//  addTrainingSample
								const std::vector<int> &outBuffExpandActions, //same length as inpBuffIdxs. true if actionID should be
																				// expanded at this idx. False otherwise
								const arma::mat &outNormParams)	//cols: [param min, param max, new range min, new range max].
																//rows: corresponds to idxs in inpBuffIdxs 
{
	//determine number of rows of _trainTestInput
	int nRows = 0;
	for(int i=0; i<inpBuffExpandActions.size(); i++) {
		if(inpBuffExpandActions[i]==1 || inpBuffExpandActions[i]==2) {
			nRows=nRows+actionList.n_rows; //we're going to be expanding the action list
		} else {
			nRows++;	//outVec parameter
		}
	}
	_trainTestInput[nnIdx].set_size(nRows,_buffActionIDTime.n_cols);

	//start populating the input training buffer
	for(int j=0; j<_trainTestInput[nnIdx].n_cols; j++) { //loop through columns of input training buffer (nTrainingSamples)
		int rowIdx=0; //actual row idx of _trainTestInput (accounts for actionList expanding)
		for(int k=0; k<inpBuffIdxs.size(); k++) { //loop through which idxs of features you want in the input
			if(inpBuffExpandActions[k] == 1) { //if we're going to expand an action ID into its vector actions and scale from global action min/max
				for(int m=0; m<actionList.n_rows; m++) { //loop through each element in a vector action
					//normalize value and copy into 
					//val_norm = (new_range_max-new_range_min)*(val-param_min)/(param_max-param_min) + new_range_min
					_trainTestInput[nnIdx](rowIdx,j) = (inpNormParams(k,3)-inpNormParams(k,2))*
												  (actionList(m,_buffActionIDTime(0,j))-inpNormParams(k,0))/
												  (inpNormParams(k,1)-inpNormParams(k,0))
												  + inpNormParams(k,2);
					rowIdx++;
				}
			} else if(inpBuffExpandActions[k] == 2) { //expand and scale from action's min/max
				for(int m=0; m<actionList.n_rows; m++) { //loop through each element in a vector action
					//normalize value and copy into 
					//val_norm = (new_range_max-new_range_min)*(val-param_min)/(param_max-param_min) + new_range_min
					if((actionList.row(m).max()-actionList.row(m).min())/std::max(std::abs(actionList.row(m).max()),std::abs(actionList.row(m).min())) > 0.00001) {
						_trainTestInput[nnIdx](rowIdx,j) = (inpNormParams(k,3)-inpNormParams(k,2))*
													  (actionList(m,_buffActionIDTime(0,j))-actionList.row(m).min())/
													  (actionList.row(m).max()-actionList.row(m).min())
													  + inpNormParams(k,2);
					} else {
						_trainTestInput[nnIdx](rowIdx,j) = inpNormParams(k,3);
					}
					rowIdx++;
				}				
			} else { //we're adding either a value from outVec that was added or the action ID # directly
				if(inpBuffIdxs[k] != -1) {	//adding outVec value
					_trainTestInput[nnIdx](rowIdx,j) = (inpNormParams(k,3)-inpNormParams(k,2))*
													  (_buffOutputVec(inpBuffIdxs[k],j)-inpNormParams(k,0))/
													  (inpNormParams(k,1)-inpNormParams(k,0))
													  + inpNormParams(k,2);
				} else { //add actionID directly (not sure why'd you want to ever do this, but i'm giving you the option)
					_trainTestInput[nnIdx](rowIdx,j) = (inpNormParams(k,3)-inpNormParams(k,2))*
												  (_buffActionIDTime(0,j)-inpNormParams(k,0))/
												  (inpNormParams(k,1)-inpNormParams(k,0))
												  + inpNormParams(k,2);						
				}
				rowIdx++;
			}
		}
	}

	//determine number of rows of _trainTestOutput
	nRows = 0;
	for(int i=0; i<outBuffExpandActions.size(); i++) {
		if(outBuffExpandActions[i]==1 || outBuffExpandActions[i]==2) {
			nRows=nRows+actionList.n_rows; //we're going to be expanding the action list
		} else {
			nRows++;	//outVec parameter
		}
	}
	_trainTestOutput[nnIdx].set_size(nRows,_buffActionIDTime.n_cols);

	//start populating the output training buffer
	for(int j=0; j<_trainTestOutput[nnIdx].n_cols; j++) { //loop through columns of output training buffer (nTrainingSamples)
		int rowIdx=0; //actual row idx of _trainTestOutput (accounts for actionList expanding)
		for(int k=0; k<outBuffIdxs.size(); k++) { //loop through which idxs of features you want in the output
			if(outBuffExpandActions[k] == 1) { //if we're going to expand an action ID into its vector actions and scale from global min/max
				for(int m=0; m<actionList.n_rows; m++) { //loop through each element in a vector action
					//val_norm = (new_range_max-new_range_min)*(val-param_min)/(param_max-param_min) + new_range_min
					_trainTestOutput[nnIdx](rowIdx,j) = (outNormParams(k,3)-outNormParams(k,2))*
												  (actionList(m,_buffActionIDTime(0,j))-outNormParams(k,0))/
												  (outNormParams(k,1)-outNormParams(k,0))
												  + outNormParams(k,2);
					rowIdx++;
				}
			} else if(outBuffExpandActions[k] == 2) { //expand and scale from action's min/max
				for(int m=0; m<actionList.n_rows; m++) { //loop through each element in a vector action
					//val_norm = (new_range_max-new_range_min)*(val-param_min)/(param_max-param_min) + new_range_min
					double rowMax = arma::max(actionList.row(m));
					double rowMin = arma::min(actionList.row(m));
					if((rowMax-rowMin)/std::max(std::abs(rowMax),std::abs(rowMin)) > 0.00001) {
						_trainTestOutput[nnIdx](rowIdx,j) = (outNormParams(k,3)-outNormParams(k,2))*
													  (actionList(m,_buffActionIDTime(0,j))-rowMin)/
													  (rowMax-rowMin)
													  + outNormParams(k,2);
					} else {
						_trainTestOutput[nnIdx](rowIdx,j) = outNormParams(k,3);
					}
					rowIdx++;
				}
			} else { //we're adding either a value from outVec that was added or the action ID # directly
				if(outBuffIdxs[k] != -1) {	//adding outVec value
					_trainTestOutput[nnIdx](rowIdx,j) = (outNormParams(k,3)-outNormParams(k,2))*
													  (_buffOutputVec(outBuffIdxs[k],j)-outNormParams(k,0))/
													  (outNormParams(k,1)-outNormParams(k,0))
													  + outNormParams(k,2);
				} else { //add actionID directly (not sure why'd you want to ever do this, but i'm giving you the option)
					_trainTestOutput[nnIdx](rowIdx,j) = (outNormParams(k,3)-outNormParams(k,2))*
												  (_buffActionIDTime(0,j)-outNormParams(k,0))/
												  (outNormParams(k,1)-outNormParams(k,0))
												  + outNormParams(k,2);						
				}
				rowIdx++;
			}
		}
	}

}

void TrainingDataBuffer::buildTrainingColumn( int nnIdx, //index of NN's buffer to build (starts at 0)
								const arma::mat & actionList, //n_action_types x n_permutations
								const std::vector<int> &inpBuffIdxs,	//idx -1 is actionID, idx0+ correspond to outVec passed into
																		//  addTrainingSample
								const std::vector<int> &inpBuffExpandActions, //same length as inpBuffIdxs. true if actionID should be
																				// expanded at this idx. False otherwise
								const arma::mat &inpNormParams,	//cols: [param min, param max, new range min, new range max].
																//rows: corresponds to idxs in inpBuffIdxs 
								const std::vector<int> &outBuffIdxs,	//idx -1 is actionID, idx0+ correspond to outVec passed into
																		//  addTrainingSample
								const std::vector<int> &outBuffExpandActions, //same length as inpBuffIdxs. true if actionID should be
																				// expanded at this idx. False otherwise
								const arma::mat &outNormParams,	//cols: [param min, param max, new range min, new range max].
																//rows: corresponds to idxs in inpBuffIdxs
								int paramToSortBy,				//-1: time, 0->size(outvec): param in outvec
								int ascendingOrderIdx,			//column number from buffer idx'd "sorted" by value in idx in ascending order
								arma::colvec &trainingInCol,	//input training vector
								arma::colvec &trainingOutCol)	//output training vector
{
	//determine number of rows of trainingCol
	int nRows = 0;
	for(int i=0; i<inpBuffExpandActions.size(); i++) {
		if(inpBuffExpandActions[i]==1 || inpBuffExpandActions[i]==2) {
			nRows=nRows+actionList.n_rows; //we're going to be expanding the action list
		} else {
			nRows++;	//outVec parameter
		}
	}
	trainingInCol.set_size(nRows);

	//find column we want to build
	//std::cout << "HERE2A" << std::endl;
	arma::uvec sortedIdxs;
	if(paramToSortBy==-1) {
		sortedIdxs = arma::sort_index(_buffActionIDTime.row(1),"ascend");
	} else {
		sortedIdxs = arma::sort_index(_buffOutputVec.row(paramToSortBy),"ascend");
	}
	//std::cout << "HERE2B" << std::endl;
	int j = (int) sortedIdxs(ascendingOrderIdx);
	//std::cout << "j: " << j << std::endl;
	//std::cout << "ascendOrderIdx" << ascendingOrderIdx << std::endl;

	//start populating the input training column
	int rowIdx=0; //actual row idx of _trainTestInput (accounts for actionList expanding)
	for(int k=0; k<inpBuffIdxs.size(); k++) { //loop through which idxs of features you want in the input
		if(inpBuffExpandActions[k] == 1) { //if we're going to expand an action ID into its vector actions and scale from global action min/max
			for(int m=0; m<actionList.n_rows; m++) { //loop through each element in a vector action
				//normalize value and copy into 
				//val_norm = (new_range_max-new_range_min)*(val-param_min)/(param_max-param_min) + new_range_min
				trainingInCol(rowIdx) = (inpNormParams(k,3)-inpNormParams(k,2))*
											  (actionList(m,_buffActionIDTime(0,j))-inpNormParams(k,0))/
											  (inpNormParams(k,1)-inpNormParams(k,0))
											  + inpNormParams(k,2);
				rowIdx++;
			}
		} else if(inpBuffExpandActions[k] == 2) { //expand and scale from action's min/max
			for(int m=0; m<actionList.n_rows; m++) { //loop through each element in a vector action
				//normalize value and copy into 
				//val_norm = (new_range_max-new_range_min)*(val-param_min)/(param_max-param_min) + new_range_min
				if((actionList.row(m).max()-actionList.row(m).min())/std::max(std::abs(actionList.row(m).max()),std::abs(actionList.row(m).min())) > 0.00001) {
					trainingInCol(rowIdx) = (inpNormParams(k,3)-inpNormParams(k,2))*
												  (actionList(m,_buffActionIDTime(0,j))-actionList.row(m).min())/
												  (actionList.row(m).max()-actionList.row(m).min())
												  + inpNormParams(k,2);
				} else {
					trainingInCol(rowIdx) = inpNormParams(k,3);
				}
				rowIdx++;
			}				
		} else { //we're adding either a value from outVec that was added or the action ID # directly
			if(inpBuffIdxs[k] != -1) {	//adding outVec value
				//std::cout << "HERE2C" << std::endl;
				trainingInCol(rowIdx) = (inpNormParams(k,3)-inpNormParams(k,2))*
												  (_buffOutputVec(inpBuffIdxs[k],j)-inpNormParams(k,0))/
												  (inpNormParams(k,1)-inpNormParams(k,0))
												  + inpNormParams(k,2);
				//std::cout << inpNormParams.row(k) << " " << _buffOutputVec(inpBuffIdxs[k],j) << std::endl;
				//std::cout << "trainingInCol(i): " << trainingInCol(rowIdx) << std::endl;
			} else { //add actionID directly (not sure why'd you want to ever do this, but i'm giving you the option)
				trainingInCol(rowIdx) = (inpNormParams(k,3)-inpNormParams(k,2))*
											  (_buffActionIDTime(0,j)-inpNormParams(k,0))/
											  (inpNormParams(k,1)-inpNormParams(k,0))
											  + inpNormParams(k,2);						
			}
			rowIdx++;
		}
	}
	//std::cout << "HERE2D" << std::endl;
	//determine number of rows of trainingOutCol
	nRows = 0;
	for(int i=0; i<outBuffExpandActions.size(); i++) {
		if(outBuffExpandActions[i]==1 || outBuffExpandActions[i]==2) {
			nRows=nRows+actionList.n_rows; //we're going to be expanding the action list
		} else {
			nRows++;	//outVec parameter
		}
	}
	trainingOutCol.set_size(nRows);
	//std::cout << "HERE2E" << std::endl;
	//std::cout << trainingOutCol.n_elem << std::endl;

	//start populating the output training buffer
	rowIdx=0; //actual row idx of trainingOutCol (accounts for actionList expanding)
	for(int k=0; k<outBuffIdxs.size(); k++) { //loop through which idxs of features you want in the output
		if(outBuffExpandActions[k] == 1) { //if we're going to expand an action ID into its vector actions and scale from global min/max
			for(int m=0; m<actionList.n_rows; m++) { //loop through each element in a vector action
				//val_norm = (new_range_max-new_range_min)*(val-param_min)/(param_max-param_min) + new_range_min
				trainingOutCol(rowIdx) = (outNormParams(k,3)-outNormParams(k,2))*
											  (actionList(m,_buffActionIDTime(0,j))-outNormParams(k,0))/
											  (outNormParams(k,1)-outNormParams(k,0))
											  + outNormParams(k,2);
				rowIdx++;
			}
		} else if(outBuffExpandActions[k] == 2) { //expand and scale from action's min/max
			for(int m=0; m<actionList.n_rows; m++) { //loop through each element in a vector action
				//val_norm = (new_range_max-new_range_min)*(val-param_min)/(param_max-param_min) + new_range_min
				double rowMax = arma::max(actionList.row(m));
				double rowMin = arma::min(actionList.row(m));
				if((rowMax-rowMin)/std::max(std::abs(rowMax),std::abs(rowMin)) > 0.00001) {
					trainingOutCol(rowIdx) = (outNormParams(k,3)-outNormParams(k,2))*
												  (actionList(m,_buffActionIDTime(0,j))-rowMin)/
												  (rowMax-rowMin)
												  + outNormParams(k,2);
				} else {
					trainingOutCol(rowIdx) = outNormParams(k,3);
				}
				rowIdx++;
			}
		} else { //we're adding either a value from outVec that was added or the action ID # directly
			if(outBuffIdxs[k] != -1) {	//adding outVec value
				trainingOutCol(rowIdx) = (outNormParams(k,3)-outNormParams(k,2))*
												  (_buffOutputVec(outBuffIdxs[k],j)-outNormParams(k,0))/
												  (outNormParams(k,1)-outNormParams(k,0))
												  + outNormParams(k,2);
			} else { //add actionID directly (not sure why'd you want to ever do this, but i'm giving you the option)
				trainingOutCol(rowIdx) = (outNormParams(k,3)-outNormParams(k,2))*
											  (_buffActionIDTime(0,j)-outNormParams(k,0))/
											  (outNormParams(k,1)-outNormParams(k,0))
											  + outNormParams(k,2);						
			}
			rowIdx++;
		}
	}

	//std::cout << "HERE2F" << std::endl;

}

/*
int main() {
	const int N_ACTIONS = 20;
	const int N_OUTPUTS = 2;
	const int N_SAMPLES = 10;
	const int N_NNs = 2;
	TrainingDataBuffer buff1(N_OUTPUTS, N_SAMPLES, N_NNs);

	std::vector<int> inpBuffIdxs0 = {-1,0};
	std::vector<int> inpBuffExpandActions0 = {1, 0};
	arma::mat inpNormParams0 = { {0,(N_ACTIONS-1)*2.5,-1,1},
								{0,1,-1,1}
							  };
	std::vector<int> inpBuffIdxs1 = {0,1,-1};
	std::vector<int> inpBuffExpandActions1 = {0, 0, 0};
	arma::mat inpNormParams1 = { {0,1,0,2},
								 {0,1,0,2},
								 {0,N_ACTIONS-1,0,2}
							   };

	arma::mat actionList(4,N_ACTIONS);

	for(int i=0; i<actionList.n_cols; i++) {
		actionList(0,i) = i*1;
		actionList(1,i) = i*1.5;
		actionList(2,i) = i*2;
		actionList(3,i) = i*2.5;
	}
	//add until full
	bool full = false;
	int actionID;
	arma::colvec outVec;
	while(!full) {
		int actionID = math::RandInt(0,N_ACTIONS);
		arma::colvec outVec = math::Random()*arma::ones<arma::colvec>(N_OUTPUTS);
		outVec(0)=outVec(0)*math::Random();
		std::cout << "actionID: " << actionID << std::endl;
		std::cout << "outVec:" << std::endl << outVec << std::endl;
		full = buff1.addTrainingSample(actionID, outVec);
		std::cout << "buffActionIDTime:" << std::endl;
		buff1.printBuffActionIDTime();
		std::cout << "buffOutputVec:" << std::endl;
		buff1.printBuffOutputVec();
	};
	std::cout << "buffActionIDTime:" << std::endl;
	buff1.printBuffActionIDTime();
	std::cout << "buffOutputVec:" << std::endl;
	buff1.printBuffOutputVec();

	//add a couple more to overwrite buffer
	for(int i=0; i<2; i++) {
		int actionID = math::RandInt(0,N_ACTIONS);
		arma::colvec outVec = math::Random()*arma::ones<arma::colvec>(N_OUTPUTS);
		outVec(0)=outVec(0)*math::Random();
		std::cout << "actionID: " << actionID << std::endl;
		std::cout << "outVec:" << std::endl << outVec << std::endl;
		full = buff1.addTrainingSample(actionID, outVec);
		std::cout << "buffActionIDTime:" << std::endl;
		buff1.printBuffActionIDTime();
		std::cout << "buffOutputVec:" << std::endl;
		buff1.printBuffOutputVec();
	}

	//add a couple of the same
	for(int i=0; i<2; i++) {
		int actionID = 0;
		arma::colvec outVec = i*arma::zeros<arma::colvec>(N_OUTPUTS);
		outVec(0)=outVec(0)*math::Random();
		std::cout << "actionID: " << actionID << std::endl;
		std::cout << "outVec:" << std::endl << outVec << std::endl;
		full = buff1.addTrainingSample(actionID, outVec);
		std::cout << "buffActionIDTime:" << std::endl;
		buff1.printBuffActionIDTime();
		std::cout << "buffOutputVec:" << std::endl;
		buff1.printBuffOutputVec();
	}

	buff1.buildTrainingSet(0,actionList,
						   inpBuffIdxs0,inpBuffExpandActions0,inpNormParams0,
						   inpBuffIdxs1,inpBuffExpandActions1,inpNormParams1);
	buff1.buildTrainingSet(1,actionList,
						   inpBuffIdxs1,inpBuffExpandActions1,inpNormParams1,
						   inpBuffIdxs0,inpBuffExpandActions0,inpNormParams0);	
	std::cout << "Action List:" << std::endl <<actionList <<  std::endl;
	std::cout << "buffActionIDTime:" << std::endl;
	buff1.printBuffActionIDTime();
	std::cout << "buffOutputVec:" << std::endl;
	buff1.printBuffOutputVec();
	std::cout << "_trainTestInput:" << std::endl;
	buff1.printTrainTestInput();
	std::cout << "_trainTestOutput:" << std::endl;
	buff1.printTrainTestOutput();

	full = buff1.pruneDataBuffer(0.5);
	std::cout << "buffActionIDTime:" << std::endl;
	buff1.printBuffActionIDTime();
	std::cout << "buffOutputVec:" << std::endl;
	buff1.printBuffOutputVec();
	std::cout << "_trainTestInput:" << std::endl;
	buff1.printTrainTestInput();
	std::cout << "_trainTestOutput:" << std::endl;
	buff1.printTrainTestOutput();

	while(!full) {
		int actionID = math::RandInt(0,N_ACTIONS);
		arma::colvec outVec = math::Random()*arma::ones<arma::colvec>(N_OUTPUTS);
		outVec(0)=outVec(0)*math::Random();
		std::cout << "actionID: " << actionID << std::endl;
		std::cout << "outVec:" << std::endl << outVec << std::endl;
		full = buff1.addTrainingSample(actionID, outVec);
		std::cout << "buffActionIDTime:" << std::endl;
		buff1.printBuffActionIDTime();
		std::cout << "buffOutputVec:" << std::endl;
		buff1.printBuffOutputVec();
	}

}
*/