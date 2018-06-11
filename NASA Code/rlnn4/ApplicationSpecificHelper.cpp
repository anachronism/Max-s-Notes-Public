#include <mlpack/core.hpp>
#include <iostream>
#include <sstream>
//#include </home/tim/Desktop/rlnn4/TrainingDataBuffer.cpp>
#include <cmath>

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


using namespace mlpack;

namespace boost {
namespace serialization {
class access;
}
}

class ApplicationSpecificHelper {
public:
	ApplicationSpecificHelper(const arma::mat &actionList,
								double frameSize,
								double maxEsN0,
								double minEsN0,
								arma::Row<int> modList,
								arma::Row<double> codList,
								arma::Row<double> rollOffList,
								arma::Row<double> symbolRateList,
								arma::Row<double> transmitPowerList,
								arma::Row<int> modCodList,
								double maxBER
							  );

	void genNNExploreInputs(arma::mat &inputs);
	void genNNExploitInputs(arma::mat &inputs);
	void processMeasurements(const arma::rowvec &measurementVec);
	void rollBackExploitInputs();
	void updateNNExploitInputs();
	void updateLastNNExploitInputs();
	void forceSetNNExploitInputs(arma::colvec &newExploitInputs);
	bool lastAndNewNNExploitInputEqual();
	bool lastNNExploitInputEmpty();
	void getFitnessParams(arma::rowvec &params);
	void setFitnessObserved(double fObserved, bool exploitFlag);
	void genTrainingSample(arma::colvec &outVec);
	int processNNExploitOutputs(std::vector<arma::mat> &predictedActionVec);
	int getFitObservedOutVecIdx();
	double getRollBackThreshold();
	int returnFallBackAction();
	TrainingDataBuffer::InpOutBuffParams * getNNInpOutBuffParms(int nn);


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
	arma::mat _actionListNormWithNSNR;
	arma::mat _actionList;
	bool _genNNExploreInputsCalledAlready;

	double _measuredEsN0Lin;
	double _measuredPowConsumedLin;
	double _measuredPowEfficiencyLog10;
	double _measuredPowConsumedLinComplement;
	double _measuredBandwidth;
	double _measuredThroughput;
	double _measuredSpectralEff;
	double _measuredBEREst;
	double _measuredBEREstdB;

	double _frameSize;
	double _maxEsN0Lin;
	double _minEsN0Lin;
	arma::Row<int> _modList;
	arma::Row<double> _codList;
	arma::Row<double> _rollOffList;
	arma::Row<double> _symbolRateList;
	arma::Row<double> _transmitPowerList;
	arma::Row<int> _modCodList;

	double _RsMax;
	double _RsMin;
	double _BWMin;
	double _BWMax;
	double _TMax;
	double _TMin;
	double _EsMinLin;
	double _EsMaxLin;
	double _PConsumMinLin;
	double _PConsumMaxLin;
	double _SpectEffMin;
	double _SpectEffMax;
	double _PEffMaxLog10;
	double _PEffMinLog10;
	double _berDBMax;
	double _berDBMin;

	std::vector<std::vector<double>> _esnoValuesTable;
	std::vector<std::vector<double>> _ferValuesTable;

	arma::rowvec _fitObservedParams;
	double _fitObserved;
	std::vector<double> _fitObservedBuffer;
	int _fitObservedBufferPtr;

	arma::colvec _nnExploitInput;
	arma::colvec _nnExploitInputLast;

	TrainingDataBuffer::InpOutBuffParams _inpOutBuffParamsExplore;
	TrainingDataBuffer::InpOutBuffParams _inpOutBuffParamsExploit;

	arma::colvec _modClassTargets;

	int _fallBackActionID;

	double estimateBER(double esN0dB, int M, double rate);

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const int version) {
		ar & _genNNExploreInputsCalledAlready;
		
		ar & _measuredEsN0Lin;
		ar & _measuredPowConsumedLin;
		ar & _measuredPowConsumedLinComplement;
		ar & _measuredPowEfficiencyLog10;
		ar & _measuredBandwidth;
		ar & _measuredThroughput;
		ar & _measuredSpectralEff;
		ar & _measuredBEREst;
		ar & _measuredBEREstdB;

		ar & _fitObservedParams;
		ar & _fitObserved;
		ar & _fitObservedBuffer;
		ar & _fitObservedBufferPtr;

		ar & _nnExploitInput;
		ar & _nnExploitInputLast;

		ar & _esnoValuesTable;
		ar & _ferValuesTable;
	}
};

ApplicationSpecificHelper::ApplicationSpecificHelper(
	const arma::mat &actionList, double frameSize, double maxEsN0, double minEsN0,
	arma::Row<int> modList, arma::Row<double> codList, arma::Row<double> rollOffList,
	arma::Row<double> symbolRateList, arma::Row<double> transmitPowerList, arma::Row<int> modCodList,
	double maxBER)
{
	//generate normalized action list
	double actionListMax = actionList.max();
	double actionListMin = actionList.min();
	_actionListNormWithNSNR.set_size(actionList.n_rows+1,actionList.n_cols); //last row is SNR
	_actionList.set_size(arma::size(actionList));
	for(int i=0; i<actionList.n_rows; i++) {
		for(int j=0; j<actionList.n_cols; j++) {
			_actionListNormWithNSNR(i,j) = (1-(-1))*(actionList(i,j)-actionListMin)/
						(actionListMax-actionListMin) + (-1);
			_actionList(i,j) = actionList(i,j);
		}
	}
	for(int j=0; j<actionList.n_cols; j++) {
		_actionListNormWithNSNR(_actionListNormWithNSNR.n_rows-1,j) = -1; //populate with dummy right now
	}

	//init vars
	_genNNExploreInputsCalledAlready = false;
	_fitObservedParams.zeros(7); //row vec [T,BER,W,SpecEff,PEff,PCons,SNRLin]
	_fitObserved = 0.0;
	_nnExploitInput.zeros(6,1); //mat (colvec)
	_nnExploitInputLast = -2*arma::ones(6,1); //mat (colvec)

	//save values
	_frameSize = frameSize;
	_modList = modList;
	_codList = codList;
	_rollOffList = rollOffList;
	_symbolRateList = symbolRateList;
	_transmitPowerList = transmitPowerList;
	_modCodList = modCodList;

	//compute value mins/maxes
	_maxEsN0Lin = pow(10,maxEsN0/10);
	_minEsN0Lin = pow(10,minEsN0/10);
	_RsMax = symbolRateList.max();
	_RsMin = symbolRateList.min();
	//_BWMin = _RsMax*(1+_rollOffList.max());
	//_BWMax = _RsMin*(1+_rollOffList.min());
	_BWMin = _RsMin*(1+_rollOffList.min());
	_BWMax = _RsMax*(1+_rollOffList.max());
	_TMax = _RsMax*log2(_modList.max())*_codList.max();
	_TMin = _RsMin*log2(_modList.min())*_codList.min();
	_EsMinLin = pow(10.0,transmitPowerList.min()/10);
	_EsMaxLin = pow(10.0,transmitPowerList.max()/10);
	_PConsumMinLin = _EsMinLin*_RsMin;
	_PConsumMaxLin = _EsMaxLin*_RsMax;
	_SpectEffMin = log2(_modList.min())*_codList.min()/(1+_rollOffList.max());
	_SpectEffMax = log2(_modList.max())*_codList.max()/(1+_rollOffList.min());
	_PEffMaxLog10 = log10(log2(_modList.max())*_codList.max()/(_EsMinLin*_RsMin));
	_PEffMinLog10 = log10(log2(_modList.min())*_codList.min()/(_EsMaxLin*_RsMax));
	_berDBMax = -10*log10(maxBER);
	_berDBMin = -10*log10(1);

	//nnExplore Buff Params
	_inpOutBuffParamsExplore.inpBuffIdxs.push_back(-1); //actions
	_inpOutBuffParamsExplore.inpBuffIdxs.push_back(7); //EsN0Lin
	_inpOutBuffParamsExplore.inpBuffExpandActions.push_back(2); //actions expanded scaled from row max/min
	_inpOutBuffParamsExplore.inpBuffExpandActions.push_back(0); //EsN0 not expanded
	_inpOutBuffParamsExplore.inpNormParams << 0 << 1 << -1 << 1 << arma::endr //scale to [-1,1]
										   << _minEsN0Lin << _maxEsN0Lin << -1 << 1 << arma::endr;
	_inpOutBuffParamsExplore.outBuffIdxs.push_back(0); //fitObserved
	_inpOutBuffParamsExplore.outBuffExpandActions.push_back(0); //no expanding
	_inpOutBuffParamsExplore.outNormParams << 0 << 1 << 0 << 1 << arma::endr; //scale to [0,1] (from [0,1])

	//nnExploit Buff Params
	_inpOutBuffParamsExploit.inpBuffIdxs.push_back(1); //throughput
	_inpOutBuffParamsExploit.inpBuffIdxs.push_back(2); //BERdB
	_inpOutBuffParamsExploit.inpBuffIdxs.push_back(3); //BW
	_inpOutBuffParamsExploit.inpBuffIdxs.push_back(4); //specEff
	_inpOutBuffParamsExploit.inpBuffIdxs.push_back(5); //PEff
	_inpOutBuffParamsExploit.inpBuffIdxs.push_back(6); //PConsum
	_inpOutBuffParamsExploit.inpBuffIdxs.push_back(7); //EsN0Lin
	_inpOutBuffParamsExploit.inpBuffExpandActions.push_back(0); //not expand
	_inpOutBuffParamsExploit.inpBuffExpandActions.push_back(0); //not expand
	_inpOutBuffParamsExploit.inpBuffExpandActions.push_back(0); //not expand
	_inpOutBuffParamsExploit.inpBuffExpandActions.push_back(0); //not expand
	_inpOutBuffParamsExploit.inpBuffExpandActions.push_back(0); //not expand
	_inpOutBuffParamsExploit.inpBuffExpandActions.push_back(0); //not expand
	_inpOutBuffParamsExploit.inpBuffExpandActions.push_back(0); //not expand
	_inpOutBuffParamsExploit.inpNormParams 	<< _TMin << _TMax << -1 << 1 << arma::endr
											<< _berDBMin << _berDBMax << -1 << 1 << arma::endr
											<< _BWMin << _BWMax << -1 << 1 << arma::endr
											<< _SpectEffMin << _SpectEffMax << -1 << 1 << arma::endr
											<< _PEffMinLog10 << _PEffMaxLog10 << -1 << 1 << arma::endr
											<< _PConsumMinLin << _PConsumMaxLin << -1 << 1 << arma::endr
											<< _minEsN0Lin << _maxEsN0Lin << -1 << 1 << arma::endr;
	_inpOutBuffParamsExploit.outBuffIdxs.push_back(-1); //actions
	_inpOutBuffParamsExploit.outBuffExpandActions.push_back(2); //expand using row max/min (not actionList global)
	_inpOutBuffParamsExploit.outNormParams << 0 << 0 << 0 << 1 << arma::endr; //since we're using action max/min
																		//the original min/max doesn't matter.

	//creating mod class targets
	arma::Mat<int> uniqueMods = arma::unique(_modList); //col vec
	_modClassTargets.set_size(uniqueMods.n_elem*uniqueMods.n_elem);
	int it=0;
	for(int i=0; i<uniqueMods.n_elem; i++) {
		for(int j=0; j<uniqueMods.n_elem; j++) {
			//std::cout << ((double) (uniqueMods(i)-uniqueMods.min()))/((double) (uniqueMods.max()-uniqueMods.min())) << "  "
			//			<< (log2((double) uniqueMods(j))-log2((double) uniqueMods.min()))/(log2((double) uniqueMods.max())-log2((double) uniqueMods.min())) << std::endl;
			_modClassTargets(it) = ((double) (uniqueMods(i)-uniqueMods.min()))/((double) (uniqueMods.max()-uniqueMods.min()))
								+ (log2((double) uniqueMods(j))-log2((double) uniqueMods.min()))/(log2((double) uniqueMods.max())-log2((double) uniqueMods.min()));
			it++;
		}
	}
	std::cout << "UNIQUE MODS: " << uniqueMods << std::endl;
	std::cout << "MODCLASS TARGETS: " << _modClassTargets << std::endl;

	_fitObservedBufferPtr = 0;
	_fitObservedBuffer.resize(200);
	for(int i=0; i<_fitObservedBuffer.size(); i++) {
		_fitObservedBuffer[i] = 0;
	}

	//load in FER curves
	std::string line;
	int lineCount;
	int lineIter;

	//count number of lines
	std::ifstream ferCurvesFile("ferCurves.txt");
	lineCount=0;
	while(std::getline(ferCurvesFile,line)) {
		lineCount++;
	}
	ferCurvesFile.close();
	std::cout << lineCount << std::endl;

	//read in values
	_esnoValuesTable.resize(lineCount);
	_ferValuesTable.resize(lineCount);
	lineIter=0;
	std::ifstream ferCurvesFile2("ferCurves.txt");
	while(std::getline(ferCurvesFile2,line)) {
		std::istringstream ss(line);
		std::string token;

		int i=0;
		while(std::getline(ss,token,',')) {
			if((i%2)==0) {
				_esnoValuesTable[lineIter].push_back(std::atof(token.c_str()));
			} else {
				_ferValuesTable[lineIter].push_back(std::atof(token.c_str()));
			}
			i++;
		}
		lineIter++;
	}
	ferCurvesFile2.close();


	//brute force search for fall back id
	for(int i=0; i<_actionList.n_cols; i++) {
		if( _actionList(0,i)==_actionList.row(0).min() &&
			_actionList(1,i)==_actionList.row(1).max() &&
			_actionList(3,i)==_actionList.row(3).max() &&
			_actionList(4,i)==_actionList.row(4).min() &&
			_actionList(5,i)==_actionList.row(5).min()
		  ) {

			_fallBackActionID = i;
		}

	}

}

void ApplicationSpecificHelper::genNNExploreInputs(arma::mat &inputs) {
	if(_genNNExploreInputsCalledAlready==false) {
		inputs.set_size(_actionListNormWithNSNR.n_rows,_actionListNormWithNSNR.n_cols);
		for(int i=0; i<inputs.n_rows-1; i++) {
			for(int j=0; j<inputs.n_cols; j++) {
				inputs(i,j) = _actionListNormWithNSNR(i,j);
			}
		}
		_genNNExploreInputsCalledAlready = true;
	}
	for(int i=0; i<inputs.n_cols; i++) {
		//inputs(inputs.n_rows-1,i) = ((_measuredEsN0Lin/_maxEsN0Lin)-0.5)/0.5; //scale [-1,1]
		inputs(inputs.n_rows-1,i) = (1-(-1))*((_measuredEsN0Lin-_minEsN0Lin)/(_maxEsN0Lin-_minEsN0Lin)) + (-1);
	}
}

void ApplicationSpecificHelper::genNNExploitInputs(arma::mat &inputs) {
	inputs.set_size(_nnExploitInput.n_elem+1,1);
	for(int i=0; i<_nnExploitInput.n_elem; i++) {
		inputs(i,0) = (_nnExploitInput(i,0)-0.5)/0.5; //scales [-1,1]
	}
	inputs(_nnExploitInput.n_elem) = (1-(-1))*((_measuredEsN0Lin-_minEsN0Lin)/(_maxEsN0Lin-_minEsN0Lin)) + (-1);
}

void ApplicationSpecificHelper::updateNNExploitInputs() {
	for(int i=0; i<_nnExploitInput.n_elem; i++) {
		_nnExploitInput(i) = _fitObservedParams(i);
	}
}

void ApplicationSpecificHelper::forceSetNNExploitInputs(arma::colvec &newExploitInputs) {
	for(int i=0; i<_nnExploitInput.n_elem; i++) {
		_nnExploitInput(i) = (newExploitInputs(i)+1.0)/2.0; //newExploitInputs is [-1,1], we need to convert to [0,1]
												            //because that's what genNNExploitInputs expects
	}
	//std::cout << "_nnExploitInput now forced to: " << _nnExploitInput << std::endl;	
}

void ApplicationSpecificHelper::updateLastNNExploitInputs() {
	for(int i=0; i<_nnExploitInput.n_elem; i++) {
		_nnExploitInputLast(i) = _nnExploitInput(i);
	}	
}

bool ApplicationSpecificHelper::lastNNExploitInputEmpty() {
	bool flag = true;
	for(int i=0; i<_nnExploitInput.n_elem; i++) {
		flag = flag && (std::abs(-2.0-_nnExploitInputLast(i))<1e-12);  //are they approx equal
	}		
	return flag;
}

void ApplicationSpecificHelper::rollBackExploitInputs() {
	for(int i=0; i<_nnExploitInput.n_elem; i++) {
		_nnExploitInput(i) = _nnExploitInputLast(i);
	}		
}

bool ApplicationSpecificHelper::lastAndNewNNExploitInputEqual() {
	bool flag = true;
	for(int i=0; i<_nnExploitInput.n_elem; i++) {
		flag = flag && (std::abs(_nnExploitInput(i)-_nnExploitInputLast(i))<1e-12);  //are they approx equal
	}		
	return flag;
}

void ApplicationSpecificHelper::getFitnessParams(arma::rowvec &params) {
	params.set_size(_fitObservedParams.n_elem-1); //we exclude SNR
	for(int i=0; i<params.n_elem; i++) {
		params(i) = _fitObservedParams(i);
	}
}

void ApplicationSpecificHelper::setFitnessObserved(double fObserved, bool exploitFlag) {
	_fitObserved = fObserved;

	if(exploitFlag) {
		_fitObservedBuffer[_fitObservedBufferPtr] = fObserved;
		_fitObservedBufferPtr++;
		if(_fitObservedBufferPtr == _fitObservedBuffer.size()) {
			_fitObservedBufferPtr = 0;
		}
	}
}

double ApplicationSpecificHelper::getRollBackThreshold() {
	//find max value
	double thresh = 0.0;
	for(int i=0; i<_fitObservedBuffer.size(); i++) {
		if(_fitObservedBuffer[i] > thresh) {
			thresh = _fitObservedBuffer[i];
		}
	}
	return 0.9*thresh;
}

int ApplicationSpecificHelper::returnFallBackAction() {

	return _fallBackActionID;

}


void ApplicationSpecificHelper::genTrainingSample(arma::colvec &outVec) {
	outVec.set_size(8);
	outVec(0) = _fitObserved;
	outVec(1) = _measuredThroughput;
	outVec(2) = _measuredBEREstdB;
	outVec(3) = _measuredBandwidth;
	outVec(4) = _measuredSpectralEff;
	outVec(5) = _measuredPowEfficiencyLog10;
	outVec(6) = _measuredPowConsumedLinComplement;
	outVec(7) = _measuredEsN0Lin;
}

int ApplicationSpecificHelper::getFitObservedOutVecIdx() {
	return 0; //see outVec defn in genTrainingSample
}

TrainingDataBuffer::InpOutBuffParams * ApplicationSpecificHelper::getNNInpOutBuffParms(int nn) {
	if(nn==0) {
		return &_inpOutBuffParamsExplore;
	} else if(nn==1) {
		return &_inpOutBuffParamsExploit;
	} else {
		return NULL;
	}

}

void ApplicationSpecificHelper::processMeasurements(const arma::rowvec &measurementVec) {
	//measurement vector:
	//EsN0 (dB), TX Power (dB), Symbol Rate (symbols/sec), Roll Off, Modulation, Code Rate

	double esN0 = measurementVec(0);
	double esAdd = measurementVec(1);
	double Rs = measurementVec(2);
	double roll_off = measurementVec(3);
	int M = measurementVec(4);
	double rate = measurementVec(5);

	_measuredEsN0Lin = pow(10.0,(esN0-esAdd)/10.0);
	_measuredPowConsumedLin = pow(10.0,esAdd/10.0)*Rs;
	_measuredPowConsumedLinComplement = _PConsumMaxLin+_PConsumMinLin - _measuredPowConsumedLin;
	_measuredPowEfficiencyLog10 = log10((log2(M)*rate)/_measuredPowConsumedLin);
	_measuredBandwidth = Rs*(1.0+roll_off);
	_measuredThroughput = Rs*log2(M)*rate;
	_measuredSpectralEff = log2(M)*rate/(1.0+roll_off);
	_measuredBEREst = estimateBER(esN0,M,rate);
	_measuredBEREstdB = -10.0*log10(_measuredBEREst);

	//populate observed params, normalized to [0,1]
	_fitObservedParams(0) = (_measuredThroughput-_TMin)/(_TMax-_TMin);
	_fitObservedParams(1) = (_measuredBEREstdB-_berDBMin)/(_berDBMax-_berDBMin);
	_fitObservedParams(2) = (_measuredBandwidth-_BWMin)/(_BWMax-_BWMin);
	_fitObservedParams(3) = (_measuredSpectralEff-_SpectEffMin)/(_SpectEffMax-_SpectEffMin);
	_fitObservedParams(4) = (_measuredPowEfficiencyLog10-_PEffMinLog10)/(_PEffMaxLog10-_PEffMinLog10);
	_fitObservedParams(5) = (_measuredPowConsumedLinComplement-_PConsumMinLin)/(_PConsumMaxLin-_PConsumMinLin);
	_fitObservedParams(6) = (_measuredEsN0Lin-_minEsN0Lin)/(_maxEsN0Lin-_minEsN0Lin);

}

int ApplicationSpecificHelper::processNNExploitOutputs(std::vector<arma::mat> &predictedActionVec) {
	// classify modcod
	arma::Row<int> uniqueMods = arma::unique(_modList); //col vec
	arma::rowvec modClassTargetsShort;
	modClassTargetsShort.zeros(uniqueMods.n_elem);
	for(int i=0; i<uniqueMods.n_elem*4; i=i+4) {
		modClassTargetsShort(i/4) = _modClassTargets(i);
	}
	int modcodClassIdx = (arma::abs(modClassTargetsShort-(predictedActionVec[2](0)+predictedActionVec[4](0)))).index_min();

	//denormalize predicted action
	double actionPredicted;
	arma::colvec actionPredictedDiscrete;
	actionPredictedDiscrete.zeros(_actionList.n_rows);
	for(int i=0; i<_actionList.n_rows; i++) {
		double aLMax = _actionList.row(i).max();
		double aLMin = _actionList.row(i).min();
		actionPredicted = (aLMax-aLMin)*(predictedActionVec[i](0)-0)/(1-0) + aLMin;
		switch(i) {
			case 0: {
				actionPredictedDiscrete(i) = _symbolRateList((arma::abs(actionPredicted - _symbolRateList)).index_min());
				break;
			}
			case 1: {
				actionPredictedDiscrete(i) = _transmitPowerList((arma::abs(actionPredicted - _transmitPowerList)).index_min());
				break;
			}
			case 2: {
				arma::rowvec log2Mod;
				log2Mod.zeros(uniqueMods.n_elem);
				for(int i=0; i<uniqueMods.n_elem; i++) {
					log2Mod(i) = log2(uniqueMods(i));
				}
				actionPredictedDiscrete(i) = log2Mod(modcodClassIdx);
				break;
			}
			case 3: {
				actionPredictedDiscrete(i) = _rollOffList((arma::abs(actionPredicted - _rollOffList)).index_min());
				break;
			}
			case 4: {
				actionPredictedDiscrete(i) = uniqueMods(modcodClassIdx);
				break;
			}
			case 5: {
				arma::rowvec codEdges = arma::trans(_codList(arma::find(_modList==actionPredictedDiscrete(4))));
				actionPredictedDiscrete(i) = codEdges((arma::abs(actionPredicted - codEdges)).index_min());
				break;
			}
		}
	}

	//find discretized action
	int actionID = -1;
	for(int i=0; i<_actionList.n_cols; i++) {
		if(arma::approx_equal(_actionList.col(i),actionPredictedDiscrete,"reldiff",0.01)) { //find within 1% error
			actionID = i;
			break;
		}
	}
	return actionID; //if we can't find an action ID, then return -1.
}

double ApplicationSpecificHelper::estimateBER(double EsN0dB, int mod, double code) {
	double FERlog10;
	double FERlin;

	/*arma::mat linearPoints = {	{-2.40, 1.10*pow(10,-3), -2.10, 1.10*pow(10,-10)}, //QPSK 1/4
								{-1.40, 1.10*pow(10,-3), -1.10, 1.10*pow(10,-10)}, //QPSK 1/3
								{-0.40, 1.10*pow(10,-3), -0.10, 1.10*pow(10,-10)}, //QPSK 2/5
								{ 0.90, 1.10*pow(10,-3),  1.20, 1.10*pow(10,-10)}, //QPSK 1/2
								{ 2.20, 1.15*pow(10,-3),  2.50, 1.00*pow(10, -8)}, //QPSK 3/5
								{ 3.35, 1.15*pow(10,-3),  3.70, 1.00*pow(10, -8)}, //QPSK 3/4
								{ 4.50, 1.10*pow(10,-3),  4.90, 7.00*pow(10,-10)}, //QPSK 4/5
								{ 5.25, 1.10*pow(10,-3),  5.70, 7.00*pow(10,-10)}, //QPSK 5/6
								{ 6.00, 5.00*pow(10,-3),  6.50, 5.50*pow(10, -9)}, //QPSK 8/9
								{ 6.80, 5.00*pow(10,-3),  7.20, 5.50*pow(10, -9)}, //QPSK 9/10 **************
								{ 5.10, 5.50*pow(10,-4),  5.40, 2.00*pow(10, -8)}, //8PSK 3/5
								{ 6.50, 5.50*pow(10,-4),  6.80, 2.00*pow(10, -8)}, //8PSK 2/3
								{ 7.90, 1.15*pow(10,-4),  8.20, 2.50*pow(10, -9)}, //8PSK 3/4
								{ 8.65, 1.15*pow(10,-4),  8.95, 2.50*pow(10, -9)}, //8PSK 4/5
								{ 9.40, 7.00*pow(10,-4),  9.70, 5.00*pow(10, -9)}, //8PSK 5/6
								{10.70, 7.00*pow(10,-4), 11.00, 1.20*pow(10, -9)}, //8PSK 8/9
								{12.00, 7.00*pow(10,-4), 12.30, 1.20*pow(10, -9)}, //8PSK 9/10
								{ 9.35, 1.00*pow(10,-3),  9.75, 1.00*pow(10, -8)}, //16APSK 2/3
								{10.10, 1.00*pow(10,-3), 10.50, 1.00*pow(10, -8)}, //16APSK 3/4
								{10.85, 1.00*pow(10,-3), 11.20, 1.00*pow(10, -8)}, //16APSK 4/5
								{11.60, 1.00*pow(10,-3), 11.90, 5.00*pow(10, -9)}, //16APSK 5/6
								{12.80, 6.00*pow(10,-4), 13.20, 3.00*pow(10, -9)}, //16APSK 8/0
								{14.00, 6.00*pow(10,-4), 14.40, 3.00*pow(10, -9)}, //16APSK 9/10
								{12.30, 1.00*pow(10,-3), 12.80, 3.00*pow(10, -8)}, //32APSK 3/4
								{13.05, 1.00*pow(10,-3), 13.45, 3.00*pow(10, -8)}, //32APSK 4/5
								{13.80, 1.00*pow(10,-3), 14.10, 5.00*pow(10, -9)}, //32APSK 5/6
								{14.90, 6.00*pow(10,-4), 15.40, 3.00*pow(10, -9)}, //32APSK 8/9
								{16.00, 6.00*pow(10,-4), 16.50, 3.00*pow(10, -9)}  //32APSK 9/10
							};

	arma::uvec modIdxs = arma::find(_modList==mod);
	int rowIdx;
	double dist=100.0; //arbitrarily large
	for(int i=0; i<modIdxs.n_elem; i++) {
		if(std::abs(_codList(modIdxs(i))-code)<dist) {
			dist = std::abs(_codList(modIdxs(i))-code);
			rowIdx = modIdxs(i);
		}
	}
	
	//linearly interpolate dB curve
	BERlog10 = (log10(linearPoints(rowIdx,3))-log10(linearPoints(rowIdx,1)))/(linearPoints(rowIdx,2)-linearPoints(rowIdx,0))
				*(EsN0dB-linearPoints(rowIdx,0))+log10(linearPoints(rowIdx,1));
	//hard limit at 0 and -12
	if(BERlog10>=0) {
		BERlog10=0;
	}
	if(BERlog10<-12) {
		BERlog10 = -12;
	}
	//convert to linear
	BERlin = pow(10,BERlog10);
	*/

	//find modcod
	arma::uvec modIdxs = arma::find(_modList==mod);
	int rowIdx;
	double dist=100.0; //arbitrarily large
	for(int i=0; i<modIdxs.n_elem; i++) {
		if(std::abs(_codList(modIdxs(i))-code)<dist) {
			dist = std::abs(_codList(modIdxs(i))-code);
			rowIdx = modIdxs(i);
		}
	}

	//find closest bounding points on curve
	int minPoint = -1;
	int maxPoint = -1;
	for(int i=0; i<_esnoValuesTable[rowIdx].size(); i++) {
		if(EsN0dB>=_esnoValuesTable[rowIdx][i]) {
			minPoint = i;
		}
		if(EsN0dB<_esnoValuesTable[rowIdx][i] && maxPoint==-1) {
			maxPoint = i;
		}
	}

	if(minPoint==-1) { //to the left of the fer curve
		minPoint = maxPoint; //use first two points and linearly interpolate leftward
		maxPoint++;
	}
	if(maxPoint==-1) { //to the right of the fer cruve
		maxPoint = minPoint; //use last two points and linearly interpolate rightward
		minPoint--;
	}

	FERlog10 = ((log10(_ferValuesTable[rowIdx][maxPoint])-log10(_ferValuesTable[rowIdx][minPoint]))/(_esnoValuesTable[rowIdx][maxPoint]-_esnoValuesTable[rowIdx][minPoint]))
				*(EsN0dB-_esnoValuesTable[rowIdx][minPoint])+log10(_ferValuesTable[rowIdx][minPoint]);

	//hard limit at 0 and -12
	if(FERlog10>=0) {
		FERlog10=0;
	}
	if(FERlog10<-6) {
		FERlog10 = -6;
	}
	//convert to linear
	FERlin = pow(10,FERlog10);

	return FERlin;
}
