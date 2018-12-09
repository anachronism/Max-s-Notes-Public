#ifndef RECURSIVELMHELPER
#define RECURSIVELMHELPER

#include <mlpack/core.hpp>
#include <iostream>
#include <sstream>
//#include </home/tim/Desktop/rlnn4/TrainingDataBuffer.cpp>
#include <cmath>

#include <mlpack/prereqs.hpp>
#include <fstream>
#include "/home/tim/Desktop/rlnn5/Logging.hpp"


using namespace mlpack;

namespace boost {
namespace serialization {
class access;
}
}

struct Neuron {
	std::vector<double> inputs;
	std::vector<double> weights;
	double netVal;
	double yVal;
	std::string activationType; //"logsig", "linear" supported

	double actFnSlope;
	std::vector<double> deltaVal;
};

class RecursiveLMHelper {
public:
	

	arma::mat pMat;
	RecursiveLMHelper();
	RecursiveLMHelper(std::vector< std::vector<Neuron> > &network);
	void recursiveUpdate(std::vector<std::vector<Neuron>> &network,
							double mu,
							arma::colvec &errorVec,
							arma::colvec &grad,
							arma::colvec &allCurrentWeights,
							const arma::mat &inpPattern,
							const arma::mat outPatternTrain
							); //* GOTTA DO THIS TO UPDATE ALL SAMPLES*//

private:

	arma::mat sMat;
	float alpha;
	int timeCount;

};

RecursiveLMHelper::RecursiveLMHelper()
{
	//int networkSize = network.size();
	//pMat = arma::mat(networkSize,networkSize,arma::fill::eye);
	sMat = arma::mat(2,2,arma::fill::eye);
	alpha = 0.98;
	timeCount = 0;
}

RecursiveLMHelper::RecursiveLMHelper(std::vector< std::vector<Neuron> > &network)
{
	int networkSize = network.size();
	pMat = arma::mat(networkSize,networkSize,arma::fill::eye);
	sMat = arma::mat(2,2,arma::fill::eye);
	alpha = 0.98;
	timeCount = 0;
}


void RecursiveLMHelper::recursiveUpdate(std::vector<std::vector<Neuron>> &network,
							double mu,
							arma::colvec &errorVec,
							arma::colvec &grad,
							arma::colvec &allCurrentWeights,
							const arma::mat &inpPattern,
							const arma::mat outPatternTrain
							)
{
	// #ifdef LOGGING
	// 	logFile <<"Recursive Update Flag 1" <<std::endl;
	// #endif

	arma::mat Omega;
	arma::colvec omegaRow;
	arma::mat lambdaMat = arma::mat(2,2,arma::fill::eye);
	const int nOutputs = network[network.size()-1].size();
	//int networkSize = inpPattern.n_cols * nOutputs;
	int networkSize = grad.size();

	 // #ifdef LOGGING
	 // 	logFile <<"Recursive Update Flag 2b,grad: "<<grad.size() <<std::endl;
	 // #endif

	omegaRow.set_size(arma::size(grad));

	// #ifdef LOGGING
	// 	logFile <<"Recursive Update Flag 2,netSize: "<<networkSize <<std::endl;
	// 	logFile <<"inpPattern:"<<inpPattern.n_rows<<","<<inpPattern.n_cols<<std::endl;
	// #endif

	if (timeCount == 0){
		pMat = arma::mat(networkSize,networkSize);

	}

	//	int jMatRows=nOutputs*inpPattern.n_cols;
	//jMat.set_size(jMatRows,jMatCols);
	//resize error vector
	//errorVec.set_size(jMatRows);
	//resize current weight vector
	//allCurrentWeights.set_size(jMatCols);
	///////////////////////////////////////
	// #ifdef LOGGING
	// 	logFile <<"Recursive Update Flag 3" <<std::endl;
	// #endif

	omegaRow.zeros();
	// #ifdef LOGGING
	// 	logFile <<"Recursive Update Flag 3a: "<<timeCount%networkSize <<std::endl;
	// 	logFile <<"omegaRow size: "<< omegaRow.size()<<std::endl;
	// #endif
	omegaRow(timeCount % networkSize) = 1;
	// #ifdef LOGGING
	// 	logFile <<"Recursive Update Flag 3b" <<std::endl;
	// #endif
	Omega = arma::join_rows(grad, omegaRow);
	
	lambdaMat(1,1) = 1/mu;
	// #ifdef LOGGING
	// 	logFile <<"Recursive Update Flag 4, lambdaMat"<<lambdaMat.n_rows<<","<<lambdaMat.n_cols<<" Omega: "<<Omega.n_rows<<","<<Omega.n_cols<< " pMat: "<<pMat.n_rows<<","<<pMat.n_cols <<std::endl;
	// #endif

	sMat = alpha * lambdaMat + Omega.t() * pMat * Omega;
	// #ifdef LOGGING
	// 	logFile <<"Recursive Update Flag 4a" <<std::endl;
	// 	sMat.print(logFile);
	// #endif
	pMat = 1/alpha * (pMat - pMat * Omega * arma::pinv(sMat) * Omega.t() * pMat);
	timeCount += 1;
	// #ifdef LOGGING
	// 	logFile <<"Recursive Update Flag 5" <<std::endl;
	// #endif
}

#endif