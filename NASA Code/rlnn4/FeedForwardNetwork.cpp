#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <mlpack/core.hpp>


class FeedForwardNetwork {
public:

	void runLM(	const arma::mat &inpPatternTrain,
				const arma::mat &outPatternTrain,
				const arma::mat &inpPatternValid,
				const arma::mat &outPatternValid,
				double minError,
				double minGrad,
				double maxMu,
				int maxIterations,
				int maxValidationFails
			);

	void runRLM(arma::mat &sMat,
				arma::mat &pMat,
				const arma::mat &inpPatternTrain,
				const arma::mat &outPatternTrain,
				const arma::mat &inpPatternValid,
				const arma::mat &outPatternValid,
				double minError,
				double minGrad,
				double maxMu,
				int maxIterations,
				int maxValidationFails
			);
	void forwardPropagate(const arma::colvec &inputs,
						  arma::colvec &outputs
						 );

	void initNetwork(int nInputs, int nLayers,
					 std::vector<int> neuronsPerLayer,
					 std::vector<std::string> activationTypePerLayer
					 );
	void reInitNetwork();
	void exportWeights(arma::colvec &weights);
	void importWeights(const arma::colvec &weights);

	void printWeights(const arma::colvec &weights);
	void printJacobianMatrix(const arma::mat &jMat);
	void printErrorVec(const arma::colvec &errorVec);
	void printDeltaVals();
	void printSlopes();
	void printNeuronInputs();
	void printWeights();
	void printOutputs(const arma::colvec &outputs);

private:
	struct Neuron {
		std::vector<double> inputs;
		std::vector<double> weights;
		double netVal;
		double yVal;
		std::string activationType; //"logsig", "linear" supported

		double actFnSlope;
		std::vector<double> deltaVal;
	};

	std::vector< std::vector<Neuron> > network_;
	arma::colvec trainedWeights_;

	void initNeuronWeights(std::vector<double> &weights);
	void applyInputs(const arma::colvec &inputs, std::vector<Neuron> &firstLayerNeurons);
	void applyLayer(std::vector<Neuron> &currentLayerNeurons, arma::colvec &outputs);
	void applyLayer(std::vector<Neuron> &currentLayerNeurons, std::vector<Neuron> &nextLayerNeurons);
	void calculateActivationFnSlope(const double &input, double &output, std::string type);
	void applyActivation(const double &input, double &output, std::string type);
	void applyWeights(const std::vector<double> &weights, const std::vector<double> &inputs, double &output);

	void updateNetworkWithWeights(std::vector<std::vector<Neuron>> &network,
								const arma::colvec &weights
							 );
	double computeMeanSquareError(const arma::colvec &errorVec, int nOutputs, int nPatterns);
	void calculateRecursiveWeights(const std::vector<std::vector<Neuron>> &network,
						   const arma::mat &pMat,
						   const arma::colvec &allCurrentWeights,
						   const arma::colvec &errorVec,
						   double mu,
						   arma::colvec &allNewWeights
						  );
	void calculateWeightUpdate(const std::vector<std::vector<Neuron>> &network,
							   const arma::mat &jMat,
							   const arma::colvec &allCurrentWeights,
							   const arma::colvec &errorVec,
							   double mu,
							   arma::colvec &allNewWeights
							  );
	void calculateJacobianMatrix(std::vector<std::vector<Neuron>> &network,
							arma::mat &jMat,
							arma::colvec &errorVec,
							arma::colvec &allCurrentWeights,
							const arma::mat &inpPattern,
							const arma::mat &outPattern
							);
	void recursiveUpdate(std::vector<std::vector<Neuron>> &network,
							arma::mat &sMat,
							arma::mat &pMat,
							double mu,
							arma::colvec &errorVec,
							arma::colvec &allCurrentWeights,
							const arma::mat &inpPattern,
							const arma::mat outPatternTrain
							);
	void backwardPropagate(std::vector<std::vector<Neuron>> &network,
							const arma::colvec &actualoutputVec,
							const arma::colvec &correctOutputVec,
							arma::colvec &errorVec
							);
	void forwardPropagate(std::vector<std::vector<Neuron>> &network,
							const arma::colvec &inputs,
							arma::colvec &outputs
						 );
	void testNN(std::vector<std::vector<Neuron>> &network_,
								const arma::mat &inpPattern,
								const arma::mat &outPattern,
								arma::colvec &errorVec
				);
};

void FeedForwardNetwork::applyWeights(const std::vector<double> &weights, const std::vector<double> &inputs, double &output) {
	double dotProduct = 0.0;
	for(int i=0; i<inputs.size(); i++) {
		dotProduct = dotProduct + weights[i]*inputs[i];
	}
	output = dotProduct;
}

void FeedForwardNetwork::applyActivation(const double &input, double &output, std::string type) {
	if(type.compare("logsig")==0) {
		output = 1 / (1 + std::exp(-1*input));
	} else if(type.compare("linear")==0) {
		output = input;
	} else {
		std::cout << "Unsupported activation function chosen." << std::endl;
	}
}

void FeedForwardNetwork::calculateActivationFnSlope(const double &input, double &output, std::string type) {
	if(type.compare("logsig")==0) {
		//sig'(x)=(1-sig(x))*sig(x)
		output = (1.0-(1.0 / (1.0 + std::exp(-1.0*input))))*(1.0 / (1.0 + std::exp(-1.0*input)));
	} else if(type.compare("linear")==0) {
		//sig'(x) = 1
		output = 1;
	} else {
		std::cout << "Unsupported activation function chosen." << std::endl;
	}	
}

void FeedForwardNetwork::applyLayer(std::vector<Neuron> &currentLayerNeurons, std::vector<Neuron> &nextLayerNeurons) {
	for(int i=0; i<currentLayerNeurons.size(); i++) {
		//apply weights
		applyWeights(currentLayerNeurons[i].weights,
					 currentLayerNeurons[i].inputs,
					 currentLayerNeurons[i].netVal
					 );
		//apply activation
		applyActivation(currentLayerNeurons[i].netVal,
						currentLayerNeurons[i].yVal,
						currentLayerNeurons[i].activationType
						);
		//calculate activation function slope
		calculateActivationFnSlope(currentLayerNeurons[i].netVal,
									currentLayerNeurons[i].actFnSlope,
									currentLayerNeurons[i].activationType
									);
	}
	//update inputs in next node
	for(int i=0; i<nextLayerNeurons.size(); i++) {
		for(int j=0; j<currentLayerNeurons.size(); j++) {
			nextLayerNeurons[i].inputs[j] = currentLayerNeurons[j].yVal;
		}
	}
}

void FeedForwardNetwork::applyLayer(std::vector<Neuron> &currentLayerNeurons, arma::colvec &outputs) {
	for(int i=0; i<currentLayerNeurons.size(); i++) {
		//apply weights
		applyWeights(currentLayerNeurons[i].weights,
					 currentLayerNeurons[i].inputs,
					 currentLayerNeurons[i].netVal
					 );
		//apply activation
		applyActivation(currentLayerNeurons[i].netVal,
						currentLayerNeurons[i].yVal,
						currentLayerNeurons[i].activationType
						);
		//calculate activation function slope
		calculateActivationFnSlope(currentLayerNeurons[i].netVal,
									currentLayerNeurons[i].actFnSlope,
									currentLayerNeurons[i].activationType
									);

	}
	//update output vector
	outputs.set_size(currentLayerNeurons.size());
	for(int i=0; i<currentLayerNeurons.size(); i++) {
		outputs(i) = currentLayerNeurons[i].yVal;
	}
}

void FeedForwardNetwork::applyInputs(const arma::colvec &inputs, std::vector<Neuron> &firstLayerNeurons) {
	for(int i=0; i<firstLayerNeurons.size(); i++) {
		for(int j=0; j<inputs.n_elem; j++) {
			firstLayerNeurons[i].inputs[j] = inputs(j);
		}
	}
}

void FeedForwardNetwork::initNeuronWeights(std::vector<double> &weights) {
	for(int i=0; i<weights.size(); i++) {
		weights[i] = 2*mlpack::math::Random()-1; //scale [-1,1]
		//weights[i] = i+1;
	}
}

void FeedForwardNetwork::reInitNetwork() {

	for(int i=0; i<network_.size(); i++) {
		for(int j=0; j<network_[i].size(); j++) {
			//init weights
			initNeuronWeights(network_[i][j].weights);
		}

	}

}

void FeedForwardNetwork::initNetwork( int nInputs, int nLayers,
				 std::vector<int> neuronsPerLayer,
				 std::vector<std::string> activationTypePerLayer
				 )
{
	//resize network_
	network_.resize(nLayers);

	for(int i=0; i<network_.size(); i++) {
		//resize layer
		network_[i].resize(neuronsPerLayer[i]);

		for(int j=0; j<network_[i].size(); j++) {
			//resize neuron inputs/weights and init weights
			if(i==0) {
				network_[i][j].inputs.resize(nInputs);
				network_[i][j].weights.resize(nInputs);
			} else {
				network_[i][j].inputs.resize(network_[i-1].size());
				network_[i][j].weights.resize(network_[i-1].size());
			}
			//init weights
			initNeuronWeights(network_[i][j].weights);
			//init activation type
			network_[i][j].activationType = activationTypePerLayer[i];
			//resize delta vector size to number of outputs
			network_[i][j].deltaVal.resize(neuronsPerLayer[neuronsPerLayer.size()-1]); 
		}

	}

}


void FeedForwardNetwork::forwardPropagate(std::vector<std::vector<Neuron>> &network,
						const arma::colvec &inputs,
						arma::colvec &outputs
					 )
{
	//apply inputs
	applyInputs(inputs,network[0]);

	//for each layer
	for(int i=0; i<network.size(); i++) {
		if(i != network.size()-1) {
			applyLayer(network[i],network[i+1]);
		} else {
			applyLayer(network[i],outputs);
		}
	}
}

void FeedForwardNetwork::forwardPropagate( const arma::colvec &inputs, arma::colvec &outputs)
{
	//apply inputs
	applyInputs(inputs,network_[0]);

	//for each layer
	for(int i=0; i<network_.size(); i++) {
		if(i != network_.size()-1) {
			applyLayer(network_[i],network_[i+1]);
		} else {
			applyLayer(network_[i],outputs);
		}
	}
}

void FeedForwardNetwork::backwardPropagate(std::vector<std::vector<Neuron>> &network,
						const arma::colvec &actualoutputVec,
						const arma::colvec &correctOutputVec,
						arma::colvec &errorVec
						)
{
	//output m
	//layer k
	//neuron j in layer k
	//i-th input to neuron j

	//resize to number of outputs
	errorVec.set_size(actualoutputVec.size());

	//iterate over all outputs
	for(int m=0; m<network[network.size()-1].size(); m++) {
		//calculate error
		errorVec(m) = correctOutputVec(m)-actualoutputVec(m);

		//iterate over all layers
		for(int k=network.size()-1; k>=0; k--) {
			//output layer
			if(k==network.size()-1) {
				//iterate on all neurons in layer
				for(int j=0; j<network[k].size(); j++) {
					if(j==m) { //if neuron j is the output m neuron
						network[k][j].deltaVal[m] = network[k][j].actFnSlope;
					} else { //all other neurons are zeroed
						network[k][j].deltaVal[m] = 0;
					}
				}
			}
			//layer before output layer
			else if(k==network.size()-2) {
				//iterate on all neurons in layer
				for(int j=0; j<network[k].size(); j++) {
					//bp delta from inputs of (k+1)-th layer to
					//outputs of k-th layer
					network[k][j].deltaVal[m] = 
						network[k+1][m].deltaVal[m] *
						network[k+1][m].weights[j];
					//bp delta from outs of k-th layer to
					//inputs of k-th layer
					network[k][j].deltaVal[m] =
						network[k][j].deltaVal[m] *
						network[k][j].actFnSlope;
				}
			}
			//all other layers
			else {
				//iterate on all neurons in layer
				for(int j=0; j<network[k].size(); j++) {
					double tmp=0;
					//iterate on all inputs neuron j contibributed to
					//in the next layer
					for(int i=0; i<network[k+1].size(); i++) {
						//bp delta from inputs of (k+1)-th layer to
						//outputs of k-th layer
						tmp = tmp + 
							  (network[k+1][i].deltaVal[m] *
							   network[k+1][i].weights[j]);
					}
					network[k][j].deltaVal[m] = tmp;
					//bp delta from outs of k-th layer to
					//inputs of k-th layer
					network[k][j].deltaVal[m] =
						network[k][j].deltaVal[m] *
						network[k][j].actFnSlope;
				}
			}
		}
	}
}

void FeedForwardNetwork::recursiveUpdate(std::vector<std::vector<Neuron>> &network,
							arma::mat &sMat,
							arma::mat &pMat,
							double mu,
							arma::colvec &errorVec,
							arma::colvec &allCurrentWeights,
							const arma::mat &inpPattern,
							const arma::mat outPatternTrain
							)
{

	arma::colvec fwPropOutput;
	arma::colvec onePatternErrorVec;
	const int nOutputs = network[network.size()-1].size();
	arma::mat Omega;
	arma::colvec omegaRow;
	float ALPHA = 0.98;
	//////////////////////////? VERIFY THAT THIS IS NEEDED
	//resize jacobian matrix
	int jMatCols=0;
	for(int i=0; i<network.size(); i++) {
		if(i==0) {
			jMatCols = jMatCols + network[i].size()*network[i][0].inputs.size();
		} else {
			jMatCols = jMatCols + network[i-1].size()*network[i].size();
		}
	}
	
	int jMatRows=nOutputs*inpPattern.n_cols;
	//jMat.set_size(jMatRows,jMatCols);
	//resize error vector
	//errorVec.set_size(jMatRows);
	//resize current weight vector
	//allCurrentWeights.set_size(jMatCols);
	///////////////////////////////////////
	int timeCount = 1; /////// CHANGE THIS
	omegaRow.zeros();
	omegaRow(timeCount % jMatRows) = 1;

	Omega = arma::join_rows(errorVec, omegaRow);
	arma::mat lambdaMat = arma::mat(2,2,arma::fill::eye);
	
	lambdaMat(1,1) = 1/mu;
	sMat = ALPHA * lambdaMat + Omega.t() * pMat * Omega;
	pMat = 1/ALPHA * (pMat - pMat * Omega * arma::pinv(sMat) * Omega.t() * pMat);


}

void FeedForwardNetwork::calculateJacobianMatrix(std::vector<std::vector<Neuron>> &network,
							arma::mat &jMat,
							arma::colvec &errorVec,
							arma::colvec &allCurrentWeights,
							const arma::mat &inpPattern,
							const arma::mat &outPattern
							) 
{
	arma::colvec fwdPropOutput;
	arma::colvec onePatternErrorVec;
	const int nOutputs = network[network.size()-1].size();

	//resize jacobian matrix
	int jMatCols=0;
	for(int i=0; i<network.size(); i++) {
		if(i==0) {
			jMatCols = jMatCols + network[i].size()*network[i][0].inputs.size();
		} else {
			jMatCols = jMatCols + network[i-1].size()*network[i].size();
		}
	}

	int jMatRows=nOutputs*inpPattern.n_cols;
	jMat.set_size(jMatRows,jMatCols);
	//resize error vector
	errorVec.set_size(jMatRows);
	//resize current weight vector
	allCurrentWeights.set_size(jMatCols);

	int colIt;
	int weightIt=0;
	for(int p=0; p<inpPattern.n_cols; p++) {
		forwardPropagate(network,inpPattern.col(p),fwdPropOutput);
		backwardPropagate(network,fwdPropOutput,outPattern.col(p),onePatternErrorVec);

		//iterate over all outputs
		for(int m=0; m<nOutputs; m++) {
			//iterate over all layers
			colIt=0;
			for(int k=0; k<network.size(); k++) {
				//iterate over all neurons in layer k
				for(int j=0; j<network[k].size(); j++) {
					//iterate over all inputs into neuron j
					for(int i=0; i<network[k][j].inputs.size(); i++) {
						//Jacobian Matrix
						jMat(p*nOutputs+m,colIt) = -1*network[k][j].deltaVal[m]*network[k][j].inputs[i];
						colIt++;

						//create weight vector on first pass through
						if(m==0 && p==0) {
							allCurrentWeights(weightIt) = network[k][j].weights[i];
							weightIt++;
						}
					}
				}
			}

			//populate error vector
			errorVec(p*nOutputs+m) = onePatternErrorVec(m);
		}

	}

}

void FeedForwardNetwork::calculateRecursiveWeights(const std::vector<std::vector<Neuron>> &network,
						   const arma::mat &pMat,
						   const arma::colvec &allCurrentWeights,
						   const arma::colvec &errorVec,
						   double mu,
						   arma::colvec &allNewWeights
						  )
{
	allNewWeights.set_size(allCurrentWeights.n_elem);
	//arma::mat IdMat;
	//IdMat.eye(jMat.n_cols,jMat.n_cols);
	//std::cout <<"here"<<std::endl;
	//std::cout << jMat << std::endl;
	arma::colvec grad; ///////ACTUALLY GET GRADIENT
	grad.zeros();
	allNewWeights = allCurrentWeights + pMat * grad * errorVec;
	//std::cout <<"J'J : " << std::endl << ((double)(1.0/jMat.n_rows))*arma::trans(jMat)*jMat << std::endl;
	//std::cout <<"Je : " << std::endl << ((double)(1.0/jMat.n_rows))*arma::trans(jMat)*errorVec;
}

void FeedForwardNetwork::calculateWeightUpdate(const std::vector<std::vector<Neuron>> &network,
						   const arma::mat &jMat,
						   const arma::colvec &allCurrentWeights,
						   const arma::colvec &errorVec,
						   double mu,
						   arma::colvec &allNewWeights
						  )
{
	allNewWeights.set_size(allCurrentWeights.n_elem);
	arma::mat IdMat;
	IdMat.eye(jMat.n_cols,jMat.n_cols);
	//std::cout <<"here"<<std::endl;
	//std::cout << jMat << std::endl;
	allNewWeights = allCurrentWeights - arma::pinv(((double)(1.0/jMat.n_rows))*arma::trans(jMat)*jMat + mu*IdMat)*(((double)(1.0/jMat.n_rows))*arma::trans(jMat)*errorVec);
	//std::cout <<"J'J : " << std::endl << ((double)(1.0/jMat.n_rows))*arma::trans(jMat)*jMat << std::endl;
	//std::cout <<"Je : " << std::endl << ((double)(1.0/jMat.n_rows))*arma::trans(jMat)*errorVec;
}

double FeedForwardNetwork::computeMeanSquareError(const arma::colvec &errorVec, int nOutputs, int nPatterns) {
	double nOutputsDbl = (double) nOutputs;
	double nPatternsDbl = (double) nPatterns;
	arma::mat error = ((double) (1.0/(nOutputsDbl*nPatternsDbl)))*arma::trans(errorVec)*errorVec;
	return error(0,0);
}

void FeedForwardNetwork::updateNetworkWithWeights(std::vector<std::vector<Neuron>> &network,
								const arma::colvec &weights
							 )
{
	int weightIt=0;
	//iterate over layer k
	for(int k=0; k<network.size(); k++) {
		//iterate over all neurons in layer k
		for(int j=0; j<network[k].size(); j++) {
			//iterate over all inputs into neuron j
			for(int i=0; i<network[k][j].inputs.size(); i++) {
				network[k][j].weights[i] = weights(weightIt);
				weightIt++;
			}
		}
	}
}

void FeedForwardNetwork::testNN(std::vector<std::vector<Neuron>> &network_,
								const arma::mat &inpPattern,
								const arma::mat &outPattern,
								arma::colvec &errorVec)
{
	arma::colvec outputs;
	arma::colvec singleRunErrorVec;

	errorVec.set_size(outPattern.n_cols*outPattern.n_rows);

	//iterate over all patterns
	for(int p=0; p<outPattern.n_cols; p++) {
		forwardPropagate(inpPattern.col(p), outputs);
		//std::cout << "old: " << errorVec.subvec(p*outPattern.n_rows,(p+1)*outPattern.n_rows-1) << std::endl;
		errorVec.subvec(p*outPattern.n_rows,(p+1)*outPattern.n_rows-1) = outPattern.col(p) - outputs;
		//std::cout << "new: " << errorVec.subvec(p*outPattern.n_rows,(p+1)*outPattern.n_rows-1) << std::endl;
	}
}

void FeedForwardNetwork::runLM(const arma::mat &inpPatternTrain,
								const arma::mat &outPatternTrain,
								const arma::mat &inpPatternValid,
								const arma::mat &outPatternValid,
								double minError,
								double minGrad,
								double maxMu,
								int maxIterations,
								int maxValidationFails
				  			 )
{
	int m;
	int loopIter;
	arma::mat jMat;
	arma::colvec errorVecForEval;
	arma::colvec errorVecForJe;
	arma::colvec allCurrentWeights;
	arma::colvec allNewWeights;
	double mu;
	double lastError = arma::datum::inf;
	double newError;
	bool reCalc;
	bool extremeMuReached;
	arma::mat gradTmpMat;
	double grad = arma::datum::inf;
	double newErrorVal;
	double lastErrorVal = arma::datum::inf;
	double bestErrorVal = arma::datum::inf;
	int valFailIter;

	//calculate jacobian
	loopIter=0;
	mu=.001;
	valFailIter=0;
	while(((lastError-minError)>1e-12) && 
		   (loopIter < maxIterations) &&
		   //((lastError-newError)>1e-12) &&
		   ((grad-minGrad)>1e-12) &&
		   (mu <= maxMu) &&
		   (valFailIter != maxValidationFails)
		 )
	{
		m = 0;
		//calculate jacobian
		//std::cout << "pre-jacobian"  << std::endl;
		calculateJacobianMatrix(network_,jMat,errorVecForJe,allCurrentWeights,inpPatternTrain,outPatternTrain);
		//std::cout << "post-jacobian" << std::endl;
		errorVecForEval = errorVecForJe;
		lastError = computeMeanSquareError(errorVecForEval,jMat.n_rows,1);

		do {
			//update weights, evaluate, compare with current weight's performance
			//std::cout << "calc weights" << std::endl;
			calculateWeightUpdate(network_,jMat,allCurrentWeights,errorVecForJe,mu,allNewWeights);
			//std::cout << "update weights" << std::endl;
			//std::cout << "new weights: " << allNewWeights << std::endl;
			updateNetworkWithWeights(network_,allNewWeights);
			
			//std::cout << "test nn" << std::endl;
			testNN(network_,inpPatternTrain,outPatternTrain,errorVecForEval);
			//std::cout << "errorVec: " << errorVecForEval << std::endl;
			newError = computeMeanSquareError(errorVecForEval,jMat.n_rows,1);

			//std::cout << loopIter << ":" << lastError << " " << newError << " " << (bool) (newError>lastError) << " " << mu << std::endl;

			if((newError-lastError)>1e-12 /*|| m>0*/) {
				//revert weights
				updateNetworkWithWeights(network_,allCurrentWeights);
				
				//need to use more grad descent
				mu = mu*10.0;
				if(mu>1e11) {
					mu=1e11;
				}

			} else {
				//can use more of newton's method
				mu = mu/10.0;
				if(mu<1e-300) {
					mu=1e-300;
				}
			}
			m=m+1;
		}
		while(((newError-lastError)>1e-12)) ;

		//Calculate Grad for Iteration
		calculateJacobianMatrix(network_,jMat,errorVecForJe,allNewWeights,inpPatternTrain,outPatternTrain);
		//grad = 2*sqrt(1/(M*P)*Je.^2)
		gradTmpMat = 2*arma::sqrt(arma::trans(((double)(1.0/jMat.n_rows))*arma::trans(jMat)*errorVecForJe)*
				(((double)(1.0/jMat.n_rows))*arma::trans(jMat)*errorVecForJe));
		grad = gradTmpMat(0,0);

		//Calculate Validation Fail for Iteration and save best performance
		testNN(network_,inpPatternValid,outPatternValid,errorVecForEval);
		newErrorVal = computeMeanSquareError(errorVecForEval,outPatternValid.n_rows,outPatternValid.n_cols);
		/*if(newErrorVal < lastErrorVal) {
			valFailIter = 0;
		} else { //validation fail
			valFailIter++;
		}
		lastErrorVal = newErrorVal;
		//Save "best performing" weights and save them
		if(newErrorVal < bestErrorVal) {
			trainedWeights_ = allNewWeights;
			bestErrorVal = newErrorVal;
		}*/
		if(newErrorVal < bestErrorVal) {
			valFailIter = 0; //reset val iterator
			trainedWeights_ = allNewWeights; //save best weights
			bestErrorVal = newErrorVal;
		} else { //validation fail
			valFailIter++;
		}

		//print progress
		//std::cout << loopIter << ":" << "grad: " << grad << ", perf: " << newError << ", valStops: " << valFailIter << std::endl;

		loopIter++;
	}

	updateNetworkWithWeights(network_,trainedWeights_);
}

void FeedForwardNetwork::runRLM(arma::mat &sMat,
								arma::mat &pMat,
								const arma::mat &inpPatternTrain,
								const arma::mat &outPatternTrain,
								const arma::mat &inpPatternValid,
								const arma::mat &outPatternValid,

								double minError,
								double minGrad,
								double maxMu,
								int maxIterations,
								int maxValidationFails
				  			 )
{
	int m;
	int loopIter;
	arma::mat jMat;
	arma::colvec errorVecForEval;
	arma::colvec errorVecForJe;
	arma::colvec allCurrentWeights;
	arma::colvec allNewWeights;
	double mu;
	double lastError = arma::datum::inf;
	double newError;
	bool reCalc;
	bool extremeMuReached;
	arma::mat gradTmpMat;
	double grad = arma::datum::inf;
	double newErrorVal;
	double lastErrorVal = arma::datum::inf;
	double bestErrorVal = arma::datum::inf;
	int valFailIter;

	//calculate jacobian
	loopIter=0;
	mu=.001;
	valFailIter=0;
	while(((lastError-minError)>1e-12) && 
		   (loopIter < maxIterations) &&
		   //((lastError-newError)>1e-12) &&
		   ((grad-minGrad)>1e-12) &&
		   (mu <= maxMu) &&
		   (valFailIter != maxValidationFails)
		 )
	{
		m = 0;
		//calculate jacobian
		//std::cout << "pre-jacobian"  << std::endl;
		
		//calculateJacobianMatrix(network_,jMat,errorVecForJe,allCurrentWeights,inpPatternTrain,outPatternTrain);
		recursiveUpdate(network_,sMat,pMat,mu,errorVecForJe,allCurrentWeights,inpPatternTrain,outPatternTrain); //* GOTTA DO THIS TO UPDATE ALL SAMPLES*//
		//std::cout << "post-jacobian" << std::endl;
		// calculateSMatrix(network_,pMat,sMat,errorVecForEval,allCurrentWeights,outPatternTrain);
		// calculatePMatrix(network_,pMat,sMat,errorVecForEval,allCurrentWeights,outPatternTrain);

		errorVecForEval = errorVecForJe;
		lastError = computeMeanSquareError(errorVecForEval,jMat.n_rows,1);

		do {
			//update weights, evaluate, compare with current weight's performance
			//std::cout << "calc weights" << std::endl;
			calculateRecursiveWeights(network_,pMat,allCurrentWeights,errorVecForJe,mu,allNewWeights);
			//calculateWeightUpdate(network_,jMat,allCurrentWeights,errorVecForJe,mu,allNewWeights);
			
			//std::cout << "update weights" << std::endl;
			//std::cout << "new weights: " << allNewWeights << std::endl;
			updateNetworkWithWeights(network_,allNewWeights);
			
			//std::cout << "test nn" << std::endl;
			testNN(network_,inpPatternTrain,outPatternTrain,errorVecForEval);
			//std::cout << "errorVec: " << errorVecForEval << std::endl;
			newError = computeMeanSquareError(errorVecForEval,jMat.n_rows,1);

			//std::cout << loopIter << ":" << lastError << " " << newError << " " << (bool) (newError>lastError) << " " << mu << std::endl;

			if((newError-lastError)>1e-12 /*|| m>0*/) {
				//revert weights
				updateNetworkWithWeights(network_,allCurrentWeights);
				
				//need to use more grad descent
				mu = mu*10.0;
				if(mu>1e11) {
					mu=1e11;
				}

			} else {
				//can use more of newton's method
				mu = mu/10.0;
				if(mu<1e-300) {
					mu=1e-300;
				}
			}
			m=m+1;
		}
		while(((newError-lastError)>1e-12)) ;

		//Calculate Grad for Iteration
		//* GET GRAD HERE THE NORMAL WAY*//

		//calculateJacobianMatrix(network_,jMat,errorVecForJe,allNewWeights,inpPatternTrain,outPatternTrain);
		//grad = 2*sqrt(1/(M*P)*Je.^2)
		//gradTmpMat = 2*arma::sqrt(arma::trans(((double)(1.0/jMat.n_rows))*arma::trans(jMat)*errorVecForJe)*
		//		(((double)(1.0/jMat.n_rows))*arma::trans(jMat)*errorVecForJe));
		//grad = gradTmpMat(0,0);

		//Calculate Validation Fail for Iteration and save best performance
		testNN(network_,inpPatternValid,outPatternValid,errorVecForEval);
		newErrorVal = computeMeanSquareError(errorVecForEval,outPatternValid.n_rows,outPatternValid.n_cols);
		/*if(newErrorVal < lastErrorVal) {
			valFailIter = 0;
		} else { //validation fail
			valFailIter++;
		}
		lastErrorVal = newErrorVal;
		//Save "best performing" weights and save them
		if(newErrorVal < bestErrorVal) {
			trainedWeights_ = allNewWeights;
			bestErrorVal = newErrorVal;
		}*/
		if(newErrorVal < bestErrorVal) {
			valFailIter = 0; //reset val iterator
			trainedWeights_ = allNewWeights; //save best weights
			bestErrorVal = newErrorVal;
		} else { //validation fail
			valFailIter++;
		}

		//print progress
		//std::cout << loopIter << ":" << "grad: " << grad << ", perf: " << newError << ", valStops: " << valFailIter << std::endl;

		loopIter++;
	}

	updateNetworkWithWeights(network_,trainedWeights_);
}

void FeedForwardNetwork::exportWeights(arma::colvec &weights) {
	int nWeights;

	//get number of weights in network
	nWeights=0;
	for(int i=0; i<network_.size(); i++) {
		for(int j=0; j<network_[i].size(); j++) {
			for(int k=0; k<network_[i][j].weights.size(); k++) {
				nWeights++;
			}
		}
	}

	weights.zeros(nWeights);

	//reformat for MLPack
	int it=0;
	for(int i=0; i<network_.size(); i++) {
		for(int k=0; k<network_[i][0].weights.size(); k++) {
			for(int j=0; j<network_[i].size(); j++) {
				weights(it) = network_[i][j].weights[k];
				it++;
			}
		}
	}
}

void FeedForwardNetwork::importWeights(const arma::colvec &weights) {
	//weight format should be for MLPack
	int it=0;
	for(int i=0; i<network_.size(); i++) {
		for(int k=0; k<network_[i][0].weights.size(); k++) {
			for(int j=0; j<network_[i].size(); j++) {
				network_[i][j].weights[k] = weights(it);
				it++;
			}
		}
	}
}

void FeedForwardNetwork::printOutputs(const arma::colvec &outputs) {
	std::cout << "Outputs: " << std::endl;
	std::cout << outputs << std::endl;
	std::cout << std::endl;	
}

void FeedForwardNetwork::printWeights() {
	std::cout << "Weights: " << std::endl;
	for(int i=0; i<network_.size(); i++) {
		std::cout << "Layer: " << i << std::endl;
		for(int j=0; j<network_[i].size(); j++) {
			for(int k=0; k<network_[i][j].weights.size(); k++) {
				std::cout << network_[i][j].weights[k] << " ";
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}

void FeedForwardNetwork::printNeuronInputs() {
	std::cout << "Neuron Inputs: " << std::endl;
	for(int i=0; i<network_.size(); i++) {
		std::cout << "Layer: " << i << std::endl;
		for(int j=0; j<network_[i].size(); j++) {
			for(int k=0; k<network_[i][j].inputs.size(); k++) {
				std::cout << network_[i][j].inputs[k] << " ";
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}

void FeedForwardNetwork::printSlopes() {
	std::cout << "Slopes: " << std::endl;
	for(int i=0; i<network_.size(); i++) {
		std::cout << "Layer: " << i << std::endl;
		for(int j=0; j<network_[i].size(); j++) {
			std::cout << network_[i][j].actFnSlope << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void FeedForwardNetwork::printDeltaVals() {
	std::cout << "deltaVals: " << std::endl;
	for(int i=0; i<network_.size(); i++) {
		std::cout << "Layer: " << i << std::endl;
		for(int j=0; j<network_[i].size(); j++) {
			for(int k=0; k<network_[i][j].deltaVal.size(); k++) {
				std::cout << network_[i][j].deltaVal[k] << " ";
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}

void FeedForwardNetwork::printErrorVec(const arma::colvec &errorVec) {
	std::cout << "Error Vec" << std::endl;
	std::cout << errorVec << std::endl;;
	std::cout << std::endl;
}

void FeedForwardNetwork::printJacobianMatrix(const arma::mat &jMat) {
	std::cout << "Jacobian Matrix:" << std::endl;
	std::cout<< jMat << std::endl;
	std::cout<<std::endl;
}

void FeedForwardNetwork::printWeights(const arma::colvec &weights) {
	std::cout << "Weights:" << std::endl;
	std::cout<< weights << std::endl;
	std::cout<<std::endl;	
}
